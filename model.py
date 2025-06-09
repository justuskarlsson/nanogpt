import os
import glob
import random
import time
import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention


# -----------------------------------------------------------------------------
# Muon optimizer


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def update(
    acc_bf16_view_u16: Tensor,
    mantissa: Tensor,
    momentum_buffer: Tensor,
    grad: Tensor,
    momentum: Tensor,
    eff_lr: Tensor,
    eff_weight_decay: Tensor,
):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(
        momentum * momentum_buffer + (1 - momentum) * grad
    )

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(
        torch.uint32
    )
    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        assert all(
            p.dtype == torch.bfloat16
            for group in self.param_groups
            for p in group["params"]
        )

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = (
                params + [torch.empty_like(params[-1])] * self.world_size
            )
            momentum = torch._as_tensor_fullprec(group["momentum"])
            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["mantissa"] = torch.zeros_like(
                            p, dtype=torch.uint16
                        )
                        state["momentum_buffer"] = torch.zeros_like(
                            p, dtype=torch.float32
                        )
                    update(
                        p.view(torch.uint16),
                        state["mantissa"],
                        state["momentum_buffer"],
                        p.grad,
                        momentum,
                        eff_lr=torch._as_tensor_fullprec(
                            group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5
                        ),
                        eff_weight_decay=torch._as_tensor_fullprec(
                            group["lr"]
                            * group["weight_decay"]
                            * getattr(p, "wd_mul", 1.0)
                        ),
                    )
                futures.append(
                    dist.all_gather(
                        params_pad[base_i : base_i + self.world_size],
                        params_pad[base_i + self.rank],
                        async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(futures).wait()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


@torch.no_grad()
def init_linear(w: Tensor):
    std = 0.5 * (
        w.size(-1) ** -0.5
    )  # 0.5 is a bit better than the default 1/sqrt(3)
    bound = (3**0.5) * std
    return w.uniform_(-bound, bound)


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat(
            [angular_freq, angular_freq.new_zeros(dim // 4)]
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = (
            self.cos[None, : x_BTHD.size(-3), None, :],
            self.sin[None, : x_BTHD.size(-3), None, :],
        )
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, max_seq_len: int, head_dim=128
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkvo_w = nn.Parameter(
            init_linear(torch.empty(4, hdim, dim)).bfloat16()
        )
        self.qkvo_w.detach()[
            3
        ].zero_()  # out zero init suggested by @Grad62304977
        self.rotary = Rotary(head_dim, max_seq_len)
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        block_mask: BlockMask,
        lambdas: Tensor,
    ):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = (
            F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        v = norm(v)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(
                v
            )  # @KoszarskyB & @Grad62304977
        else:  # skip mid-layers token value embeddings by @YouJiacheng
            v = lambdas[0] * v
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            scale=self.attn_scale,
        ).transpose(1, 2)
        y = y.contiguous().view(
            B, T, self.num_heads * self.head_dim
        )  # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w[3])
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc_w = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.proj_w = nn.Parameter(torch.zeros(dim, hdim).bfloat16())
        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0

    def forward(self, x: Tensor):
        x = F.linear(x, self.fc_w)
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.proj_w)
        return x


class Block(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int
    ):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = (
            CausalSelfAttention(dim, num_heads, max_seq_len)
            if layer_idx != 7
            else None
        )
        self.mlp = MLP(dim)

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        x0: Tensor,
        block_mask: BlockMask,
        lambdas: Tensor,
        sa_lambdas: Tensor,
    ):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, block_mask, sa_lambdas)
        x = x + self.mlp(norm(x))
        return x


# -----------------------------------------------------------------------------
# The main model


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList(
            [nn.Embedding(vocab_size, model_dim) for _ in range(3)]
        )
        self.blocks = nn.ModuleList(
            [
                Block(model_dim, num_heads, max_seq_len, i)
                for i in range(num_layers)
            ]
        )
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head_w = nn.Parameter(
            torch.zeros(next_multiple_of_n(vocab_size, n=128), model_dim)
        )
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    torch.ones(num_layers),  # skip_weights
                    *[
                        torch.tensor([1.0, 0.0]) for _ in range(num_layers)
                    ],  # block lambdas
                    *[
                        torch.tensor([0.5, 0.5]) for _ in range(num_layers)
                    ],  # SA lambdas
                ]
            )
        )

    def create_blockmasks(
        self, input_seq: Tensor, sliding_window_num_blocks: Tensor
    ):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = (
                dense_blockmask.int()
                .argsort(dim=-1, descending=False, stable=True)
                .flip(-1)
                .to(torch.int32)
            )
            return (
                num_blocks[None, None].contiguous(),
                indices[None, None].contiguous(),
            )

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (
            docs_high[:, None] >= docs_low
        )
        document_blockmask_all = (docs_low[:, None] == docs_high) & (
            docs_high[:, None] == docs_low
        )
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(
            blockmask_any & ~blockmask_all
        )
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(
                    partial_kv_num_blocks,
                    torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1),
                ),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(
            sliding_window_num_blocks // 2
        )

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor | None,
        sliding_window_num_blocks: Tensor,
    ):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = (
            [ve[0], ve[1], ve[2]]
            + [None] * (len(self.blocks) - 6)
            + [ve[0], ve[1], ve[2]]
        )
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(
            input_seq, sliding_window_num_blocks
        )
        block_masks = [
            long_bm,
            short_bm,
            short_bm,
            short_bm,
            long_bm,
            short_bm,
            short_bm,
            short_bm,
            short_bm,
            short_bm,
            short_bm,
            long_bm,
            short_bm,
            short_bm,
            short_bm,
            long_bm,
        ]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(
            self.embed(input_seq)[None]
        )  # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        skip_map = {
            9: 6,
            10: 4,
            11: 2,
        }
        skip_weights = self.scalars[: len(self.blocks)]
        lambdas = self.scalars[
            1 * len(self.blocks) : 3 * len(self.blocks)
        ].view(-1, 2)
        sa_lambdas = self.scalars[
            3 * len(self.blocks) : 5 * len(self.blocks)
        ].view(-1, 2)
        for i in range(len(self.blocks)):
            if i in skip_map:
                x = (
                    x
                    + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
                )
            x = self.blocks[i](
                x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i]
            )
            skip_connections.append(x)

        x = norm(x)
        if target_seq is None:
            logits: Tensor = F.linear(
                x.flatten(end_dim=1), self.lm_head_w.bfloat16()
            ).float()
            return logits
        if self.training:
            logits: Tensor = F.linear(
                x.flatten(end_dim=1), self.lm_head_w.bfloat16()
            ).float()
            loss = F.cross_entropy(
                15 * logits * torch.rsqrt(logits.square() + 225), target_seq
            )
            return loss
        else:
            loss = 0
            for i in range(4):
                logits: Tensor = F.linear(
                    x.flatten(end_dim=1).chunk(4)[i], self.lm_head_w.bfloat16()
                ).float()
                loss += (
                    F.cross_entropy(
                        15 * logits * torch.rsqrt(logits.square() + 225),
                        target_seq.chunk(4)[i],
                    )
                    / 4
                )
            return loss


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _load_data_shard(file: Path):
    header = torch.from_file(
        str(file), False, 256, dtype=torch.int32
    )  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=True
        )  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(
            tokens.numpy()
        )  # avoid bytes->array copy by @YouJiacheng
        assert (
            nbytes == 2 * num_tokens
        ), "number of tokens read does not match header"
    return tokens


last_val = 0


def distributed_data_generator(
    batch_size: int, rank: int, world_size: int, is_train: bool
):
    mode = "train" if is_train else "val"
    globs = [
        sorted(glob.glob(f"data/finewebedu10B/finewebedu_{mode}_*.bin"))[:50],
        sorted(glob.glob(f"data/fineweb10B/fineweb_{mode}_*.bin"))[:10],
    ]
    files = [Path(file) for file in sum(globs, [])]
    if is_train:
        # Interleave for all
        files = []
        for i in range(10):
            files += globs[0][i * 5 : (i + 1) * 5]
            files += globs[1][i : i + 1]
        files = [Path(file) for file in files]
        if rank == 0:
            print(f"Training files:", files)
        # Shuffle, each node has different seed
        # Works like shit, not good.
        # random.seed(rank)
        # random.shuffle(files)
        # print(f"Validation files ({rank}):", files)
    else:
        # We only have 2 val file (edu or regular)
        global last_val
        last_val = 0 if last_val == 1 else 1
        if rank == 0:
            print("FineWebEdu" if last_val == 0 else "FineWeb", "validation")
        files = [files[last_val]]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(
        files
    )  # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size :][: local_batch_size + 1]
        inputs = buf[:-1].to(
            device="cuda", dtype=torch.int32, non_blocking=True
        )  # no sync on host side;
        targets = buf[1:].to(
            device="cuda", dtype=torch.int64, non_blocking=True
        )  # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets


# -----------------------------------------------------------------------------
# Utility functions


@dataclass
class Hyperparameters:
    # data
    val_tokens = 10485760  # 10M
    train_seq_len = 32 * 1024  # 128 K per step
    val_seq_len = 4 * 32 * 1024
    # optimization
    # 128 M / 1 k steps.
    # Target: ish 6 B tokens
    # 60k ->  50k steps
    # cmp with medium_org (6000 steps -> 24k equiv)
    # PROD
    num_iterations = 50000  # number of iterations to run
    val_loss_every = 500
    # DEV
    # num_iterations = 200  # number of iterations to run
    # val_loss_every = 50
    # 0.5 s per step -> 30k steps -> 30k s -> 4 h
    cooldown_frac = 0.8  # 0.7 -> 0.8 because of more tokens
    # architecture
    vocab_size = 50257
    # evaluation and logging
    save_checkpoint = True


# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(
        window_size // 128, dtype=torch.int32, pin_memory=True
    ).cuda(non_blocking=True)


def get_window_size_blocks(step: int, num_iterations: int):
    x = step / num_iterations  # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    factor = 4 * x**3 - 6 * x**2 + 3 * x
    window_size = next_multiple_of_n(3456 * factor, n=128)
    return get_window_size_blocks_helper(window_size)


def create_model(vocab_size: int = 50257, max_seq_len: int = 256 * 1024) -> GPT:
    """Create a 130M GPT-2 model with default parameters."""
    return GPT(
        vocab_size=vocab_size,
        num_layers=16,
        num_heads=8,
        model_dim=1024,
        max_seq_len=max_seq_len,
    )


def setup_optimizers(model: GPT, rank: int = 0, world_size: int = 1):
    """Setup optimizers for the model."""
    # collect the parameters to optimize
    hidden_matrix_params = sorted(
        (p for p in model.blocks.parameters() if p.ndim >= 2),
        key=lambda x: x.size(),
        reverse=True,
    )
    embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
    scalar_params = [model.scalars]
    head_params: list[nn.Parameter] = [model.lm_head_w]
    # sanity check
    params_collections = [
        hidden_matrix_params,
        embed_params,
        scalar_params,
        head_params,
    ]
    optimized_parameters_set = {
        p for params in params_collections for p in params
    }
    assert optimized_parameters_set == {*model.parameters()}
    assert len(optimized_parameters_set) == sum(
        len(lst) for lst in params_collections
    )

    # init the optimizer(s)
    adam_param_groups = [
        dict(params=head_params, lr=1 / 320),
        dict(params=embed_params, lr=0.3),
        dict(params=scalar_params, lr=0.015),
    ]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.AdamW(
        adam_param_groups,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=True,
    )
    optimizer2 = Muon(
        hidden_matrix_params,
        lr=0.025,
        momentum=0.95,
        rank=rank,
        world_size=world_size,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer1, optimizer2]
    return optimizers


def get_lr(step: int, num_iterations: int, cooldown_frac: float = 0.4):
    """Learning rate schedule: stable then decay."""
    x = step / num_iterations  # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac


def load_checkpoint(model: GPT, checkpoint_path: str):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cuda", weights_only=True
    )

    def strip_prefix_if_present(state_dict, prefix):
        return {
            k[len(prefix) :] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }

    state_dict = strip_prefix_if_present(checkpoint["model"], "_orig_mod.")
    model.load_state_dict(state_dict)
    return checkpoint
