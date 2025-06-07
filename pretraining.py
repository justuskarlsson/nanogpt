import os
import sys
import uuid
import time
import dotenv

# This script has been modified to train a ~1B parameter model instead of the original ~130M model
# Sequence length has been reduced and gradient accumulation has been added to accommodate the larger model
# on 4x A100 40GB GPUs

# Load environment variables first
dotenv.load_dotenv()

# Set PyTorch CUDA allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist

# Import all shared components from model.py
from model import (
    GPT,
    Hyperparameters,
    distributed_data_generator,
    get_window_size_blocks,
    setup_optimizers,
    get_lr,
    create_model,
)

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

torch.empty(1, device="cuda", requires_grad=True).backward()

# Initialize hyperparameters
args = Hyperparameters()


def main():

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert torch.cuda.is_available()

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = (
        rank == 0
    )  # this process will do logging, checkpointing etc.

    # Begin logging
    logfile = None
    if master_process:
        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)

    def print0(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)

    # Log information about the hardware/software environment
    print0(f"Running Python {sys.version}")
    print0(
        f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
    )

    def nvidia_smi():
        import subprocess

        return subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout

    print0(nvidia_smi())
    print0("=" * 100)

    ########################################
    #    Construct model and optimizer     #
    ########################################

    model = create_model(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()

    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()

    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    # Setup optimizers
    optimizers = setup_optimizers(model, rank, world_size)

    # Learning rate schedule
    def get_lr_for_step(step: int):
        return get_lr(step, args.num_iterations, args.cooldown_frac)

    # Compile model
    model = torch.compile(model, dynamic=False)

    ########################################
    #        Training and validation       #
    ########################################

    train_loader = distributed_data_generator(
        world_size * args.train_seq_len, rank, world_size, is_train=True
    )
    training_time_ms = 0

    # Start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Begin training
    train_steps = args.num_iterations
    print(f"Starting training for {train_steps} steps")

    for step in range(train_steps + 1):
        last_step = step == train_steps

        # --------------- VALIDATION SECTION -----------------
        if last_step or (
            args.val_loss_every > 0
            and step % args.val_loss_every == 0
            and step != 0
        ):
            # Stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = int(args.val_tokens // val_batch_size)
            val_loader = distributed_data_generator(
                val_batch_size, rank, world_size, is_train=False
            )
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(
                        inputs,
                        targets,
                        get_window_size_blocks(step, args.num_iterations),
                    )
            val_loss /= val_steps
            del val_loader
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            time_s = training_time_ms / 1000
            time_m = int(time_s / 60)
            time_s = time_s % 60
            print0(
                f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{time_m}m {time_s:.2f}s step_avg:{training_time_ms/max(step, 1):.2f}ms",
                console=True,
            )
            model.train()
            # Start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(
                    step=step,
                    model=model.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers],
                )
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            # The last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        model(
            inputs, targets, get_window_size_blocks(step, args.num_iterations)
        ).backward()

        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        # Set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr_for_step(step)

        # Momentum warmup for Muon optimizer
        for group in optimizers[1].param_groups:  # optimizers[1] is Muon
            frac = min(step / 300, 1)
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        # Step the optimizers
        for opt in optimizers:
            opt.step()

        # Null the gradients
        model.zero_grad(set_to_none=True)

    print0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
        console=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
