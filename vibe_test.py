import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

# Import shared components from model.py
from model import (
    create_model,
    load_checkpoint,
    get_window_size_blocks,
    Hyperparameters,
)

prompts = [
    ("The capital of France is", 20),
    ("What is the capital of Germany?", 20),
    ("Be me, 24 year old programmer.", 30),
    ("There once was a", 50),
    ("Let me tell you a story about a cat.", 100),
    (
        "Key facts about Sweden Sweden is part of the Nordic region in northern Europe, together with the countries of Denmark, Finland, Iceland and Norway. Sweden is large in size, small in population. Capital:",
        100,
    ),
]


def main():
    args = Hyperparameters()

    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()

    # Load checkpoint
    checkpoint_path = "logs/_2025-06-09_18-45-23/state_step070000.pt"
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
        print("Model loaded from checkpoint")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        exit()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def pad_input(input_ids):
        BLOCK_SIZE = 128
        if len(input_ids) % BLOCK_SIZE != 0:
            padding_length = BLOCK_SIZE - (len(input_ids) % BLOCK_SIZE)
            input_ids = torch.nn.functional.pad(
                input_ids, (0, padding_length), value=tokenizer.eos_token_id
            )
        return input_ids

    for prompt, max_length in prompts:

        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.reshape(-1).cuda()

        # Pad to block size
        last_idx = len(input_ids) - 1
        input_ids = pad_input(input_ids)
        # Get logits
        model.eval()

        generated_tokens = []
        current_input = input_ids.clone()
        for i in range(max_length):
            with torch.no_grad():
                output = model(
                    pad_input(current_input),
                    None,
                    get_window_size_blocks(
                        args.num_iterations, args.num_iterations
                    ),
                )

            # Get logits for last position
            token_logits = output[0, last_idx + i, :50257]
            # Apply temperature
            temperature = 0.7
            token_logits = token_logits / temperature
            probs = torch.nn.functional.softmax(token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_tokens.append(next_token)

            # Update input for next iteration
            if last_idx + i + 1 < len(current_input):
                current_input[last_idx + i + 1] = next_token
            else:
                # Extend input sequence
                current_input = torch.cat(
                    [
                        current_input,
                        torch.tensor([next_token], device=current_input.device),
                    ]
                )

        # Print generated text
        full_text = prompt + tokenizer.decode(generated_tokens)
        print(full_text, "\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
