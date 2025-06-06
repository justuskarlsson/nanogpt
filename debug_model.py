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


def main():
    args = Hyperparameters()

    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()

    # Load checkpoint
    checkpoint_path = (
        "logs/dab154c6-6fe0-4ff7-a6eb-897d6439936e/state_step004000.pt"
    )
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
        print("Model loaded from checkpoint")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        exit()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Simple prompt
    prompt = "Human: What is the capital of France?\nAssistant:"

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.reshape(-1).cuda()

    # Pad to block size
    BLOCK_SIZE = 128
    last_idx = len(input_ids) - 1
    print(input_ids)
    if len(input_ids) % BLOCK_SIZE != 0:
        padding_length = BLOCK_SIZE - (len(input_ids) % BLOCK_SIZE)
        input_ids = torch.nn.functional.pad(
            input_ids, (0, padding_length), value=tokenizer.eos_token_id
        )

    # Get logits
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids,
            None,  # No targets for inference
            get_window_size_blocks(args.num_iterations, args.num_iterations),
        )

    # Extract logits for last token
    token_logits = logits[0, last_idx, :50257]  # Only use valid GPT-2 vocab

    # Get top tokens
    top_k = 20
    top_values, top_indices = torch.topk(token_logits, top_k)

    # Convert to probabilities
    top_probs = F.softmax(top_values, dim=0)

    # Print token distribution
    print(f"Top {top_k} tokens for prompt: '{prompt}'")
    print("-" * 50)
    print("Token\tID\tLogit\tProb\tText")
    print("-" * 50)

    for i in range(top_k):
        token_id = top_indices[i].item()
        token_text = tokenizer.decode([token_id])
        token_text = token_text.replace("\n", "\\n")  # Make newlines visible
        print(
            f"{i+1}\t{token_id}\t{top_values[i]:.2f}\t{top_probs[i]:.4f}\t'{token_text}'"
        )

    # Show full distribution statistics
    all_probs = F.softmax(token_logits, dim=0)
    print("\nProbability distribution stats:")
    print(f"Max prob: {all_probs.max().item():.6f}")
    print(f"Min prob: {all_probs.min().item():.6f}")
    print(f"Mean prob: {all_probs.mean().item():.6f}")
    print(
        f"Entropy: {(-all_probs * torch.log2(all_probs+1e-10)).sum().item():.2f} bits"
    )

    # Generate some text
    print("\nGenerated text:")
    generated_tokens = []
    current_input = input_ids.clone()

    for i in range(30):  # Generate 30 tokens
        with torch.no_grad():
            output = model(
                current_input,
                None,
                get_window_size_blocks(
                    args.num_iterations, args.num_iterations
                ),
            )

        # Get logits for last position
        token_logits = output[0, last_idx + i, :50257]

        # Apply temperature
        token_logits = token_logits / 0.7

        # Get probabilities
        probs = F.softmax(token_logits, dim=0)

        # Sample
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
    print(full_text)

    # Print token IDs and their decoded values
    print("\nToken IDs and their decoded values:")
    for token_id in generated_tokens[:10]:  # Show first 10 tokens
        token_text = tokenizer.decode([token_id])
        token_text = token_text.replace("\n", "\\n")
        print(f"ID: {token_id}, Text: '{token_text}'")


if __name__ == "__main__":
    main()
