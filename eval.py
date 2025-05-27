import os
import torch
from transformers import AutoTokenizer

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

    # Load checkpoint (you'll need to update this path)
    checkpoint_path = (
        "logs/dab154c6-6fe0-4ff7-a6eb-897d6439936e/state_step004000.pt"
    )
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
        print("Model loaded from checkpoint")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model")

    # Set embedding layers to bfloat16
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Interactive evaluation
    print("Model ready for evaluation!")
    print("Enter 'quit' to exit")

    while True:
        prompt = input("\nEnter a prompt: ")
        if prompt.lower() == "quit":
            break

        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.reshape(-1).cuda()

        # Pad to block size
        BLOCK_SIZE = 128
        last_idx = len(input_ids) - 1
        if len(input_ids) % BLOCK_SIZE != 0:
            padding_length = BLOCK_SIZE - (len(input_ids) % BLOCK_SIZE)
            input_ids = torch.nn.functional.pad(
                input_ids, (0, padding_length), value=tokenizer.pad_token_id
            )

        # Generate text
        generated_text = prompt
        model.eval()

        with torch.no_grad():
            for i in range(64):  # Generate 64 tokens
                logits = model(
                    input_ids,
                    None,  # No targets for inference
                    get_window_size_blocks(
                        args.num_iterations, args.num_iterations
                    ),
                )

                # Get next token probabilities and sample
                next_token_logits = logits[0, last_idx, :]
                probs = torch.nn.functional.softmax(
                    next_token_logits / 0.8, dim=-1
                )
                next_token = torch.multinomial(probs, num_samples=1)

                # Update input
                input_ids[last_idx + 1] = next_token
                last_idx += 1

                # Decode and add to generated text
                gen_text = tokenizer.decode(next_token)
                generated_text += gen_text

                # Stop if we hit end of sequence or padding
                if next_token.item() in [
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                ]:
                    break

        print(f"\nGenerated: {generated_text}")


if __name__ == "__main__":
    main()
