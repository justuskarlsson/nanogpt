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


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


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

    # Interactive chat
    print("\n" + "=" * 50)
    print("🤖 GPT Chat Interface")
    print("=" * 50)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'clear' - Clear the screen")
    print("  'new' - Start a new conversation")
    print("=" * 50)

    conversation = ""

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        elif user_input.lower() == "clear":
            clear_screen()
            continue
        elif user_input.lower() == "new":
            conversation = ""
            print("🔄 Started new conversation")
            continue
        elif not user_input:
            continue

        # Add user input to conversation
        conversation += f"Human: {user_input}\nAssistant: "

        # Tokenize input
        input_ids = tokenizer(conversation, return_tensors="pt").input_ids
        input_ids = input_ids.reshape(-1).cuda()

        # Pad to block size
        BLOCK_SIZE = 128
        last_idx = len(input_ids) - 1
        if len(input_ids) % BLOCK_SIZE != 0:
            padding_length = BLOCK_SIZE - (len(input_ids) % BLOCK_SIZE)
            input_ids = torch.nn.functional.pad(
                input_ids, (0, padding_length), value=tokenizer.pad_token_id
            )

        # Generate response
        print("🤖 Assistant: ", end="", flush=True)
        model.eval()
        response = ""

        with torch.no_grad():
            for i in range(128):  # Generate up to 128 tokens
                logits = model(
                    input_ids,
                    None,  # No targets for inference
                    get_window_size_blocks(
                        args.num_iterations, args.num_iterations
                    ),
                )

                # Get next token probabilities and sample
                next_token_logits = logits[0, last_idx, :]
                # Use temperature sampling for more interesting responses
                probs = torch.nn.functional.softmax(
                    next_token_logits / 0.7, dim=-1
                )
                next_token = torch.multinomial(probs, num_samples=1)

                # Update input
                if last_idx + 1 < len(input_ids):
                    input_ids[last_idx + 1] = next_token
                    last_idx += 1
                else:
                    # Need to extend the sequence
                    break

                # Decode token
                gen_text = tokenizer.decode(next_token)
                response += gen_text
                print(gen_text, end="", flush=True)

                # Stop on end of sequence, newlines (for chat), or specific stop tokens
                if (
                    next_token.item()
                    in [tokenizer.eos_token_id, tokenizer.pad_token_id]
                    or "\n" in gen_text
                    or "Human:" in response
                ):
                    break

        print()  # New line after response

        # Clean up response and add to conversation
        response = response.split("Human:")[
            0
        ].strip()  # Remove any "Human:" that might appear
        response = response.split("\n")[0].strip()  # Take only first line
        conversation += response + "\n"

        # Truncate conversation if it gets too long
        if len(conversation) > 2000:
            # Keep last part of conversation
            conversation = conversation[-1500:]
            # Find a good breaking point
            human_idx = conversation.find("Human:")
            if human_idx > 0:
                conversation = conversation[human_idx:]


if __name__ == "__main__":
    main()
