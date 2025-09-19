#!/usr/bin/env python3
"""
Interactive chat interface for the fine-tuned Alpaca model.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys

from model import create_model, Hyperparameters, get_window_size_blocks
from alpaca_dataset import pad_to_block_size


def load_model():
    """Load the fine-tuned model."""
    print("Loading model...")

    # Create model
    args = Hyperparameters()
    model = create_model(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()

    # Try to load the best checkpoint first
    checkpoint_paths = ["alpaca_finetuned_best.pt", "alpaca_finetuned_final.pt"]
    checkpoint_path = None

    for path in checkpoint_paths:
        try:
            print(f"Trying to load: {path}")
            checkpoint = torch.load(path, map_location="cuda", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            checkpoint_path = path
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    if checkpoint_path is None:
        print("❌ No fine-tuned model found! Please run fine-tuning first.")
        sys.exit(1)

    print(f"✅ Loaded model from: {checkpoint_path}")
    model.eval()
    return model


def generate_response(model, tokenizer, prompt, max_new_tokens=150, temperature=0.8):
    """Generate a response from the model."""
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda().squeeze(0)
    input_ids = pad_to_block_size(input_ids)

    generated_tokens = []
    current_length = len(tokenizer.encode(prompt))

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits
            logits = model(
                input_ids,
                None,
                get_window_size_blocks(1000, 1000)
            )

            # Sample next token
            if current_length - 1 < logits.size(1):
                next_token_logits = logits[0, current_length - 1, :]
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Add to sequence
                if current_length < len(input_ids):
                    input_ids[current_length] = next_token
                    current_length += 1

                    # Decode and check for stopping
                    gen_text = tokenizer.decode(next_token)
                    generated_tokens.append(gen_text)

                    # Stop at EOS token
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                else:
                    break
            else:
                break

    return "".join(generated_tokens).strip()


def main():
    print("🤖 NanoGPT Alpaca Chat Interface")
    print("=" * 50)
    print("Loading model and tokenizer...")

    # Load model and tokenizer
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\n🎉 Ready to chat!")
    print("\nInstructions:")
    print("- Type your questions or instructions")
    print("- Type 'quit', 'exit', or 'q' to exit")
    print("- Type 'temp <value>' to change temperature (0.1-2.0)")
    print("- Type 'help' for this message")
    print("\nExample: What is the capital of France?")
    print("-" * 50)

    temperature = 0.3

    # Add system instruction to the beginning
    system_instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"

    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()

            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            elif user_input.lower() == 'help':
                print("\nInstructions:")
                print("- Type your questions or instructions")
                print("- Type 'quit', 'exit', or 'q' to exit")
                print("- Type 'temp <value>' to change temperature (0.1-2.0)")
                print("- Type 'help' for this message")
                continue

            elif user_input.lower().startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"🌡️  Temperature set to {temperature}")
                    else:
                        print("❌ Temperature must be between 0.1 and 2.0")
                except:
                    print("❌ Invalid temperature format. Use: temp 0.8")
                continue

            elif not user_input:
                continue

            # Format the prompt
            prompt = f"{system_instruction}### Instruction:\n{user_input}\n\n### Response:\n"

            # Generate response
            print("🤖 Assistant: ", end="", flush=True)
            try:
                response = generate_response(model, tokenizer, prompt, temperature=temperature)

                # Clean up response (remove any remaining special tokens)
                response = response.replace("<|endoftext|>", "").strip()

                if response:
                    print(response)
                else:
                    print("(No response generated)")

            except Exception as e:
                print(f"❌ Error generating response: {e}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()