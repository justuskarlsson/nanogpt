import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import json
from pathlib import Path

# Import shared components from model.py
from model import (
    create_model,
    load_checkpoint,
    get_window_size_blocks,
    Hyperparameters,
)


class ConversationDataset(Dataset):
    """Dataset for fine-tuning on conversational data."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []

        # Load conversation data (expecting JSON format)
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data = json.load(f)
                self.conversations = data.get("conversations", [])
        else:
            print(f"Data file {data_path} not found. Using dummy data.")
            # Create some dummy conversation data for demonstration
            self.conversations = [
                {
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"},
                        {
                            "role": "assistant",
                            "content": "I'm doing well, thank you! How can I help you today?",
                        },
                    ]
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the capital of France?",
                        },
                        {
                            "role": "assistant",
                            "content": "The capital of France is Paris.",
                        },
                    ]
                },
            ]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]

        # Format conversation as a single text string
        text = ""
        for message in conversation["messages"]:
            role = message["role"]
            content = message["content"]
            if role == "user":
                text += f"Human: {content}\n"
            else:
                text += f"Assistant: {content}\n"

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (
                self.max_length - len(tokens)
            )

        # For causal language modeling, input and target are the same but shifted
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, labels


def pad_to_block_size(tensor, block_size=128):
    """Pad tensor to be a multiple of block_size."""
    current_length = tensor.size(-1)
    if current_length % block_size != 0:
        padding_length = block_size - (current_length % block_size)
        tensor = F.pad(tensor, (0, padding_length), value=0)
    return tensor


def main():
    # Configuration
    args = Hyperparameters()
    learning_rate = 5e-5
    num_epochs = 3
    batch_size = 1  # Keep small due to memory constraints
    data_path = "finetune_data.json"  # Path to your fine-tuning data
    checkpoint_path = (
        "logs/dab154c6-6fe0-4ff7-a6eb-897d6439936e/state_step004000.pt"
    )
    save_path = "finetuned_model.pt"

    print("🚀 Starting Fine-tuning")
    print("=" * 50)

    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
        print(f"✅ Loaded pretrained model from {checkpoint_path}")
    else:
        print(f"⚠️  Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model")

    # Set embedding layers to bfloat16
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and dataloader
    dataset = ConversationDataset(data_path, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"📊 Dataset size: {len(dataset)} conversations")

    # Setup optimizer (simpler than pretraining)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    total_loss = 0
    step = 0

    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.cuda()
            labels = labels.cuda()

            # Flatten batch dimension (model expects 1D input)
            input_ids = input_ids.squeeze(0)  # Remove batch dimension
            labels = labels.squeeze(0)

            # Pad to block size
            input_ids = pad_to_block_size(input_ids)
            labels = pad_to_block_size(labels)

            # Forward pass
            loss = model(
                input_ids,
                labels,
                get_window_size_blocks(
                    args.num_iterations, args.num_iterations
                ),
            )

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            total_loss += loss.item()
            epoch_loss += loss.item()
            step += 1

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"  📈 Average epoch loss: {avg_epoch_loss:.4f}")

    avg_total_loss = total_loss / step
    print(f"\n🏁 Training completed!")
    print(f"📊 Average loss: {avg_total_loss:.4f}")

    # Save fine-tuned model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hyperparameters": args,
        },
        save_path,
    )
    print(f"💾 Fine-tuned model saved to {save_path}")

    # Test the fine-tuned model
    print("\n🧪 Testing fine-tuned model:")
    model.eval()
    test_prompt = "Human: Hello, how are you?\nAssistant: "

    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").cuda()
    input_ids = input_ids.squeeze(0)
    input_ids = pad_to_block_size(input_ids)

    with torch.no_grad():
        logits = model(
            input_ids,
            None,
            get_window_size_blocks(args.num_iterations, args.num_iterations),
        )

        # Generate a few tokens
        last_idx = len(tokenizer.encode(test_prompt)) - 1
        generated = test_prompt

        for _ in range(20):
            next_token_logits = logits[0, last_idx, :]
            probs = F.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if last_idx + 1 < len(input_ids):
                input_ids[last_idx + 1] = next_token
                last_idx += 1

                gen_text = tokenizer.decode(next_token)
                generated += gen_text

                if "\n" in gen_text:
                    break

        print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
