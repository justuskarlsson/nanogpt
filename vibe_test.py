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

# Instruction-following prompts for finetuned models
instruction_prompts = [
    ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n", 50),
    ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain photosynthesis in simple terms.\n\n### Response:\n", 150),
    ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a short poem about the ocean.\n\n### Response:\n", 100),
    ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nList three benefits of exercise.\n\n### Response:\n", 80),
    ("Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nTranslate the following text to Spanish.\n\n### Input:\nHello, how are you today?\n\n### Response:\n", 50),
]


def main():
    import sys
    args = Hyperparameters()

    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()

    # Auto-detect model type and checkpoint
    model_type = "pretrained"
    checkpoint_path = None
    
    # Check for finetuned models first
    finetuned_paths = [
        "alpaca_finetuned_best.pt",
        "alpaca_finetuned_final.pt",
    ]
    
    for path in finetuned_paths:
        if os.path.exists(path):
            checkpoint_path = path
            model_type = "finetuned"
            break
    
    # Fall back to pretrained checkpoint
    if checkpoint_path is None:
        # Look for latest pretrained checkpoint
        import glob
        pretrained_checkpoints = glob.glob("logs/*/state_step*.pt")
        if pretrained_checkpoints:
            # Get the latest one
            checkpoint_path = max(pretrained_checkpoints, 
                                key=lambda x: int(x.split("step")[-1].split(".")[0]))
            model_type = "pretrained"
    
    if checkpoint_path is None:
        print("No checkpoint found. Please run pretraining or finetuning first.")
        sys.exit(1)
    
    print(f"Loading {model_type} model from: {checkpoint_path}")
    
    if model_type == "finetuned":
        # Load finetuned model
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Finetuned model loaded")
    else:
        # Load pretrained model
        load_checkpoint(model, checkpoint_path)
        print("Pretrained model loaded")
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

    # Choose prompts based on model type
    test_prompts = instruction_prompts if model_type == "finetuned" else prompts
    print(f"\nTesting {model_type} model with {len(test_prompts)} prompts:")
    print("=" * 60)

    for prompt, max_length in test_prompts:

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

        # Print generated text with better formatting for instructions
        if model_type == "finetuned" and "### Instruction:" in prompt:
            instruction = prompt.split("### Instruction:")[1].split("### Response:")[0].strip()
            if "### Input:" in instruction:
                parts = instruction.split("### Input:")
                task = parts[0].strip()
                context = parts[1].strip()
                print(f"Task: {task}")
                print(f"Input: {context}")
            else:
                print(f"Task: {instruction}")
            
            response = tokenizer.decode(generated_tokens).strip()
            print(f"Response: {response}")
        else:
            full_text = prompt + tokenizer.decode(generated_tokens)
            print(full_text)
        
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
