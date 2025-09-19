#!/usr/bin/env python3
"""
Fine-tune nanoGPT on Alpaca instruction-following dataset.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import time
from pathlib import Path

# Import shared components from model.py
from model import (
    create_model,
    load_checkpoint,
    get_window_size_blocks,
    Hyperparameters,
)

# Import our Alpaca dataset
from alpaca_dataset import AlpacaDataset, pad_to_block_size
from smart_batching import create_smart_dataloader


def print_with_timestamp(message):
    """Print message with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def evaluate_model(model, eval_loader, tokenizer, max_eval_steps=50):
    """Evaluate the model on a subset of data."""
    model.eval()
    total_loss = 0
    num_steps = 0
    
    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(eval_loader):
            if batch_idx >= max_eval_steps:
                break
                
            input_ids = input_ids.cuda().squeeze(0)
            labels = labels.cuda().squeeze(0)
            
            # Pad to block size
            input_ids = pad_to_block_size(input_ids)
            labels = pad_to_block_size(labels)
            
            # Forward pass
            loss = model(
                input_ids,
                labels,
                get_window_size_blocks(1000, 1000)  # Use fixed window for eval
            )
            
            # Normalize loss by sequence length (same as pretraining)
            loss = loss / len(input_ids)
            
            total_loss += loss.item()
            num_steps += 1
    
    model.train()
    return total_loss / max(num_steps, 1)


def generate_sample_response(model, tokenizer, prompt, max_new_tokens=50):
    """Generate a sample response from the model."""
    model.eval()
    
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
                probs = F.softmax(next_token_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to sequence
                if current_length < len(input_ids):
                    input_ids[current_length] = next_token
                    current_length += 1
                    
                    # Decode and check for stopping
                    gen_text = tokenizer.decode(next_token)
                    generated_tokens.append(gen_text)
                    
                    # Stop at newline or end
                    if "\n" in gen_text or next_token.item() == tokenizer.eos_token_id:
                        break
                else:
                    break
            else:
                break
    
    model.train()
    return "".join(generated_tokens)


def main():
    print_with_timestamp("🚀 Starting Alpaca Fine-tuning")
    print("=" * 60)
    
    # Configuration
    args = Hyperparameters()
    learning_rate = 1e-5  # Lower LR for more stable training
    num_epochs = 2
    save_every = 1000
    eval_every = 500
    
    # Find latest checkpoint
    checkpoint_path = None
    logs_dir = Path("logs")
    if logs_dir.exists():
        checkpoint_files = list(logs_dir.rglob("state_step*.pt"))
        if checkpoint_files:
            # Get the latest checkpoint
            checkpoint_path = max(checkpoint_files, key=lambda x: int(x.stem.split("step")[-1]))
            print_with_timestamp(f"Found latest checkpoint: {checkpoint_path}")
    
    if checkpoint_path is None:
        print_with_timestamp("⚠️  No checkpoint found. Please run pretraining first.")
        return
    
    # Create model
    print_with_timestamp("Creating model...")
    model = create_model(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()
    
    # Load pretrained checkpoint
    print_with_timestamp(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(model, str(checkpoint_path))
    
    # Set embedding layers to bfloat16
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    
    # Load tokenizer
    print_with_timestamp("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create smart batching dataloader with conservative sequence length
    print_with_timestamp("Creating smart batching dataloader...")
    train_loader = create_smart_dataloader(tokenizer, target_seq_len=8*1024)  # 4K instead of 16K
    
    # Create eval loader with smart batching too
    print_with_timestamp("Creating evaluation dataset with smart batching...")
    eval_base_dataset = AlpacaDataset(tokenizer, max_length=None, limit=1000)
    
    from smart_batching import SmartBatchDataset
    eval_smart_dataset = SmartBatchDataset(eval_base_dataset, target_seq_len=8*1024)  # 2K for eval
    eval_loader = DataLoader(eval_smart_dataset, batch_size=1, shuffle=False)
    
    print_with_timestamp(f"📊 Training batches: {len(train_loader)}")
    print_with_timestamp(f"📊 Evaluation batches: {len(eval_loader)}")
    
    # Setup optimizers (same as pretraining)
    from model import setup_optimizers
    optimizers = setup_optimizers(model, rank=0, world_size=1)
    
    # Override learning rates for finetuning
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * (learning_rate / 0.025)  # Scale relative to Muon's base LR
    
    # Training loop
    model.train()
    total_loss = 0
    step = 0
    best_eval_loss = float('inf')
    
    # Sample prompts for testing
    test_prompts = [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain photosynthesis in simple terms.\n\n### Response:\n"
    ]
    
    for epoch in range(num_epochs):
        print_with_timestamp(f"🔄 Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.cuda().squeeze(0)  # Remove dataloader batch dim
            labels = labels.cuda().squeeze(0)
            
            # Already flattened by smart batching, no need for view(-1)
            
            # Pad to block size
            input_ids = pad_to_block_size(input_ids)
            labels = pad_to_block_size(labels)
            
            # Forward pass
            loss = model(
                input_ids,
                labels,
                get_window_size_blocks(step, 10000)  # Use progressive window sizing
            )
            
            # Normalize loss by sequence length (same as pretraining)
            loss = loss / len(input_ids)
            
            # Backward pass
            loss.backward()
            
            # Step optimizers (same as pretraining)
            for opt in optimizers:
                opt.step()
            
            model.zero_grad(set_to_none=True)
            
            # Logging
            total_loss += loss.item()
            epoch_loss += loss.item()
            step += 1
            
            # Show GPU memory usage every batch
            if step % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                total_steps_in_epoch = len(train_loader)
                step_in_epoch = batch_idx + 1
                print_with_timestamp(f"Epoch {epoch + 1}, Step {step_in_epoch}/{total_steps_in_epoch} (Global {step}): Loss = {loss.item():.4f}, GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
            # Periodic evaluation
            if step % eval_every == 0:
                eval_loss = evaluate_model(model, eval_loader, tokenizer)
                print_with_timestamp(
                    f"Step {step}: Train Loss = {loss.item():.4f}, "
                    f"Eval Loss = {eval_loss:.4f}"
                )
                
                # Generate sample responses
                if step % (eval_every * 2) == 0:
                    print_with_timestamp("🧪 Sample generations:")
                    for i, prompt in enumerate(test_prompts[:1]):  # Just one sample
                        response = generate_sample_response(model, tokenizer, prompt)
                        print(f"  Prompt {i+1}: {prompt.split('### Response:')[0].split('### Instruction:')[1].strip()}")
                        print(f"  Response: {response.strip()}")
                        print()
                
                # Save if best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_path = f"alpaca_finetuned_best.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
                        "step": step,
                        "eval_loss": eval_loss,
                        "hyperparameters": args,
                    }, save_path)
                    print_with_timestamp(f"💾 Saved best model to {save_path}")
            
            # Periodic saving
            if step % save_every == 0:
                save_path = f"alpaca_finetuned_step_{step}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
                    "step": step,
                    "hyperparameters": args,
                }, save_path)
                print_with_timestamp(f"💾 Checkpoint saved to {save_path}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print_with_timestamp(f"📈 Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation and save
    final_eval_loss = evaluate_model(model, eval_loader, tokenizer)
    avg_total_loss = total_loss / step
    
    print_with_timestamp("🏁 Training completed!")
    print_with_timestamp(f"📊 Final average training loss: {avg_total_loss:.4f}")
    print_with_timestamp(f"📊 Final evaluation loss: {final_eval_loss:.4f}")
    
    # Save final model
    final_save_path = "alpaca_finetuned_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
        "step": step,
        "eval_loss": final_eval_loss,
        "hyperparameters": args,
    }, final_save_path)
    print_with_timestamp(f"💾 Final model saved to {final_save_path}")
    
    # Final test generation
    print_with_timestamp("\n🧪 Final test generations:")
    for i, prompt in enumerate(test_prompts):
        response = generate_sample_response(model, tokenizer, prompt, max_new_tokens=100)
        instruction = prompt.split('### Response:')[0].split('### Instruction:')[1].strip()
        print(f"Instruction: {instruction}")
        print(f"Response: {response.strip()}")
        print("-" * 40)


if __name__ == "__main__":
    main()