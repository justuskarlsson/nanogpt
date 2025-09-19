#!/usr/bin/env python3
"""
Alpaca Dataset preprocessing with instruction masking for finetuning.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import json
from typing import List, Dict, Tuple


class AlpacaDataset(Dataset):
    """Dataset for fine-tuning on Alpaca instruction-following data with proper masking."""

    def __init__(self, tokenizer, max_length: int = None, split: str = "train", limit: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load Alpaca dataset
        print(f"Loading Alpaca dataset ({split} split)...")
        dataset = load_dataset("tatsu-lab/alpaca", split=split)
        
        # Process and tokenize all examples
        self.examples = []
        print("Processing examples...")
        for i, example in enumerate(dataset):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(dataset)} examples...")
            if limit is not None and i >= limit:
                break
            
            processed = self._process_example(example)
            if processed is not None:
                self.examples.append(processed)
        
        print(f"Loaded {len(self.examples)} examples from Alpaca dataset")

    def _process_example(self, example: Dict) -> Dict | None:
        """Process a single Alpaca example into tokenized format with masking."""
        instruction = example["instruction"].strip()
        input_text = example["input"].strip() if example["input"] else ""
        output = example["output"].strip()
        
        # Format the prompt similar to Alpaca training format
        if input_text:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Combine prompt and response
        full_text = prompt + output
        
        # Tokenize the full text
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        
        # Tokenize just the prompt to find where response starts
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        # Skip if too long (only if max_length is set)
        if self.max_length is not None and len(full_tokens) > self.max_length:
            return None
            
        # Create input tokens (shifted for causal LM)
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        
        # Create mask: -100 for prompt tokens (no loss), target token ids for response tokens
        labels = [-100] * len(target_tokens)
        
        # Find where response starts (after prompt)
        response_start = len(prompt_tokens) - 1  # -1 because we shift by one for causal LM
        
        # Only compute loss on response tokens
        if response_start < len(labels):
            for i in range(response_start, len(labels)):
                labels[i] = target_tokens[i]
        
        # Only pad if max_length is specified
        if self.max_length is not None:
            pad_length = self.max_length - 1 - len(input_tokens)  # -1 for the shift
            if pad_length > 0:
                input_tokens.extend([self.tokenizer.pad_token_id] * pad_length)
                labels.extend([-100] * pad_length)
        
        return {
            "input_ids": input_tokens,
            "labels": labels,
            "prompt": prompt,
            "response": output
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = example["input_ids"]
        labels = example["labels"]
        
        # Add EOS token at the end for document separation
        input_ids = input_ids + [self.tokenizer.eos_token_id]
        labels = labels + [self.tokenizer.eos_token_id]
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )


def pad_to_block_size(tensor, block_size=128):
    """Pad tensor to be a multiple of block_size."""
    current_length = tensor.size(-1)
    if current_length % block_size != 0:
        padding_length = block_size - (current_length % block_size)
        if tensor.dtype == torch.long:
            # For token tensors, pad with tokenizer's pad token
            tensor = F.pad(tensor, (0, padding_length), value=0)
        else:
            # For loss masks, pad with -100
            tensor = F.pad(tensor, (0, padding_length), value=-100)
    return tensor


def create_alpaca_dataloader(tokenizer, batch_size=1, max_length=512, num_workers=0):
    """Create a DataLoader for Alpaca dataset."""
    from torch.utils.data import DataLoader
    
    dataset = AlpacaDataset(tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing Alpaca dataset...")
    dataset = AlpacaDataset(tokenizer, max_length=256)
    
    # Print first example
    example = dataset[0]
    print(f"Input shape: {example[0].shape}")
    print(f"Labels shape: {example[1].shape}")
    
    # Decode to verify
    input_ids = example[0]
    labels = example[1]
    
    # Find where labels start (not -100)
    response_start = None
    for i, label in enumerate(labels):
        if label != -100:
            response_start = i
            break
    
    if response_start is not None:
        prompt_text = tokenizer.decode(input_ids[:response_start+1])
        response_text = tokenizer.decode([l for l in labels[response_start:] if l != -100])
        
        print(f"\nPrompt: {prompt_text}")
        print(f"Response: {response_text}")
    else:
        print("No response tokens found (all masked)")