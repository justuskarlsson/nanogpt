#!/usr/bin/env python3
"""
Smart batching for variable-length sequences that maximizes GPU utilization
by packing sequences to target length without padding waste.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple
from alpaca_dataset import AlpacaDataset


class SmartBatchDataset(Dataset):
    """Dataset that creates batches with target total sequence length."""
    
    def __init__(self, base_dataset: AlpacaDataset, target_seq_len: int = 16*1024, margin: int = 256):
        self.base_dataset = base_dataset
        self.target_seq_len = target_seq_len
        self.margin = margin  # Safety margin for block padding
        
        # Preload all examples into memory (dataset is tiny)
        print("Preloading entire dataset into memory...")
        self.preloaded_examples = []
        for i in range(len(self.base_dataset)):
            self.preloaded_examples.append(self.base_dataset[i])
        print(f"Preloaded {len(self.preloaded_examples)} examples")
        
        self.batches = self._create_batches()
        
    def _create_batches(self) -> List[List[int]]:
        """Pack examples into batches that sum to approximately target_seq_len."""
        print(f"Creating smart batches with target length {self.target_seq_len}...")
        
        # Get all example lengths from preloaded data
        examples_with_lengths = []
        for i, (input_ids, labels) in enumerate(self.preloaded_examples):
            length = len(input_ids)
            examples_with_lengths.append((i, length))
        
        # Sort by length for better packing (optional)
        examples_with_lengths.sort(key=lambda x: x[1])
        
        batches = []
        current_batch = []
        current_length = 0
        
        for idx, length in examples_with_lengths:
            # Check if adding this example would exceed target
            if current_length + length + self.margin > self.target_seq_len and current_batch:
                # Finalize current batch
                batches.append(current_batch)
                current_batch = [idx]
                current_length = length
            else:
                # Add to current batch
                current_batch.append(idx)
                current_length += length
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        # Shuffle batches for training
        random.shuffle(batches)
        
        print(f"Created {len(batches)} smart batches")
        print(f"Average examples per batch: {sum(len(b) for b in batches) / len(batches):.1f}")
        print(f"Sample batch sizes: {[len(b) for b in batches[:5]]}")
        
        return batches
    
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a concatenated batch of examples from preloaded memory."""
        batch_indices = self.batches[idx]
        
        all_input_ids = []
        all_labels = []
        
        for example_idx in batch_indices:
            input_ids, labels = self.preloaded_examples[example_idx]
            all_input_ids.extend(input_ids.tolist())
            all_labels.extend(labels.tolist())
        
        return (
            torch.tensor(all_input_ids, dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.long)
        )


def create_smart_dataloader(tokenizer, target_seq_len: int = 16*1024, num_workers: int = 0):
    """Create a DataLoader with smart batching."""
    
    # Create base dataset (without max_length truncation)
    base_dataset = AlpacaDataset(tokenizer, max_length=None)  # No truncation
    
    # Create smart batching dataset
    smart_dataset = SmartBatchDataset(base_dataset, target_seq_len=target_seq_len)
    
    # DataLoader with batch_size=1 since batching is handled internally
    dataloader = DataLoader(
        smart_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the smart batching
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing smart batching...")
    dataloader = create_smart_dataloader(tokenizer, target_seq_len=2048)
    
    for i, (input_ids, labels) in enumerate(dataloader):
        if i >= 3:  # Show first 3 batches
            break
        
        input_ids = input_ids.squeeze(0)
        labels = labels.squeeze(0)
        
        print(f"Batch {i+1}:")
        print(f"  Length: {len(input_ids)} tokens")
        print(f"  After block padding: {((len(input_ids) + 127) // 128) * 128} tokens")
        print(f"  Efficiency: {len(input_ids) / (((len(input_ids) + 127) // 128) * 128) * 100:.1f}%")
        print()