#!/usr/bin/env python3
"""
Download and examine the Alpaca dataset from HuggingFace.
"""

from datasets import load_dataset
import json

# Download the Alpaca dataset
print("Downloading tatsu-lab/alpaca dataset...")
dataset = load_dataset("tatsu-lab/alpaca")

# Examine the dataset structure
print(f"Dataset keys: {dataset.keys()}")
print(f"Train set size: {len(dataset['train'])}")

# Look at first few examples
print("\nFirst 3 examples:")
for i in range(3):
    example = dataset['train'][i]
    print(f"\nExample {i+1}:")
    for key, value in example.items():
        print(f"  {key}: {value}")

# Save a sample for inspection
print("\nSaving first 100 examples to alpaca_sample.json...")
sample = dataset['train'].select(range(100))
with open('alpaca_sample.json', 'w') as f:
    json.dump(sample.to_dict(), f, indent=2)

print("Done! Check alpaca_sample.json to see the data format.")