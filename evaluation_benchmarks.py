#!/usr/bin/env python3
"""
Evaluation benchmarks for instruction-following models.
Includes SWAG, HellaSwag, and other common sense reasoning tasks.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Tuple
import json

from model import create_model, load_checkpoint, get_window_size_blocks, Hyperparameters
from alpaca_dataset import pad_to_block_size


def load_model_and_tokenizer(checkpoint_path: str = None):
    """Load model and tokenizer."""
    args = Hyperparameters()
    
    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()
    
    # Auto-detect checkpoint if not provided
    if checkpoint_path is None:
        import os
        import glob
        
        # Check for finetuned models first
        finetuned_paths = ["alpaca_finetuned_best.pt", "alpaca_finetuned_final.pt"]
        for path in finetuned_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        # Fall back to pretrained
        if checkpoint_path is None:
            pretrained_checkpoints = glob.glob("logs/*/state_step*.pt")
            if pretrained_checkpoints:
                checkpoint_path = max(pretrained_checkpoints, 
                                    key=lambda x: int(x.split("step")[-1].split(".")[0]))
    
    if checkpoint_path is None:
        raise ValueError("No checkpoint found")
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model
    if "alpaca_finetuned" in checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        load_checkpoint(model, checkpoint_path)
    
    # Set embeddings to bfloat16
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_logits_for_choices(model, tokenizer, context: str, choices: List[str], max_length: int = 512):
    """Get logits for multiple choice answers."""
    model.eval()
    choice_logprobs = []
    
    with torch.no_grad():
        for choice in choices:
            # Create full text
            full_text = context + choice
            
            # Tokenize
            tokens = tokenizer.encode(full_text, max_length=max_length, truncation=True)
            context_tokens = tokenizer.encode(context, max_length=max_length, truncation=True)
            
            # Skip if choice is too short or context is too long
            if len(tokens) <= len(context_tokens) or len(context_tokens) >= max_length - 10:
                choice_logprobs.append(-float('inf'))
                continue
            
            # Convert to tensor and pad
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long).cuda()
            target_ids = torch.tensor(tokens[1:], dtype=torch.long).cuda()
            
            input_ids = pad_to_block_size(input_ids)
            target_ids = pad_to_block_size(target_ids)
            
            # Get logits
            logits = model(
                input_ids,
                None,
                get_window_size_blocks(1000, 1000)
            )
            
            # Calculate log probability for the choice tokens only
            choice_start = len(context_tokens) - 1  # -1 for shift
            choice_end = len(tokens) - 1  # -1 for shift
            
            if choice_start >= 0 and choice_end > choice_start and choice_end <= len(target_ids):
                choice_logits = logits[0, choice_start:choice_end, :]
                choice_targets = target_ids[choice_start:choice_end]
                
                # Get log probabilities
                log_probs = F.log_softmax(choice_logits, dim=-1)
                choice_log_prob = log_probs.gather(1, choice_targets.unsqueeze(1)).squeeze(1)
                choice_logprobs.append(choice_log_prob.sum().item())
            else:
                choice_logprobs.append(-float('inf'))
    
    return choice_logprobs


def evaluate_hellaswag(model, tokenizer, num_samples: int = 100):
    """Evaluate on HellaSwag dataset."""
    print("Loading HellaSwag dataset...")
    try:
        dataset = load_dataset("hellaswag", split="validation")
    except:
        print("Could not load HellaSwag dataset. Skipping...")
        return {}
    
    correct = 0
    total = 0
    
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        if i % 20 == 0:
            print(f"Processing HellaSwag example {i}/{min(num_samples, len(dataset))}")
        
        # Format context
        context = example["ctx"]
        choices = example["endings"]
        correct_idx = int(example["label"])
        
        # Get logits for each choice
        choice_scores = get_logits_for_choices(model, tokenizer, context, choices)
        
        # Predict
        predicted_idx = np.argmax(choice_scores)
        
        if predicted_idx == correct_idx:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"HellaSwag Accuracy: {accuracy:.3f} ({correct}/{total})")
    return {"hellaswag_accuracy": accuracy, "hellaswag_correct": correct, "hellaswag_total": total}


def evaluate_commonsense_qa(model, tokenizer, num_samples: int = 100):
    """Evaluate on CommonsenseQA dataset."""
    print("Loading CommonsenseQA dataset...")
    try:
        dataset = load_dataset("commonsense_qa", split="validation")
    except:
        print("Could not load CommonsenseQA dataset. Skipping...")
        return {}
    
    correct = 0
    total = 0
    
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        if i % 20 == 0:
            print(f"Processing CommonsenseQA example {i}/{min(num_samples, len(dataset))}")
        
        # Format as multiple choice
        question = example["question"]
        choices = example["choices"]["text"]
        correct_answer = example["answerKey"]
        
        # Map answer key to index
        answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        if correct_answer not in answer_map or len(choices) <= answer_map[correct_answer]:
            continue
        
        correct_idx = answer_map[correct_answer]
        
        # Format context
        context = f"Question: {question}\nAnswer: "
        
        # Get logits for each choice
        choice_scores = get_logits_for_choices(model, tokenizer, context, choices)
        
        # Predict
        predicted_idx = np.argmax(choice_scores)
        
        if predicted_idx == correct_idx:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"CommonsenseQA Accuracy: {accuracy:.3f} ({correct}/{total})")
    return {"commonsense_qa_accuracy": accuracy, "commonsense_qa_correct": correct, "commonsense_qa_total": total}


def evaluate_instruction_following_quality(model, tokenizer, num_samples: int = 50):
    """Evaluate instruction following quality with simple prompts."""
    print("Evaluating instruction following quality...")
    
    test_instructions = [
        "List three colors.",
        "Count from 1 to 5.",
        "What is 2 + 2?",
        "Name a fruit.",
        "What day comes after Monday?",
        "Spell the word 'cat'.",
        "What is the opposite of hot?",
        "Name an animal that flies.",
        "What color is grass?",
        "How many legs does a dog have?",
    ]
    
    # Expected patterns in responses
    expected_patterns = [
        ["red", "blue", "green", "yellow", "purple", "orange"],  # colors
        ["1", "2", "3", "4", "5"],  # counting
        ["4", "four"],  # math
        ["apple", "banana", "orange", "grape", "strawberry"],  # fruit
        ["tuesday"],  # day
        ["c-a-t", "c a t"],  # spelling
        ["cold", "cool"],  # opposite
        ["bird", "bat", "eagle", "hawk"],  # flying animal
        ["green"],  # grass color
        ["four", "4"],  # dog legs
    ]
    
    model.eval()
    scores = []
    
    for i, (instruction, patterns) in enumerate(zip(test_instructions, expected_patterns)):
        if i >= num_samples:
            break
        
        # Format as instruction
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Generate response
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda().squeeze(0)
        input_ids = pad_to_block_size(input_ids)
        
        generated_tokens = []
        current_length = len(tokenizer.encode(prompt))
        
        with torch.no_grad():
            for _ in range(30):  # Max 30 tokens
                if current_length - 1 < len(input_ids):
                    logits = model(input_ids, None, get_window_size_blocks(1000, 1000))
                    next_token_logits = logits[0, current_length - 1, :]
                    probs = F.softmax(next_token_logits / 0.7, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    if current_length < len(input_ids):
                        input_ids[current_length] = next_token
                        current_length += 1
                        
                        gen_text = tokenizer.decode(next_token)
                        generated_tokens.append(gen_text)
                        
                        if "\n" in gen_text or next_token.item() == tokenizer.eos_token_id:
                            break
                    else:
                        break
                else:
                    break
        
        response = "".join(generated_tokens).strip().lower()
        
        # Check if response contains expected patterns
        score = 0
        for pattern in patterns:
            if pattern.lower() in response:
                score = 1
                break
        
        scores.append(score)
        
        if i < 5:  # Show first few examples
            print(f"Instruction: {instruction}")
            print(f"Response: {response}")
            print(f"Score: {score}")
            print("-" * 40)
    
    accuracy = np.mean(scores) if scores else 0
    print(f"Instruction Following Accuracy: {accuracy:.3f}")
    return {"instruction_following_accuracy": accuracy}


def main():
    """Run all evaluations."""
    print("=" * 60)
    print("Starting Evaluation Benchmarks")
    print("=" * 60)
    
    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run evaluations
    results = {}
    
    # Instruction following quality (quick test)
    results.update(evaluate_instruction_following_quality(model, tokenizer, num_samples=10))
    
    # HellaSwag (common sense reasoning)
    results.update(evaluate_hellaswag(model, tokenizer, num_samples=50))
    
    # CommonsenseQA
    results.update(evaluate_commonsense_qa(model, tokenizer, num_samples=50))
    
    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in results.items():
        if "accuracy" in key:
            print(f"{key}: {value:.3f}")
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()