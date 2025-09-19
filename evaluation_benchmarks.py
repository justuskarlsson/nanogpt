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
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
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


def generate_text_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
    """Generate a text response from the model."""
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
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Add to sequence
                if current_length < len(input_ids):
                    input_ids[current_length] = next_token
                    current_length += 1

                    # Decode and check for stopping
                    gen_text = tokenizer.decode(next_token)
                    generated_tokens.append(gen_text)

                    # Stop at EOS token or newline
                    if next_token.item() == tokenizer.eos_token_id or "\n" in gen_text:
                        break
                else:
                    break
            else:
                break

    return "".join(generated_tokens).strip()


def assess_response_quality(instruction, response):
    """Simple heuristic to assess response quality."""
    if not response or len(response.strip()) < 2:
        return False

    # Check for repetition (same token repeated multiple times)
    tokens = response.split()
    if len(tokens) > 3:
        # Check if more than half the tokens are the same
        from collections import Counter
        token_counts = Counter(tokens)
        most_common_count = token_counts.most_common(1)[0][1]
        if most_common_count > len(tokens) / 2:
            return False

    # Simple keyword matching for specific questions
    instruction_lower = instruction.lower()
    response_lower = response.lower()

    if "capital of france" in instruction_lower:
        return "paris" in response_lower
    elif "2 + 2" in instruction_lower or "2+2" in instruction_lower:
        return "4" in response_lower
    elif "colors" in instruction_lower:
        colors = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown"]
        return any(color in response_lower for color in colors)
    elif "photosynthesis" in instruction_lower:
        keywords = ["light", "sun", "plant", "energy", "carbon", "oxygen", "glucose"]
        return any(keyword in response_lower for keyword in keywords)
    elif "exercise" in instruction_lower:
        keywords = ["health", "fit", "strong", "heart", "muscle", "weight", "energy"]
        return any(keyword in response_lower for keyword in keywords)
    elif "shape" in instruction_lower and "earth" in instruction_lower:
        return any(word in response_lower for word in ["round", "sphere", "ball", "circular"])
    elif "days" in instruction_lower and "week" in instruction_lower:
        return "7" in response_lower or "seven" in response_lower
    elif "fruit" in instruction_lower:
        fruits = ["apple", "banana", "orange", "grape", "strawberry", "pear", "peach", "cherry"]
        return any(fruit in response_lower for fruit in fruits)
    elif "bees make" in instruction_lower:
        return "honey" in response_lower
    elif "opposite of hot" in instruction_lower:
        return any(word in response_lower for word in ["cold", "cool", "chilly", "freezing"])

    # If no specific check, just check that response has reasonable length and no obvious repetition
    return len(response.strip()) > 5 and len(response.strip()) < 200


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


def evaluate_hellaswag_detailed(model, tokenizer, num_samples: int = 100):
    """Evaluate on HellaSwag dataset with detailed tracking."""
    print("Loading HellaSwag dataset...")
    try:
        dataset = load_dataset("hellaswag", split="validation")
    except:
        print("Could not load HellaSwag dataset. Skipping...")
        return {}

    correct = 0
    total = 0
    details = []

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

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        # Save details
        details.append({
            "input": context,
            "choices": choices,
            "correct_answer": choices[correct_idx],
            "predicted_answer": choices[predicted_idx],
            "correct": is_correct,
            "scores": choice_scores
        })

    accuracy = correct / total if total > 0 else 0
    print(f"HellaSwag Accuracy: {accuracy:.3f} ({correct}/{total})")
    return {
        "hellaswag_accuracy": accuracy,
        "hellaswag_correct": correct,
        "hellaswag_total": total,
        "hellaswag_details": details
    }


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


def evaluate_commonsense_qa_detailed(model, tokenizer, num_samples: int = 100):
    """Evaluate on CommonsenseQA dataset with detailed tracking."""
    print("Loading CommonsenseQA dataset...")
    try:
        dataset = load_dataset("commonsense_qa", split="validation")
    except:
        print("Could not load CommonsenseQA dataset. Skipping...")
        return {}

    correct = 0
    total = 0
    details = []

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

        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        # Save details
        details.append({
            "question": question,
            "choices": choices,
            "correct_answer": choices[correct_idx],
            "predicted_answer": choices[predicted_idx],
            "correct": is_correct,
            "scores": choice_scores
        })

    accuracy = correct / total if total > 0 else 0
    print(f"CommonsenseQA Accuracy: {accuracy:.3f} ({correct}/{total})")
    return {
        "commonsense_qa_accuracy": accuracy,
        "commonsense_qa_correct": correct,
        "commonsense_qa_total": total,
        "commonsense_qa_details": details
    }


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


def evaluate_instruction_following_quality_detailed(model, tokenizer, num_samples: int = 50):
    """Evaluate instruction following quality with simple prompts and detailed tracking."""
    test_instructions = [
        "What is the capital of France?",
        "Name three colors.",
        "What is 2 + 2?",
        "Explain what photosynthesis is in one sentence.",
        "List two benefits of exercise.",
        "What shape is the Earth?",
        "How many days are in a week?",
        "Name a type of fruit.",
        "What do bees make?",
        "What is the opposite of hot?"
    ]

    details = []
    good_responses = 0

    for i in range(min(num_samples, len(test_instructions))):
        instruction = test_instructions[i % len(test_instructions)]

        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

        # Generate response
        response = generate_text_response(model, tokenizer, prompt, max_new_tokens=50)

        # Simple quality assessment
        is_good = assess_response_quality(instruction, response)
        if is_good:
            good_responses += 1

        # Save details
        details.append({
            "instruction": instruction,
            "response": response,
            "assessment": "Good" if is_good else "Poor"
        })

        print(f"Instruction: {instruction}")
        print(f"Response: {response}")
        print(f"Assessment: {'Good' if is_good else 'Poor'}")
        print("-" * 40)

    accuracy = good_responses / len(details) if details else 0
    print(f"Instruction Following Accuracy: {accuracy:.3f}")
    return {
        "instruction_following_accuracy": accuracy,
        "instruction_following_details": details
    }


def write_detailed_results(results, output_file="evaluation_details.txt"):
    """Write detailed evaluation results to a text file."""
    with open(output_file, "w") as f:
        f.write("DETAILED EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        # HellaSwag details
        if "hellaswag_details" in results:
            f.write("HELLASWAG RESULTS\n")
            f.write("-" * 30 + "\n")
            for i, detail in enumerate(results["hellaswag_details"]):
                f.write(f"Example {i+1}:\n")
                f.write(f"Input: {detail['input']}\n")
                f.write(f"Choices: {detail['choices']}\n")
                f.write(f"Correct Answer: {detail['correct_answer']}\n")
                f.write(f"Predicted Answer: {detail['predicted_answer']}\n")
                f.write(f"Correct: {'YES' if detail['correct'] else 'NO'}\n")
                f.write(f"Scores: {[f'{s:.3f}' for s in detail['scores']]}\n")
                f.write("\n")

        # CommonsenseQA details
        if "commonsense_qa_details" in results:
            f.write("\nCOMMONSENSE QA RESULTS\n")
            f.write("-" * 30 + "\n")
            for i, detail in enumerate(results["commonsense_qa_details"]):
                f.write(f"Example {i+1}:\n")
                f.write(f"Question: {detail['question']}\n")
                f.write(f"Choices: {detail['choices']}\n")
                f.write(f"Correct Answer: {detail['correct_answer']}\n")
                f.write(f"Predicted Answer: {detail['predicted_answer']}\n")
                f.write(f"Correct: {'YES' if detail['correct'] else 'NO'}\n")
                f.write(f"Scores: {[f'{s:.3f}' for s in detail['scores']]}\n")
                f.write("\n")

        # Instruction following details
        if "instruction_following_details" in results:
            f.write("\nINSTRUCTION FOLLOWING RESULTS\n")
            f.write("-" * 30 + "\n")
            for i, detail in enumerate(results["instruction_following_details"]):
                f.write(f"Example {i+1}:\n")
                f.write(f"Instruction: {detail['instruction']}\n")
                f.write(f"Generated Response: {detail['response']}\n")
                f.write(f"Quality Assessment: {detail['assessment']}\n")
                f.write("\n")


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

    # Run evaluations with detailed tracking
    results = {}

    # Instruction following quality (quick test)
    results.update(evaluate_instruction_following_quality_detailed(model, tokenizer, num_samples=10))

    # HellaSwag (common sense reasoning)
    results.update(evaluate_hellaswag_detailed(model, tokenizer, num_samples=50))

    # CommonsenseQA
    results.update(evaluate_commonsense_qa_detailed(model, tokenizer, num_samples=50))

    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in results.items():
        if "accuracy" in key:
            print(f"{key}: {value:.3f}")

    # Save results
    with open("evaluation_results.json", "w") as f:
        # Remove detailed results from JSON (too large)
        json_results = {k: v for k, v in results.items() if "_details" not in k}
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to evaluation_results.json")

    # Save detailed results to text file
    write_detailed_results(results)
    print(f"Detailed results saved to evaluation_details.txt")


if __name__ == "__main__":
    main()