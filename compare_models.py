#!/usr/bin/env python3
"""
Compare SEC-trained model with baseline model on CRM tasks
"""

import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os
import glob
import shutil
from pathlib import Path
sys.path.append('crm-arena/src')
from crm_reward_functions import CRMVerifiableRewards

# Import VERL model merger
sys.path.append('.')
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig


def convert_verl_checkpoint(checkpoint_path):
    """Convert VERL FSDP checkpoint to HuggingFace format"""
    # Create output directory
    output_dir = checkpoint_path + "_hf"
    
    # Check if already converted
    hf_dir = os.path.join(output_dir, "huggingface")
    if os.path.exists(hf_dir) and os.path.exists(os.path.join(hf_dir, "config.json")):
        print(f"Using existing converted checkpoint at {hf_dir}")
        return hf_dir
    
    print(f"Converting checkpoint from {checkpoint_path} to {output_dir}")
    
    # Create merger config
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=checkpoint_path,
        target_dir=output_dir,
        hf_model_config_path=os.path.join(checkpoint_path, "huggingface")
    )
    
    # Create and run the merger
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    
    # The merged model should now be in HuggingFace format
    if os.path.exists(hf_dir):
        return hf_dir
    else:
        # Sometimes it's directly in output_dir
        return output_dir


def load_model_and_tokenizer(model_path, device="cuda", is_verl_checkpoint=False, base_model_name=None):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    if is_verl_checkpoint:
        # Convert VERL checkpoint to HuggingFace format first
        print("Converting VERL checkpoint to HuggingFace format...")
        hf_path = convert_verl_checkpoint(model_path)
        
        # Load the converted model
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer
    else:
        # Regular model loading for baseline
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """Generate response for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def evaluate_model(model, tokenizer, test_data, reward_fn, num_samples=None):
    """Evaluate model on test data"""
    if num_samples:
        test_data = test_data.sample(min(num_samples, len(test_data)))
    
    results = []
    skill_scores = {}
    task_scores = {}
    difficulty_scores = {}
    
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating"):
        # Parse prompt from JSON
        prompt_json = json.loads(row['prompt'])
        prompt_text = prompt_json[0]['content']
        
        # Generate response
        response = generate_response(model, tokenizer, prompt_text)
        
        # Compute reward
        task_name = row['task_name']
        ground_truth = row['response']
        reward = reward_fn.compute_reward(task_name, response, ground_truth)
        
        # Store results
        results.append({
            'task_name': task_name,
            'skill': row['skill'],
            'difficulty': row['difficulty'],
            'reward': reward,
            'prompt': prompt_text[:100] + "...",
            'generated': response[:200] + "...",
            'ground_truth': ground_truth[:200] + "..."
        })
        
        # Aggregate scores
        skill = row['skill']
        difficulty = row['difficulty']
        
        if skill not in skill_scores:
            skill_scores[skill] = []
        skill_scores[skill].append(reward)
        
        if task_name not in task_scores:
            task_scores[task_name] = []
        task_scores[task_name].append(reward)
        
        if difficulty not in difficulty_scores:
            difficulty_scores[difficulty] = []
        difficulty_scores[difficulty].append(reward)
    
    return results, skill_scores, task_scores, difficulty_scores

def print_comparison(baseline_results, sec_results):
    """Print comparison between baseline and SEC models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Overall performance
    baseline_mean = np.mean([r['reward'] for r in baseline_results[0]])
    sec_mean = np.mean([r['reward'] for r in sec_results[0]])
    improvement = ((sec_mean - baseline_mean) / baseline_mean) * 100
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Baseline Model: {baseline_mean:.3f} ({baseline_mean*100:.1f}% correct)")
    print(f"  SEC-Trained Model: {sec_mean:.3f} ({sec_mean*100:.1f}% correct)")
    print(f"  Improvement: {improvement:+.1f}%")
    
    # By skill
    print(f"\nPERFORMANCE BY SKILL:")
    print(f"{'Skill':<15} {'Baseline':<10} {'SEC':<10} {'Change':<10}")
    print("-" * 45)
    
    for skill in baseline_results[1].keys():
        baseline_skill = np.mean(baseline_results[1][skill])
        sec_skill = np.mean(sec_results[1][skill]) if skill in sec_results[1] else 0
        change = sec_skill - baseline_skill
        print(f"{skill:<15} {baseline_skill:<10.3f} {sec_skill:<10.3f} {change:+10.3f}")
    
    # By difficulty
    print(f"\nPERFORMANCE BY DIFFICULTY:")
    print(f"{'Difficulty':<15} {'Baseline':<10} {'SEC':<10} {'Change':<10}")
    print("-" * 45)
    
    for diff in baseline_results[3].keys():
        baseline_diff = np.mean(baseline_results[3][diff])
        sec_diff = np.mean(sec_results[3][diff]) if diff in sec_results[3] else 0
        change = sec_diff - baseline_diff
        print(f"{diff:<15} {baseline_diff:<10.3f} {sec_diff:<10.3f} {change:+10.3f}")
    
    # Top improved tasks
    print(f"\nTOP 5 MOST IMPROVED TASKS:")
    task_improvements = []
    for task in baseline_results[2].keys():
        baseline_task = np.mean(baseline_results[2][task])
        sec_task = np.mean(sec_results[2][task]) if task in sec_results[2] else 0
        improvement = sec_task - baseline_task
        task_improvements.append((task, baseline_task, sec_task, improvement))
    
    task_improvements.sort(key=lambda x: x[3], reverse=True)
    print(f"{'Task':<40} {'Baseline':<10} {'SEC':<10} {'Change':<10}")
    print("-" * 70)
    for task, baseline, sec, improvement in task_improvements[:5]:
        print(f"{task:<40} {baseline:<10.3f} {sec:<10.3f} {improvement:+10.3f}")

def main():
    parser = argparse.ArgumentParser(description='Compare baseline and SEC-trained models')
    parser.add_argument('--baseline-model', type=str, default='Qwen/Qwen2.5-3B',
                        help='Path to baseline model')
    parser.add_argument('--sec-model', type=str, default='./checkpoint/global_step_4/actor',
                        help='Path to SEC-trained model checkpoint')
    parser.add_argument('--test-data', type=str, default='./data/verl_format/test.parquet',
                        help='Path to test data')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to evaluate (default: 100)')
    parser.add_argument('--output-file', type=str, default='comparison_results.json',
                        help='Output file for detailed results')
    args = parser.parse_args()
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_parquet(args.test_data)
    print(f"Loaded {len(test_data)} test examples")
    
    # Initialize reward function
    reward_fn = CRMVerifiableRewards()
    
    # Evaluate SEC-trained model FIRST
    print("\n" + "="*50)
    print("EVALUATING SEC-TRAINED MODEL")
    print("="*50)
    
    # Load VERL checkpoint (will be converted automatically)
    sec_model, sec_tokenizer = load_model_and_tokenizer(
        args.sec_model,
        is_verl_checkpoint=True,
        base_model_name=args.baseline_model
    )
    sec_results = evaluate_model(sec_model, sec_tokenizer, test_data, reward_fn, args.num_samples)
    
    # Free memory
    del sec_model
    torch.cuda.empty_cache()
    
    # Evaluate baseline model
    print("\n" + "="*50)
    print("EVALUATING BASELINE MODEL")
    print("="*50)
    baseline_model, baseline_tokenizer = load_model_and_tokenizer(args.baseline_model)
    baseline_results = evaluate_model(baseline_model, baseline_tokenizer, test_data, reward_fn, args.num_samples)
    
    # Print comparison
    print_comparison(baseline_results, sec_results)
    
    # Save detailed results
    detailed_results = {
        'baseline': {
            'overall_score': np.mean([r['reward'] for r in baseline_results[0]]),
            'results': baseline_results[0][:10],  # Save first 10 for inspection
            'skill_scores': {k: np.mean(v) for k, v in baseline_results[1].items()},
            'task_scores': {k: np.mean(v) for k, v in baseline_results[2].items()},
            'difficulty_scores': {k: np.mean(v) for k, v in baseline_results[3].items()}
        },
        'sec': {
            'overall_score': np.mean([r['reward'] for r in sec_results[0]]),
            'results': sec_results[0][:10],  # Save first 10 for inspection
            'skill_scores': {k: np.mean(v) for k, v in sec_results[1].items()},
            'task_scores': {k: np.mean(v) for k, v in sec_results[2].items()},
            'difficulty_scores': {k: np.mean(v) for k, v in sec_results[3].items()}
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()