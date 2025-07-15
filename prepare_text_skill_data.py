#!/usr/bin/env python3
"""
Prepare TEXT skill only dataset for VERL training with SEC curriculum.
Creates 350 train, 50 val, 100 test examples.
"""

import sys
import pandas as pd
import json
from datasets import load_dataset
from pathlib import Path

def main():
    print("="*80)
    print("PREPARING TEXT SKILL DATASET FOR SEC CURRICULUM")
    print("="*80)
    
    # TEXT skill tasks (5 categories)
    text_tasks = [
        "named_entity_disambiguation",
        "knowledge_qa", 
        "sales_insight_mining",
        "activity_priority",
        "wrong_stage_rectification"
    ]
    
    print(f"\nTEXT skill tasks: {text_tasks}")
    print(f"Total categories: {len(text_tasks)}")
    
    
    # Load the dataset and filter for TEXT tasks only
    print("\nLoading CRMArenaPro dataset...")
    dataset = load_dataset('Salesforce/CRMArenaPro', 'CRMArenaPro')
    
    # Filter for TEXT tasks
    text_data = []
    for example in dataset['b2b']:
        if example['task'] in text_tasks:
            text_data.append(example)
    
    print(f"Found {len(text_data)} TEXT skill examples")
    
    # Count examples per task
    task_counts = {}
    for ex in text_data:
        task = ex['task']
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print("\nExamples per task:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} examples")
    
    # We need exactly 500 examples total (350 train + 50 val + 100 test)
    # Let's take 100 examples from each of the 5 tasks
    print("\nSelecting 100 examples per task for balanced dataset...")
    
    balanced_data = []
    for task in text_tasks:
        task_examples = [ex for ex in text_data if ex['task'] == task]
        # Take first 100 examples from each task
        balanced_data.extend(task_examples[:100])
    
    print(f"Total balanced examples: {len(balanced_data)}")
    
    # Shuffle the data for better distribution
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(balanced_data)
    
    # Split into train/val/test
    train_data = balanced_data[:350]  # 350 examples
    val_data = balanced_data[350:400]  # 50 examples  
    test_data = balanced_data[400:500]  # 100 examples
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    
    # Convert to VERL format
    def convert_to_verl_format(examples, split_name):
        """Convert CRMArena examples to VERL format"""
        verl_data = []
        
        for ex in examples:
            # Format the answer
            answer = ex['answer']
            if isinstance(answer, list):
                answer_str = ", ".join(str(a) for a in answer if a is not None)
            else:
                answer_str = str(answer) if answer is not None else ""
            
            # Create VERL format entry with messages format
            messages = [
                {"role": "user", "content": ex['query']}
            ]
            
            verl_entry = {
                'prompt': json.dumps(messages),  # JSON string of messages
                'response': answer_str,
                'data_source': ex['task'],
                'original_task': ex['task'],
                'task_name': ex['task'],
                'reward_model': {
                    'data_source': ex['task'],
                    'ground_truth': answer_str
                },
                'metadata': ex.get('metadata', {})
            }
            
            verl_data.append(verl_entry)
        
        return pd.DataFrame(verl_data)
    
    # Create output directory
    output_dir = Path("./data/verl_text_only")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the data
    print("\nSaving data files...")
    
    train_df = convert_to_verl_format(train_data, "train")
    val_df = convert_to_verl_format(val_data, "val")
    test_df = convert_to_verl_format(test_data, "test")
    
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    test_df.to_parquet(test_path)
    
    print(f"\n✓ Data files generated successfully!")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")
    
    # Verify the distribution
    print("\n" + "="*80)
    print("VERIFYING DATA DISTRIBUTION")
    print("="*80)
    
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{split_name.upper()} split:")
        print(f"  Total examples: {len(df)}")
        print(f"  Task distribution:")
        task_counts = df['task_name'].value_counts()
        for task, count in sorted(task_counts.items()):
            print(f"    {task}: {count} examples")
        
        # Check ground truth
        non_empty = (df['response'] != '').sum()
        print(f"  Ground truth present: {non_empty}/{len(df)} ({non_empty/len(df)*100:.1f}%)")
    
    print("\n✓ TEXT skill dataset prepared successfully!")
    print(f"✓ Total examples: 500 (350 train + 50 val + 100 test)")
    print(f"✓ Tasks: {text_tasks}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)