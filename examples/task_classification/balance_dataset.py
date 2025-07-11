#!/usr/bin/env python3
"""Balance task classification dataset to 60% on-task, 40% off-task."""

import argparse
import json
import random
from pathlib import Path
import pandas as pd
import numpy as np

def balance_dataset(input_dir: str, output_dir: str, target_on_task_ratio: float = 0.6):
    """Balance dataset by duplicating minority class or removing majority class."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load existing data
    train_df = pd.read_parquet(input_path / "desktop_train.parquet")
    val_df = pd.read_parquet(input_path / "desktop_val.parquet")
    
    # Combine for rebalancing
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"Original dataset: {len(all_df)} samples")
    print(f"Original on-task ratio: {(all_df['label'] == 'on-task').mean():.2%}")
    
    # Separate by class
    on_task = all_df[all_df['label'] == 'on-task'].copy()
    off_task = all_df[all_df['label'] == 'off-task'].copy()
    
    print(f"On-task samples: {len(on_task)}")
    print(f"Off-task samples: {len(off_task)}")
    
    # Calculate target numbers
    total_samples = len(all_df)
    target_on_task = int(total_samples * target_on_task_ratio)
    target_off_task = total_samples - target_on_task
    
    print(f"\nTarget on-task: {target_on_task}")
    print(f"Target off-task: {target_off_task}")
    
    # Balance the dataset
    balanced_on_task = balance_class(on_task, target_on_task)
    balanced_off_task = balance_class(off_task, target_off_task)
    
    # Combine balanced data
    balanced_df = pd.concat([balanced_on_task, balanced_off_task], ignore_index=True)
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset: {len(balanced_df)} samples")
    print(f"Balanced on-task ratio: {(balanced_df['label'] == 'on-task').mean():.2%}")
    
    # Split back into train/val (80/20)
    split_idx = int(0.8 * len(balanced_df))
    balanced_train = balanced_df[:split_idx]
    balanced_val = balanced_df[split_idx:]
    
    # Save balanced datasets
    balanced_train.to_parquet(output_path / "desktop_train.parquet")
    balanced_val.to_parquet(output_path / "desktop_val.parquet")
    
    # Update metadata
    metadata = {
        'total_samples': len(balanced_df),
        'train_samples': len(balanced_train),
        'val_samples': len(balanced_val),
        'on_task_ratio': (balanced_df['label'] == 'on-task').mean(),
        'target_ratio': target_on_task_ratio,
        'balancing_method': 'duplicate_minority_or_remove_majority'
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nBalanced dataset saved to: {output_path}")
    print(f"Training samples: {len(balanced_train)}")
    print(f"Validation samples: {len(balanced_val)}")
    
    return output_path

def balance_class(class_df, target_count):
    """Balance a single class to target count by duplicating or removing samples."""
    current_count = len(class_df)
    
    if current_count == target_count:
        return class_df
    elif current_count < target_count:
        # Duplicate samples to reach target
        shortage = target_count - current_count
        duplicates_needed = shortage
        
        # Randomly sample with replacement
        additional_samples = class_df.sample(n=duplicates_needed, replace=True, random_state=42)
        return pd.concat([class_df, additional_samples], ignore_index=True)
    else:
        # Remove samples to reach target
        return class_df.sample(n=target_count, random_state=42)

def main():
    parser = argparse.ArgumentParser(description="Balance task classification dataset")
    parser.add_argument("--input-dir", required=True, help="Input directory with unbalanced data")
    parser.add_argument("--output-dir", required=True, help="Output directory for balanced data")
    parser.add_argument("--target-ratio", type=float, default=0.6, help="Target on-task ratio (default: 0.6)")
    
    args = parser.parse_args()
    
    balance_dataset(args.input_dir, args.output_dir, args.target_ratio)

if __name__ == "__main__":
    main()