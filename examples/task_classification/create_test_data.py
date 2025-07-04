#!/usr/bin/env python3
"""Create a small test dataset to verify VERL format compatibility."""

import json
import pandas as pd
from pathlib import Path

def create_test_data():
    """Create a minimal test dataset in VERL format."""
    
    output_dir = Path("desktop_task_data_verl_test")
    output_dir.mkdir(exist_ok=True)
    
    # Create a few test samples
    test_data = []
    
    # Sample 1: On-task
    sample1 = {
        'prompt': json.dumps([
            {
                "role": "user",
                "content": "<image>Is this screenshot showing someone on-task for: Writing a Python script? Answer with only 'on-task' or 'off-task'."
            }
        ]),
        'response': 'on-task',
        'task_description': 'Writing a Python script',
        'screenshot': '/path/to/screenshot1.png',
        'ground_truth': 'on-task',
        'images': [{"image_url": "file:///path/to/screenshot1.png"}]
    }
    
    # Sample 2: Off-task
    sample2 = {
        'prompt': json.dumps([
            {
                "role": "user",
                "content": "<image>Is this screenshot showing someone on-task for: Reviewing code? Answer with only 'on-task' or 'off-task'."
            }
        ]),
        'response': 'off-task',
        'task_description': 'Reviewing code',
        'screenshot': '/path/to/screenshot2.png',
        'ground_truth': 'off-task',
        'images': [{"image_url": "file:///path/to/screenshot2.png"}]
    }
    
    test_data.extend([sample1, sample2])
    
    # Create train and val splits
    train_df = pd.DataFrame([sample1])
    val_df = pd.DataFrame([sample2])
    
    # Save as parquet
    train_df.to_parquet(output_dir / "verl_train.parquet")
    val_df.to_parquet(output_dir / "verl_val.parquet")
    
    print(f"Created test data in {output_dir}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"\nSample structure:")
    for key in train_df.columns:
        print(f"  - {key}: {type(train_df.iloc[0][key])}")

if __name__ == "__main__":
    create_test_data()