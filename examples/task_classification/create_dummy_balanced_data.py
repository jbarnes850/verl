#!/usr/bin/env python3
"""Create dummy balanced dataset for testing VERL format locally."""

import pandas as pd
from pathlib import Path
import random

def create_dummy_data():
    """Create dummy balanced data with 60/40 split."""
    
    output_dir = Path("desktop_task_data_balanced")
    output_dir.mkdir(exist_ok=True)
    
    # Create dummy data - 6 on-task, 4 off-task
    tasks = [
        ("Writing code in VS Code", "on-task"),
        ("Reviewing pull request on GitHub", "on-task"),
        ("Running tests in terminal", "on-task"),
        ("Debugging Python script", "on-task"),
        ("Writing documentation", "on-task"),
        ("Analyzing logs", "on-task"),
        ("Browsing social media", "off-task"),
        ("Watching YouTube videos", "off-task"),
        ("Reading news articles", "off-task"),
        ("Playing games", "off-task"),
    ]
    
    # Create train data (8 samples)
    train_data = []
    for i in range(8):
        task, label = tasks[i]
        train_data.append({
            'task_description': task,
            'label': label,
            'screenshot': f'/dummy/path/screenshot_{i}.png',
            'analysis': f'Dummy analysis for {task}'
        })
    
    # Create val data (2 samples)
    val_data = []
    for i in range(8, 10):
        task, label = tasks[i]
        val_data.append({
            'task_description': task,
            'label': label,
            'screenshot': f'/dummy/path/screenshot_{i}.png',
            'analysis': f'Dummy analysis for {task}'
        })
    
    # Save as parquet
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_df.to_parquet(output_dir / "desktop_train.parquet")
    val_df.to_parquet(output_dir / "desktop_val.parquet")
    
    print(f"Created dummy balanced data in {output_dir}")
    print(f"Train: {len(train_df)} samples - {(train_df['label'] == 'on-task').mean():.0%} on-task")
    print(f"Val: {len(val_df)} samples - {(val_df['label'] == 'on-task').mean():.0%} on-task")
    print(f"Total: {len(train_df) + len(val_df)} samples - 60% on-task")

if __name__ == "__main__":
    create_dummy_data()