#!/usr/bin/env python3
"""
Verify Arc Vision data files are properly formatted and ready for training.
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path

def check_parquet_file(file_path):
    """Check if a parquet file is valid and has the expected structure."""
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return False
    
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        print(f"\n✓ Successfully read: {file_path}")
        print(f"  Number of samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['data_source', 'prompt', 'images', 'ground_truth', 'reward_model']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠️  Missing columns: {missing_cols}")
        else:
            print(f"  ✓ All required columns present")
        
        # Sample the first row to check data format
        if len(df) > 0:
            first_row = df.iloc[0]
            print(f"\n  Sample data from first row:")
            print(f"  - data_source: {first_row.get('data_source', 'N/A')}")
            
            # Check prompt format
            if 'prompt' in first_row:
                try:
                    prompt_data = json.loads(first_row['prompt'])
                    if isinstance(prompt_data, list) and len(prompt_data) > 0:
                        print(f"  - prompt: Valid chat format with {len(prompt_data)} messages")
                        # Check if prompt contains our updated format
                        user_content = prompt_data[0].get('content', '')
                        if 'provide the bounding box coordinates' in user_content:
                            print(f"    ✓ Prompt contains coordinate instructions")
                            if 'simple array' in user_content or 'normalized format' in user_content:
                                print(f"    ✓ Using updated flexible prompt format")
                            else:
                                print(f"    ⚠️  Using old prompt format - consider re-preparing data")
                except:
                    print(f"  - prompt: Invalid format")
            
            # Check images
            if 'images' in first_row:
                images = first_row['images']
                if isinstance(images, list) and len(images) > 0:
                    first_image = images[0]
                    if isinstance(first_image, dict) and 'image' in first_image:
                        image_path = first_image['image']
                        print(f"  - images: Valid format, first image: {image_path}")
                        if os.path.exists(image_path):
                            print(f"    ✓ Image file exists")
                        else:
                            print(f"    ⚠️  Image file not found - may need to adjust paths")
            
            # Check ground truth
            if 'ground_truth' in first_row:
                gt = first_row['ground_truth']
                if isinstance(gt, list) and len(gt) == 4:
                    print(f"  - ground_truth: {gt}")
                    if all(0 <= x <= 1 for x in gt):
                        print(f"    ✓ Normalized coordinates")
                    else:
                        print(f"    ⚠️  Non-normalized coordinates")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading {file_path}: {e}")
        return False

def main():
    print("="*60)
    print("Arc Vision Data Verification")
    print("="*60)
    
    # Default data directory
    default_dir = "/root/data/arc_vision/screenspot"
    
    # Check if custom directory provided
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = default_dir
    
    print(f"\nChecking data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"✗ Directory not found: {data_dir}")
        print("\nTry running:")
        print(f"  python verify_data.py /path/to/your/data")
        return
    
    # Check each parquet file
    files_to_check = ['train.parquet', 'validation.parquet', 'test.parquet']
    valid_files = []
    
    for file_name in files_to_check:
        file_path = os.path.join(data_dir, file_name)
        if check_parquet_file(file_path):
            valid_files.append(file_name)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if len(valid_files) >= 2 and 'train.parquet' in valid_files:
        print("✓ Data is ready for training!")
        print(f"  Valid files: {', '.join(valid_files)}")
        
        # Check if we need to update paths
        if data_dir != default_dir:
            print(f"\n⚠️  Note: Your data is not in the default location.")
            print(f"  Update your training command:")
            print(f"  TRAIN_DATA={os.path.join(data_dir, 'train.parquet')} \\")
            print(f"  VAL_DATA={os.path.join(data_dir, 'validation.parquet')} \\")
            print(f"  N_GPUS=2 bash examples/arc_vision/run_arc_vision_grpo.sh")
    else:
        print("✗ Data is not ready for training")
        print(f"  Missing required files")
        print("\nRun prepare_screenspot_data.py to prepare the data:")
        print(f"  cd examples/arc_vision")
        print(f"  python prepare_screenspot_data.py --local_dir {data_dir}")

if __name__ == "__main__":
    main()