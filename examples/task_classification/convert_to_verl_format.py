#!/usr/bin/env python3
"""Convert task classification data to VERL expected format."""

import argparse
import pandas as pd
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO

def convert_to_verl_format(input_dir: str, output_dir: str):
    """Convert our data format to VERL's expected format with 'prompt' key."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create images subdirectory
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val']:
        # Load our format
        df = pd.read_parquet(input_path / f"desktop_{split}.parquet")
        print(f"\n{split.upper()} split:")
        print(f"Original columns: {list(df.columns)}")
        print(f"Sample count: {len(df)}")
        
        # Convert to VERL format
        verl_data = []
        for idx, row in df.iterrows():
            # Create prompt in chat format with image placeholder
            prompt = [
                {
                    "role": "user",
                    "content": f"<image>Is this screenshot showing someone on-task for: {row['task_description']}? Answer with only 'on-task' or 'off-task'."
                }
            ]
            
            # The response is the label
            response = row['label']
            
            # Process image path - use absolute path
            image_path = str(Path(row['screenshot']).absolute())
            
            # For compatibility with different qwen_vl_utils versions
            # Try the format that works with the remote GPU's version
            image_dict = {
                "image": image_path,  # This key works in newer versions
                "image_url": image_path  # This key works in older versions
            }
            
            # Create VERL format entry
            verl_entry = {
                'data_source': 'task_classification',  # Add data source
                'prompt': prompt,  # Store as list directly, not JSON string
                'response': response,
                'task_description': row['task_description'],
                'screenshot': image_path,
                'ground_truth': response,  # For reward calculation
                'images': [image_dict]  # Include both keys for compatibility
            }
            
            # Add analysis if available
            if 'analysis' in row:
                verl_entry['analysis'] = row['analysis']
            
            verl_data.append(verl_entry)
        
        # Save in VERL format
        verl_df = pd.DataFrame(verl_data)
        output_file = output_path / f"verl_{split}.parquet"
        verl_df.to_parquet(output_file)
        
        print(f"Converted columns: {list(verl_df.columns)}")
        print(f"Saved to: {output_file}")
        
        # Show sample
        if len(verl_df) > 0:
            print(f"\nSample entry:")
            sample = verl_df.iloc[0].to_dict()
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Input directory with balanced data")
    parser.add_argument("--output-dir", required=True, help="Output directory for VERL format")
    args = parser.parse_args()
    
    convert_to_verl_format(args.input_dir, args.output_dir)
    print("\nConversion complete!")

if __name__ == "__main__":
    main()