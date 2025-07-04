#!/usr/bin/env python3
"""Verify Arc Vision setup on GPU instance."""

import os
import sys
from datasets import load_dataset
from PIL import Image

def verify_setup():
    """Verify the Arc Vision dataset is correctly set up on GPU."""
    
    data_dir = "/root/data/arc_vision/screenspot"
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    print(f"✅ Data directory exists: {data_dir}")
    
    # Check for parquet files
    splits = ["train", "validation", "test"]
    for split in splits:
        parquet_path = os.path.join(data_dir, f"{split}.parquet")
        if os.path.exists(parquet_path):
            print(f"✅ Found {split}.parquet")
            
            # Load and check a sample
            try:
                dataset = load_dataset("parquet", data_files=parquet_path, split="train")
                sample = dataset[0]
                
                # Check for required fields
                assert "prompt" in sample, f"Missing 'prompt' field in {split}"
                assert "images" in sample, f"Missing 'images' field in {split}"
                assert "data_source" in sample, f"Missing 'data_source' field in {split}"
                assert sample["data_source"] == "arc_vision", f"Wrong data_source in {split}"
                
                # Check image path
                image_info = sample["images"][0]
                image_path = image_info["image"]
                
                print(f"  - Sample image path: {image_path}")
                
                # Verify image exists
                if os.path.exists(image_path):
                    print(f"  - ✅ Image file exists")
                    
                    # Try to load the image
                    try:
                        img = Image.open(image_path)
                        print(f"  - ✅ Image loadable: {img.size}")
                    except Exception as e:
                        print(f"  - ❌ Error loading image: {e}")
                else:
                    print(f"  - ❌ Image file not found: {image_path}")
                    return False
                
                # Check prompt format
                prompt_content = sample["prompt"][0]["content"]
                if "<image>" in prompt_content:
                    print(f"  - ✅ Correct <image> token in prompt")
                else:
                    print(f"  - ❌ Missing <image> token in prompt")
                    return False
                    
            except Exception as e:
                print(f"❌ Error loading {split} dataset: {e}")
                return False
        else:
            print(f"❌ Missing {split}.parquet")
            return False
    
    # Check image directories
    for split in splits:
        image_dir = os.path.join(data_dir, f"{split}_images")
        if os.path.exists(image_dir):
            num_images = len([f for f in os.listdir(image_dir) if f.endswith('.png')])
            print(f"✅ Found {split}_images/ with {num_images} images")
        else:
            print(f"❌ Missing {split}_images/ directory")
    
    print("\n✅ All checks passed! Ready to train.")
    return True

if __name__ == "__main__":
    if verify_setup():
        print("\nTo start training, run:")
        print("cd /root/verl/examples/arc_vision")
        print("bash run_arc_vision_grpo.sh")
    else:
        print("\n❌ Setup verification failed. Please check the errors above.")
        sys.exit(1)