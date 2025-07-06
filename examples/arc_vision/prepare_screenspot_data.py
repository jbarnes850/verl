# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prepare ScreenSpot dataset for Arc Vision RL training.

This script converts the official ScreenSpot dataset from Hugging Face
(https://huggingface.co/datasets/rootsautomation/ScreenSpot) 
to VERL-compatible parquet format.

IMPORTANT: This script uses the <|image_pad|> token that Qwen2.5-VL expects
for image placeholders in prompts. This is critical for the model to properly
process images during training.
"""

import argparse
import os
from typing import Dict, List, Any

import datasets
from PIL import Image
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs


def create_reasoning_prompt(instruction: str) -> str:
    """Create an optimized prompt for UI detection that works with small models.
    
    Key changes based on research:
    1. Removes explicit confidence scores (models hallucinate these)
    2. Uses decision-based approach (tool or no tool)
    3. Maintains compatibility with reward function
    4. Leverages model's training on reasoning and tool usage
    """
    # For Qwen2.5-VL, we use <image> in the prompt text
    prompt = f"""<image>
{instruction}

Step 1: Analyze the target element
<reasoning>
Examine these specific factors:
- Size: Is the element smaller than 5% of screen area?
- Visibility: Can you clearly see all edges and boundaries?
- Contrast: Is it easily distinguishable from the background?
- Occlusion: Is any part hidden or overlapped?
</reasoning>

Step 2: Decide your approach based on the analysis above

IF the element is CLEAR (all edges visible, good contrast, >5% screen area):
→ Provide direct detection:
<bbox>[x1, y1, x2, y2]</bbox>

IF the element is UNCLEAR (small, blurry, partially hidden, or low contrast):
→ Use tools FIRST, then detect:
<use_tool>zoom_ui_element</use_tool> - for small elements
<use_tool>inspect_element</use_tool> - for unclear boundaries
<use_tool>wait_for_ui</use_tool> - if elements are still loading

CRITICAL RULES:
- Coordinates must be normalized (0.0 to 1.0)
- Format: [x1, y1, x2, y2] where x1<x2 and y1<y2
- x1,y1 = top-left; x2,y2 = bottom-right
- Example: <bbox>[0.142, 0.058, 0.384, 0.126]</bbox>

Decision threshold: If ANY of these are true, use tools:
- Element width or height < 0.05 (5% of screen)
- Cannot see all four edges clearly
- Low contrast with background
- Overlapped by other elements"""
    
    return prompt


def process_screenspot_sample(sample: Dict[str, Any], idx: int, split: str, image_dir: str) -> Dict[str, Any]:
    """Process a single ScreenSpot sample into VERL format.
    
    Args:
        sample: Raw ScreenSpot sample
        idx: Sample index
        split: Dataset split (train/validation/test)
        image_dir: Directory to save images
    
    Returns:
        Processed sample in VERL format
    """
    # Extract fields from ScreenSpot
    instruction = sample.get("instruction", "")
    image = sample.get("image")
    bbox = sample.get("bbox", [0, 0, 0, 0])  # [x_min, y_min, x_max, y_max]
    
    # Ensure bbox is in normalized format (0-1)
    # ScreenSpot bboxes should already be normalized, but let's verify
    bbox_normalized = [
        max(0, min(1, float(bbox[0]))),  # x1
        max(0, min(1, float(bbox[1]))),  # y1
        max(0, min(1, float(bbox[2]))),  # x2
        max(0, min(1, float(bbox[3])))   # y2
    ]
    
    # Save image to disk
    image_filename = f"{split}_{idx}.png"
    full_image_path = os.path.join(image_dir, image_filename)
    if image:
        image.save(full_image_path)
    
    # Use absolute path for container environment
    # VERL copies parquet files to cache but not image directories,
    # so we need absolute paths to find the images
    image_path = full_image_path
    
    # Create reasoning-enhanced prompt
    enhanced_prompt = create_reasoning_prompt(instruction)
    
    # Format as chat messages
    messages = [
        {
            "role": "user",
            "content": enhanced_prompt
        }
    ]
    
    # Create VERL-compatible record
    record = {
        "data_source": "arc_vision",  # Use consistent data source name
        "prompt": messages,  # Store as list directly, like geo3k example
        "images": [{"image": image_path}],  # Use dict format expected by Qwen2.5-VL
        "ability": "ui_detection",
        "ground_truth": bbox_normalized,  # Add at top level for reward function
        "reward_model": {  # Store as dict, not JSON string
            "style": "arc_vision",
            "ground_truth": bbox_normalized,
            "confidence_threshold": 0.7,
            "reward_weights": {
                "task": 0.6,
                "tool": 0.3,
                "gate": 0.1
            }
        },
        "extra_info": {  # Store as dict, not JSON string
            "split": split,
            "index": idx,
            "original_instruction": instruction,
            "original_bbox": bbox,
            "element_type": sample.get("element_type", "unknown"),
            "screenshot_id": sample.get("screenshot_id", f"{split}_{idx}")
        }
    }
    
    return record


def main():
    parser = argparse.ArgumentParser(description="Prepare ScreenSpot dataset for Arc Vision RL")
    parser.add_argument("--local_dir", default="~/data/arc_vision/screenspot", 
                        help="Local directory to save processed data")
    parser.add_argument("--hdfs_dir", default=None, 
                        help="Optional HDFS directory to copy data to")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for debugging)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        help="Dataset splits to process")
    parser.add_argument("--split_test_data", action="store_true",
                        help="Split test data into train/validation/test (60/20/20)")
    
    args = parser.parse_args()
    
    # Expand local directory path
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    print("Loading ScreenSpot dataset from Hugging Face...")
    
    # Load the official ScreenSpot dataset
    dataset_dict = datasets.load_dataset("rootsautomation/ScreenSpot")
    
    # Check what splits are actually available
    available_splits = list(dataset_dict.keys())
    print(f"Available splits in ScreenSpot: {available_splits}")
    
    # If only test split exists and user wants to create train/val splits
    if len(available_splits) == 1 and 'test' in available_splits and args.split_test_data:
        print("\nOnly test split found. Creating train/validation/test splits from test data...")
        test_dataset = dataset_dict['test']
        
        # Limit samples if specified
        if args.max_samples:
            test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
        
        total_samples = len(test_dataset)
        print(f"Total samples to split: {total_samples}")
        
        # Calculate split sizes - optimize for training data
        # Original would be 60/20/20, but we cap validation at 60
        test_size = int(0.2 * total_samples)  # Keep test at 20%
        val_size = min(60, int(0.2 * total_samples))  # Cap validation at 60 samples for efficiency
        train_size = total_samples - test_size - val_size  # Rest goes to training
        
        print(f"Split sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
        
        # Create splits
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_samples))
        
        splits_to_process = [
            ('train', test_dataset.select(train_indices)),
            ('validation', test_dataset.select(val_indices)),
            ('test', test_dataset.select(test_indices))
        ]
    else:
        # Process original splits
        splits_to_process = []
        for split in args.splits:
            if split not in dataset_dict:
                print(f"Warning: Split '{split}' not found in dataset. Skipping.")
                continue
            
            dataset = dataset_dict[split]
            
            # Limit samples if specified
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            
            splits_to_process.append((split, dataset))
    
    # Process each split
    for split, dataset in splits_to_process:
        print(f"\nProcessing {split} split...")
        print(f"Total samples in {split}: {len(dataset)}")
        
        # Create directory for images
        image_dir = os.path.join(local_dir, f"{split}_images")
        os.makedirs(image_dir, exist_ok=True)
        
        # Process samples
        records = []
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split}")):
            try:
                record = process_screenspot_sample(sample, idx, split, image_dir)
                records.append(record)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        print(f"Successfully processed {len(records)} samples")
        
        # Convert to HuggingFace Dataset and save as parquet (preserves list format)
        from datasets import Dataset
        dataset = Dataset.from_list(records)
        output_file = os.path.join(local_dir, f"{split}.parquet")
        dataset.to_parquet(output_file)
        print(f"Saved {split} data to: {output_file}")
        
        # Print sample statistics
        print(f"\n{split} statistics:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Images saved to: {image_dir}")
        
        # Calculate statistics from the dataset
        instructions = [r['extra_info']['original_instruction'] for r in records]
        avg_instruction_len = sum(len(inst) for inst in instructions) / len(instructions)
        print(f"  Average instruction length: {avg_instruction_len:.1f} chars")
        
        # Check bbox distribution
        bboxes = [r['reward_model']['ground_truth'] for r in records]
        bbox_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        print(f"  Average bbox area: {sum(bbox_areas) / len(bbox_areas):.3f}")
        print(f"  Min bbox area: {min(bbox_areas):.3f}")
        print(f"  Max bbox area: {max(bbox_areas):.3f}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir:
        print(f"\nCopying data to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print("Data copied to HDFS successfully")
    
    print("\nData preparation complete!")
    print(f"Local data directory: {local_dir}")
    
    # Print example usage
    print("\nTo use this data in training, add to your config:")
    print(f"  data.train_files: {os.path.join(local_dir, 'train.parquet')}")
    print(f"  data.val_files: {os.path.join(local_dir, 'validation.parquet')}")
    print(f"  data.test_files: {os.path.join(local_dir, 'test.parquet')}")


if __name__ == "__main__":
    main()