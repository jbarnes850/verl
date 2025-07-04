#!/usr/bin/env python3
"""Simplified desktop screenshot dataset preparation focusing on ScreenSpot."""

import argparse
import base64
import io
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from datasets import load_dataset
from huggingface_hub import InferenceClient
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task descriptions for workplace scenarios
WORK_TASKS = [
    "Writing code in the IDE",
    "Debugging application errors", 
    "Reviewing pull requests",
    "Responding to Slack messages",
    "Attending video meetings",
    "Updating project documentation",
    "Checking email",
    "Updating Jira tickets",
    "Researching technical solutions",
    "Creating presentation slides"
]

class SimpleDesktopDataPreparer:
    """Prepare desktop screenshot dataset using ScreenSpot only."""
    
    def __init__(self, hf_token: str):
        """Initialize with HuggingFace token for VLM judge."""
        self.hf_token = hf_token
        self.client = InferenceClient(
            provider="auto",
            api_key=hf_token
        )
        
    def load_screenspot_only(self) -> List[Dict]:
        """Load ScreenSpot dataset."""
        logger.info("Loading ScreenSpot dataset...")
        
        try:
            dataset = load_dataset("rootsautomation/ScreenSpot", split="test")
            samples = []
            
            # Take first N samples regardless of platform
            for idx, item in enumerate(dataset):
                if idx >= 200:  # Limit for testing
                    break
                    
                samples.append({
                    'image': item['image'],
                    'instruction': item.get('instruction', ''),
                    'idx': idx
                })
                
            logger.info(f"Loaded {len(samples)} screenshots from ScreenSpot")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load ScreenSpot: {e}")
            return []
    
    def analyze_and_label(self, image: Image.Image, task: str) -> Tuple[str, str]:
        """Analyze screenshot and determine if on-task."""
        try:
            # Save image temporarily to get a URL (simplified approach)
            temp_path = f"/tmp/temp_image_{hash(str(image.tobytes()))}.png"
            image.save(temp_path, "PNG")
            
            # For demo, we'll create a simple data URL
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            data_url = f"data:image/png;base64,{img_base64}"
            
            # Single prompt to both analyze and judge
            prompt = f"""You are analyzing a computer screenshot to determine if someone is on-task for their work.

Work Task: "{task}"

Analyze this screenshot and answer:
1. What application or website is visible?
2. Is this on-task or off-task for the given work task?

Guidelines:
- Code editors, terminals, documentation = on-task for coding tasks
- Communication apps (Slack, email) = on-task for communication tasks  
- YouTube, social media, games = typically off-task
- News, shopping = typically off-task

Respond in this format:
Application: [name]
Label: [on-task or off-task]"""

            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]
                }
            ]
            
            # Use the correct HF API format
            completion = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                messages=messages,
                max_tokens=50,
                temperature=0.1
            )
            
            response = completion.choices[0].message.content
            
            # Parse response
            response_lower = response.lower()
            if "on-task" in response_lower:
                label = "on-task"
            elif "off-task" in response_lower:
                label = "off-task"
            else:
                label = "off-task"  # Default
                
            return label, response
            
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return "off-task", f"Error: {e}"
    
    def prepare_dataset(self, num_samples: int = 100, output_dir: str = "desktop_task_data_simple"):
        """Prepare simplified dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load ScreenSpot samples
        samples = self.load_screenspot_only()
        
        if not samples:
            logger.error("No samples loaded!")
            return
        
        # Limit to requested number
        if len(samples) > num_samples:
            samples = samples[:num_samples]
        
        logger.info(f"Processing {len(samples)} screenshots...")
        
        # Process samples
        labeled_data = []
        on_task_count = 0
        
        for idx, sample in enumerate(tqdm(samples, desc="Processing")):
            try:
                image = sample['image']
                if not isinstance(image, Image.Image):
                    continue
                
                # Save screenshot
                screenshot_path = output_path / f"screenshot_{idx:04d}.png"
                image.save(screenshot_path, "PNG")
                
                # Assign random work task
                task = random.choice(WORK_TASKS)
                
                # Analyze and label
                label, analysis = self.analyze_and_label(image, task)
                
                if label == "on-task":
                    on_task_count += 1
                
                labeled_data.append({
                    'screenshot': str(screenshot_path),
                    'task_description': task,
                    'label': label,
                    'analysis': analysis
                })
                
                # Log progress
                if (idx + 1) % 10 == 0:
                    ratio = on_task_count / (idx + 1)
                    logger.info(f"Progress: {idx+1}/{len(samples)}, On-task ratio: {ratio:.2%}")
                    
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
        
        # Split into train/val
        random.shuffle(labeled_data)
        split_idx = int(0.8 * len(labeled_data))
        
        train_data = labeled_data[:split_idx]
        val_data = labeled_data[split_idx:]
        
        # Save datasets
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        
        train_df.to_parquet(output_path / "desktop_train.parquet")
        val_df.to_parquet(output_path / "desktop_val.parquet")
        
        # Save metadata
        metadata = {
            'total_samples': len(labeled_data),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'on_task_ratio': on_task_count / len(labeled_data) if labeled_data else 0
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\nDataset preparation complete!")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"On-task ratio: {metadata['on_task_ratio']:.2%}")
        logger.info(f"Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="desktop_task_data_simple")
    args = parser.parse_args()
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    
    preparer = SimpleDesktopDataPreparer(hf_token)
    preparer.prepare_dataset(args.num_samples, args.output_dir)


if __name__ == "__main__":
    main()