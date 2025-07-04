#!/usr/bin/env python3
"""Prepare desktop screenshot dataset for task classification.

This script uses actual desktop/application screenshots from established datasets
instead of nature photos, properly aligning with the customer's use case.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import InferenceClient
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Task descriptions for workplace scenarios
WORK_TASKS = [
    # Development tasks
    "Writing code in the IDE",
    "Debugging application errors", 
    "Reviewing pull requests",
    "Running unit tests",
    "Checking build status",
    "Reading technical documentation",
    "Searching for code examples",
    
    # Communication tasks
    "Responding to Slack messages",
    "Attending video meetings",
    "Writing email responses",
    "Updating team in chat",
    "Reviewing meeting notes",
    
    # Documentation tasks
    "Updating project documentation",
    "Writing technical specifications",
    "Creating presentation slides",
    "Editing wiki pages",
    
    # Project management
    "Updating Jira tickets",
    "Reviewing project timeline",
    "Checking sprint board",
    "Planning next iteration",
    
    # Research tasks
    "Researching technical solutions",
    "Reading API documentation",
    "Analyzing performance metrics",
    "Reviewing system logs"
]

# Application keywords for classification
ON_TASK_APPS = {
    "vscode", "code", "ide", "intellij", "pycharm", "sublime", "vim", "emacs",
    "terminal", "console", "powershell", "cmd", "bash",
    "chrome", "firefox", "safari", "edge", "browser",
    "slack", "teams", "zoom", "meet", "discord",
    "jira", "confluence", "notion", "trello", "asana",
    "github", "gitlab", "bitbucket", "git",
    "excel", "sheets", "docs", "word", "powerpoint",
    "postman", "insomnia", "swagger",
    "datadog", "grafana", "kibana", "splunk"
}

OFF_TASK_APPS = {
    "youtube", "netflix", "spotify", "twitch", "hulu",
    "facebook", "instagram", "twitter", "tiktok", "reddit",
    "amazon", "ebay", "shopping", "store",
    "game", "steam", "epic", "minecraft", "fortnite",
    "news", "cnn", "bbc", "espn", "sports"
}


class DesktopScreenshotPreparer:
    """Prepare desktop screenshot dataset for task classification."""
    
    def __init__(self, hf_token: str):
        """Initialize with HuggingFace token for VLM judge."""
        self.hf_token = hf_token
        self.client = InferenceClient(token=hf_token)
        
    def load_screenspot_dataset(self) -> List[Dict]:
        """Load ScreenSpot dataset with desktop screenshots."""
        logger.info("Loading ScreenSpot dataset...")
        
        try:
            # Load the dataset - ScreenSpot only has 'test' split
            dataset = load_dataset("rootsautomation/ScreenSpot", split="test")
            
            samples = []
            for idx, item in enumerate(dataset):
                # Log progress
                if idx % 100 == 0:
                    logger.info(f"Processing ScreenSpot item {idx}...")
                
                # Debug: log platform to see what's available
                platform = item.get('platform', 'unknown')
                if idx < 5:  # Log first few platforms
                    logger.info(f"Sample {idx} platform: {platform}")
                
                # More flexible filtering - include all non-mobile platforms
                if platform not in ['iOS', 'Android', 'iPad']:
                    samples.append({
                        'image': item['image'],
                        'instruction': item.get('instruction', ''),
                        'platform': platform
                    })
            
            logger.info(f"Loaded {len(samples)} desktop screenshots from ScreenSpot")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load ScreenSpot: {e}")
            return []
    
    def load_osatlas_dataset(self) -> List[Dict]:
        """Load OS-Atlas dataset samples."""
        logger.info("Loading OS-Atlas dataset...")
        
        try:
            # Try to load OS-Atlas dataset
            dataset = load_dataset("OS-Copilot/OS-Atlas-data", split="train", streaming=True)
            
            samples = []
            count = 0
            logger.info("Starting to stream OS-Atlas samples...")
            
            for idx, item in enumerate(dataset):
                # Log progress every 10 items
                if idx % 10 == 0:
                    logger.info(f"Processed {idx} OS-Atlas items, collected {count} samples...")
                
                if count >= 500:  # Limit samples for demo
                    break
                
                # Debug first few items
                if idx < 3:
                    logger.info(f"OS-Atlas item {idx} keys: {list(item.keys())}")
                    logger.info(f"Platform: {item.get('platform', 'unknown')}")
                    
                # Check different possible image keys
                image_key = None
                for key in ['screenshot', 'image', 'img']:
                    if key in item:
                        image_key = key
                        break
                
                if image_key and item.get('platform') in ['Windows', 'Linux', 'macOS']:
                    samples.append({
                        'image': item[image_key],
                        'platform': item['platform'],
                        'elements': item.get('elements', [])
                    })
                    count += 1
            
            logger.info(f"Loaded {len(samples)} desktop screenshots from OS-Atlas")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load OS-Atlas: {e}")
            return []
    
    def classify_screenshot_content(self, image: Image.Image) -> Tuple[str, float]:
        """Use VLM to understand what application/content is shown."""
        try:
            # Create analysis prompt
            prompt = """Analyze this desktop screenshot and identify the primary application or website being used.

List the main application/website visible and describe what the user appears to be doing.

Format your response as:
Application: [name]
Activity: [brief description]
Category: [productivity/communication/development/entertainment/other]"""

            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"}
                    ]
                }
            ]
            
            # Get VLM analysis
            stream = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                messages=messages,
                max_tokens=150,
                temperature=0.3,
                stream=True,
                modality="image",
                image=image
            )
            
            response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            
            return response, 0.9  # High confidence for VLM analysis
            
        except Exception as e:
            logger.error(f"VLM classification failed: {e}")
            return "unknown", 0.5
    
    def judge_task_alignment(self, task: str, screenshot_analysis: str) -> str:
        """Judge if screenshot content aligns with task."""
        try:
            # Check for obvious on-task keywords
            analysis_lower = screenshot_analysis.lower()
            
            # Check for on-task applications
            for app in ON_TASK_APPS:
                if app in analysis_lower:
                    return "on-task"
            
            # Check for off-task applications  
            for app in OFF_TASK_APPS:
                if app in analysis_lower:
                    return "off-task"
            
            # If uncertain, use VLM judge
            judge_prompt = f"""Given the work task: "{task}"

And this screenshot analysis:
{screenshot_analysis}

Would this screenshot show someone who is on-task or off-task for their work assignment?

Consider:
- Development tools (IDEs, terminals) are on-task for coding tasks
- Communication tools (Slack, email) are on-task for collaboration tasks
- Documentation tools are on-task for writing tasks
- Entertainment sites (YouTube, social media) are typically off-task
- News/shopping sites are typically off-task unless directly work-related

Answer with only "on-task" or "off-task"."""

            response = self.client.text_generation(
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                prompt=judge_prompt,
                max_new_tokens=10,
                temperature=0.1
            )
            
            response = response.strip().lower()
            if "on-task" in response:
                return "on-task"
            elif "off-task" in response:
                return "off-task"
            else:
                # Default based on category
                if any(cat in screenshot_analysis.lower() for cat in ["productivity", "development", "communication"]):
                    return "on-task"
                else:
                    return "off-task"
                    
        except Exception as e:
            logger.error(f"Task alignment judgment failed: {e}")
            return "off-task"  # Conservative default
    
    def prepare_dataset(self, num_samples: int = 1000, output_dir: str = "desktop_task_data"):
        """Prepare dataset with real desktop screenshots."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load datasets
        screenspot_samples = self.load_screenspot_dataset()
        osatlas_samples = self.load_osatlas_dataset()
        
        # Combine samples
        all_samples = screenspot_samples + osatlas_samples
        
        if not all_samples:
            logger.error("No samples loaded from datasets!")
            return
        
        logger.info(f"Total samples available: {len(all_samples)}")
        
        # Limit to requested number
        if len(all_samples) > num_samples:
            all_samples = random.sample(all_samples, num_samples)
        
        # Process samples
        labeled_data = []
        
        logger.info(f"Starting to process {len(all_samples)} screenshots...")
        for idx, sample in enumerate(tqdm(all_samples, desc="Processing screenshots")):
            try:
                # Get image
                image = sample['image']
                if not isinstance(image, Image.Image):
                    continue
                
                # Save screenshot
                screenshot_path = output_path / f"screenshot_{idx:04d}.png"
                image.save(screenshot_path, "PNG")
                
                # Analyze screenshot content
                analysis, confidence = self.classify_screenshot_content(image)
                
                # Assign random work task
                task = random.choice(WORK_TASKS)
                
                # Judge task alignment
                label = self.judge_task_alignment(task, analysis)
                
                labeled_data.append({
                    'screenshot': str(screenshot_path),
                    'task_description': task,
                    'screenshot_analysis': analysis,
                    'label': label,
                    'confidence': confidence,
                    'platform': sample.get('platform', 'unknown')
                })
                
                # Log progress
                if (idx + 1) % 10 == 0:
                    on_task_count = sum(1 for d in labeled_data if d['label'] == 'on-task')
                    logger.info(f"Progress: {idx+1}/{len(all_samples)}, On-task ratio: {on_task_count/len(labeled_data):.2%}")
                    
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
            'on_task_ratio': sum(1 for d in labeled_data if d['label'] == 'on-task') / len(labeled_data),
            'platforms': list(set(d['platform'] for d in labeled_data))
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\nDataset preparation complete!")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"On-task ratio: {metadata['on_task_ratio']:.2%}")
        logger.info(f"Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare desktop screenshot dataset")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to prepare")
    parser.add_argument("--output-dir", default="desktop_task_data", help="Output directory")
    args = parser.parse_args()
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    
    # Prepare dataset
    preparer = DesktopScreenshotPreparer(hf_token)
    preparer.prepare_dataset(args.num_samples, args.output_dir)


if __name__ == "__main__":
    main()