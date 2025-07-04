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
"""Task Classification Dataset for binary screenshot classification."""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

import torch
from PIL import Image
import pandas as pd

from torch.utils.data import Dataset
from verl.utils.dataset.vision_utils import process_image

logger = logging.getLogger(__name__)


class TaskClassificationDataset(Dataset):
    """Dataset for binary task classification with screenshots.
    
    Handles:
    - Screenshot images + task descriptions as inputs
    - Binary labels (on-task/off-task) as targets
    - Synthetic data generation from feedback
    - VLM judge bootstrap labeling
    """
    
    def __init__(self,
                 data_path: str,
                 tokenizer,
                 processor,
                 config=None,
                 max_prompt_length: int = 256,
                 max_response_length: int = 16,
                 image_key: str = "images",
                 feedback_buffer_size: int = 10000,
                 synthetic_variants_per_feedback: int = 10):
        """Initialize task classification dataset.
        
        Args:
            data_path: Path to dataset file (parquet, jsonl, or json)
            tokenizer: Model tokenizer
            processor: Vision processor
            config: Dataset configuration (DictConfig)
            max_prompt_length: Maximum prompt tokens
            max_response_length: Maximum response tokens  
            image_key: Key for image data in dataset
            feedback_buffer_size: Size of feedback buffer
            synthetic_variants_per_feedback: Number of synthetic variants per feedback
        """
        from omegaconf import DictConfig
        
        # Create config if not provided
        if config is None:
            config = DictConfig({
                "max_prompt_length": max_prompt_length,
                "max_response_length": max_response_length,
                "image_key": image_key,
                "prompt_key": "prompt",
                "cache_dir": "~/.cache/verl/task_classification"
            })
        
        # Store max lengths from config
        self.max_prompt_length = config.get("max_prompt_length", max_prompt_length)
        self.max_response_length = config.get("max_response_length", max_response_length)
        
        # Store data path for loading
        self.data_path = data_path
        
        # Store attributes
        self.processor = processor
        self.image_key = image_key
        self.feedback_buffer_size = feedback_buffer_size
        self.synthetic_variants_per_feedback = synthetic_variants_per_feedback
        self.tokenizer = tokenizer
        self.config = config
        
        # Set dataset attributes
        self.cache_dir = config.get("cache_dir", "~/.cache/verl/task_classification")
        self.prompt_key = config.get("prompt_key", "prompt")
        
        # Feedback buffer for continuous learning
        self.feedback_buffer = []
        self.synthetic_data = []
        self.data = []
        
        # Load dataset
        self._load_data()
        
        # Classification prompt template
        self.prompt_template = """Task: {task_description}

Look at this screenshot and classify if the user is currently on-task or off-task.

Classification:"""
        
        logger.info(f"Loaded {len(self.data)} samples for task classification")
    
    def _load_data(self):
        """Load task classification data from file."""
        file_path = Path(self.data_path)
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            self.data = []
            return
        
        try:
            if file_path.suffix == ".parquet":
                df = pd.read_parquet(self.data_path)
                self.data = df.to_dict('records')
            elif file_path.suffix == ".jsonl":
                self.data = []
                with open(self.data_path, 'r') as f:
                    for line in f:
                        self.data.append(json.loads(line.strip()))
            elif file_path.suffix == ".json":
                with open(self.data_path, 'r') as f:
                    self.data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_path}: {e}")
            self.data = []
    
    def __len__(self) -> int:
        """Return total dataset size including synthetic data."""
        return len(self.data) + len(self.synthetic_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single dataset item."""
        # Determine if this is original or synthetic data
        if idx < len(self.data):
            item = self.data[idx]
        else:
            synthetic_idx = idx - len(self.data)
            item = self.synthetic_data[synthetic_idx]
        
        # Extract components
        screenshot_path = item.get("screenshot", item.get("image_path", ""))
        task_description = item.get("task_description", "")
        label = item.get("label", item.get("ground_truth", "off-task"))
        
        # Ensure label is normalized
        label = label.lower()
        if label not in ["on-task", "off-task"]:
            label = "off-task"
        
        # Create prompt
        prompt = self.prompt_template.format(task_description=task_description)
        
        # Process image
        try:
            if screenshot_path and os.path.exists(screenshot_path):
                image = Image.open(screenshot_path).convert("RGB")
                # Use vision_utils for consistent processing
                processed_image = process_image(image)
            else:
                # Create dummy image if screenshot not found
                image = Image.new("RGB", (224, 224), color="white")
                processed_image = process_image(image)
                logger.warning(f"Screenshot not found: {screenshot_path}")
        except Exception as e:
            logger.error(f"Failed to process image {screenshot_path}: {e}")
            # Create dummy image
            image = Image.new("RGB", (224, 224), color="white")
            processed_image = process_image(image)
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt"
        )
        
        # Tokenize response (ground truth label)
        response_tokens = self.tokenizer(
            label,
            padding=False,
            truncation=True,
            max_length=self.max_response_length,
            return_tensors="pt"
        )
        
        return {
            "prompt": prompt,
            "response": label,
            "input_ids": prompt_tokens["input_ids"].squeeze(0),
            "attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "response_input_ids": response_tokens["input_ids"].squeeze(0),
            "response_attention_mask": response_tokens["attention_mask"].squeeze(0),
            self.image_key: processed_image,
            "screenshot_path": screenshot_path,
            "task_description": task_description,
            "ground_truth": label,
            "metadata": {
                "original_index": idx,
                "is_synthetic": idx >= len(self.data)
            }
        }
    
    def add_feedback(self, 
                    screenshot_path: str,
                    task_description: str, 
                    correct_label: str,
                    model_prediction: str = None,
                    confidence: float = 0.0):
        """Add human feedback for continuous learning.
        
        Args:
            screenshot_path: Path to screenshot
            task_description: Task description
            correct_label: Correct classification from human
            model_prediction: Model's original prediction
            confidence: Model's confidence score
        """
        feedback_item = {
            "screenshot": screenshot_path,
            "task_description": task_description,
            "label": correct_label.lower(),
            "model_prediction": model_prediction,
            "confidence": confidence,
            "timestamp": pd.Timestamp.now().isoformat(),
            "type": "human_feedback"
        }
        
        # Add to feedback buffer
        self.feedback_buffer.append(feedback_item)
        
        # Maintain buffer size
        if len(self.feedback_buffer) > self.feedback_buffer_size:
            self.feedback_buffer.pop(0)
        
        # Generate synthetic variants
        self._generate_synthetic_variants(feedback_item)
        
        logger.info(f"Added feedback: {correct_label} for task '{task_description}'")
    
    def _generate_synthetic_variants(self, feedback_item: Dict[str, Any]):
        """Generate synthetic variants from feedback item.
        
        Args:
            feedback_item: Original feedback item
        """
        try:
            screenshot_path = feedback_item["screenshot"]
            task_description = feedback_item["task_description"]
            label = feedback_item["label"]
            
            if not os.path.exists(screenshot_path):
                logger.warning(f"Cannot generate variants - screenshot not found: {screenshot_path}")
                return
            
            # Load original image
            original_image = Image.open(screenshot_path).convert("RGB")
            
            # Generate variants with augmentations
            for i in range(self.synthetic_variants_per_feedback):
                # Apply image augmentations
                variant_image = self._augment_image(original_image)
                
                # Apply task description variations
                variant_task = self._augment_task_description(task_description)
                
                # Save variant image (temporary)
                variant_path = f"/tmp/variant_{len(self.synthetic_data)}_{i}.png"
                variant_image.save(variant_path)
                
                # Create synthetic data item
                synthetic_item = {
                    "screenshot": variant_path,
                    "task_description": variant_task,
                    "label": label,
                    "type": "synthetic_variant",
                    "original_feedback_id": len(self.feedback_buffer) - 1
                }
                
                self.synthetic_data.append(synthetic_item)
        
        except Exception as e:
            logger.error(f"Failed to generate synthetic variants: {e}")
    
    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Apply image augmentations for synthetic variants.
        
        Args:
            image: Original PIL image
            
        Returns:
            Augmented PIL image
        """
        import random
        from PIL import ImageEnhance
        
        # Copy image
        augmented = image.copy()
        
        # Random brightness adjustment (±20%)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(augmented)
            factor = random.uniform(0.8, 1.2)
            augmented = enhancer.enhance(factor)
        
        # Random contrast adjustment (±20%)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(augmented)
            factor = random.uniform(0.8, 1.2)
            augmented = enhancer.enhance(factor)
        
        # Random crop (up to 5% from edges)
        if random.random() < 0.3:
            width, height = augmented.size
            crop_pct = random.uniform(0.0, 0.05)
            crop_pixels = int(min(width, height) * crop_pct)
            
            left = random.randint(0, crop_pixels)
            top = random.randint(0, crop_pixels)
            right = width - random.randint(0, crop_pixels)
            bottom = height - random.randint(0, crop_pixels)
            
            augmented = augmented.crop((left, top, right, bottom))
            augmented = augmented.resize((width, height))
        
        return augmented
    
    def _augment_task_description(self, task_description: str) -> str:
        """Apply text augmentations to task description.
        
        Args:
            task_description: Original task description
            
        Returns:
            Augmented task description
        """
        import random
        
        # Simple text variations to maintain semantic meaning
        variations = [
            task_description,  # Original (50% chance)
            task_description.replace("working on", "completing"),
            task_description.replace("reviewing", "examining"),
            task_description.replace("writing", "creating"),
            task_description.replace("coding", "programming"),
            task_description.replace("developing", "building"),
        ]
        
        # Add context variations
        if "should be" in task_description:
            variations.append(task_description.replace("should be", "needs to be"))
            variations.append(task_description.replace("should be", "is supposed to be"))
        
        return random.choice(variations)
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback buffer statistics."""
        if not self.feedback_buffer:
            return {"total_feedback": 0, "synthetic_variants": 0}
        
        # Count by label
        label_counts = {}
        for item in self.feedback_buffer:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            "total_feedback": len(self.feedback_buffer),
            "synthetic_variants": len(self.synthetic_data),
            "label_distribution": label_counts,
            "buffer_utilization": len(self.feedback_buffer) / self.feedback_buffer_size
        }


def collate_fn(batch):
    """Collate function for TaskClassificationDataset compatible with VERL."""
    # Separate different components
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    response_input_ids = [item['response_input_ids'] for item in batch]
    response_attention_mask = [item['response_attention_mask'] for item in batch]
    images = [item.get('images', None) for item in batch]
    
    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    response_input_ids = pad_sequence(response_input_ids, batch_first=True, padding_value=0)
    response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
    
    # Stack images if present
    if images[0] is not None:
        images = torch.stack(images)
    else:
        images = None
    
    # Create batch dict
    batch_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'response_input_ids': response_input_ids,
        'response_attention_mask': response_attention_mask,
        'prompts': [item['prompt'] for item in batch],
        'responses': [item['response'] for item in batch],
        'ground_truths': [item['ground_truth'] for item in batch],
        'metadata': [item.get('metadata', {}) for item in batch]
    }
    
    if images is not None:
        batch_dict['images'] = images
    
    return batch_dict