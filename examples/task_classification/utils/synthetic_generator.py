"""Synthetic Data Generator for Task Classification

Generates synthetic variants from human feedback for data augmentation.
"""

import logging
import random
from typing import Dict, List, Tuple
from pathlib import Path
import json

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError as e:
    logging.warning(f"Missing PIL dependencies: {e}")

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generates synthetic variants from feedback for data augmentation."""
    
    def __init__(self, 
                 output_dir: str = "synthetic_data",
                 variants_per_sample: int = 10):
        """Initialize synthetic data generator.
        
        Args:
            output_dir: Directory to save synthetic variants
            variants_per_sample: Number of variants to generate per feedback
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.variants_per_sample = variants_per_sample
        
        # Image augmentation parameters
        self.brightness_range = (0.8, 1.2)
        self.contrast_range = (0.8, 1.2)
        self.saturation_range = (0.8, 1.2)
        self.crop_range = (0.0, 0.05)  # Up to 5% crop from edges
        
        # Task description variations
        self.task_synonyms = {
            "working on": ["completing", "handling", "processing", "tackling"],
            "reviewing": ["examining", "checking", "analyzing", "inspecting"],
            "writing": ["creating", "composing", "drafting", "authoring"],
            "coding": ["programming", "developing", "implementing", "building"],
            "debugging": ["troubleshooting", "fixing", "resolving", "solving"],
            "testing": ["validating", "verifying", "checking", "evaluating"],
            "meeting": ["conference", "discussion", "session", "call"],
            "documentation": ["docs", "manual", "guide", "reference"]
        }
    
    def generate_variants(self, 
                         feedback_item: Dict) -> List[Dict]:
        """Generate synthetic variants from feedback item.
        
        Args:
            feedback_item: Original feedback item
            
        Returns:
            List of synthetic variant data items
        """
        variants = []
        
        screenshot_path = feedback_item.get("screenshot_path", "")
        task_description = feedback_item.get("task_description", "")
        correct_label = feedback_item.get("correct_label", "off-task")
        
        if not screenshot_path or not Path(screenshot_path).exists():
            logger.warning(f"Screenshot not found: {screenshot_path}")
            return variants
        
        try:
            # Load original image
            original_image = Image.open(screenshot_path).convert("RGB")
            
            for i in range(self.variants_per_sample):
                # Generate image variant
                variant_image = self._augment_image(original_image, seed=i)
                
                # Generate task description variant
                variant_task = self._augment_task_description(task_description, seed=i)
                
                # Save variant image
                variant_filename = f"variant_{feedback_item.get('id', 'unknown')}_{i:03d}.png"
                variant_path = self.output_dir / variant_filename
                variant_image.save(variant_path)
                
                # Create variant data item
                variant_data = {
                    "screenshot": str(variant_path),
                    "task_description": variant_task,
                    "label": correct_label,
                    "source": "synthetic_variant",
                    "original_feedback_id": feedback_item.get("id"),
                    "variant_index": i,
                    "confidence": 0.9,  # High confidence for synthetic variants
                    "augmentation_applied": self._get_augmentation_info(i)
                }
                
                variants.append(variant_data)
            
            logger.info(f"Generated {len(variants)} variants for feedback {feedback_item.get('id')}")
            
        except Exception as e:
            logger.error(f"Failed to generate variants: {e}")
        
        return variants
    
    def _augment_image(self, image: Image.Image, seed: int = None) -> Image.Image:
        """Apply image augmentations to create variant.
        
        Args:
            image: Original PIL image
            seed: Random seed for reproducible augmentations
            
        Returns:
            Augmented PIL image
        """
        if seed is not None:
            random.seed(seed)
        
        augmented = image.copy()
        
        # Random brightness adjustment
        if random.random() < 0.6:
            enhancer = ImageEnhance.Brightness(augmented)
            factor = random.uniform(*self.brightness_range)
            augmented = enhancer.enhance(factor)
        
        # Random contrast adjustment
        if random.random() < 0.6:
            enhancer = ImageEnhance.Contrast(augmented)
            factor = random.uniform(*self.contrast_range)
            augmented = enhancer.enhance(factor)
        
        # Random saturation adjustment
        if random.random() < 0.4:
            enhancer = ImageEnhance.Color(augmented)
            factor = random.uniform(*self.saturation_range)
            augmented = enhancer.enhance(factor)
        
        # Random crop (slight)
        if random.random() < 0.3:
            width, height = augmented.size
            crop_pct = random.uniform(*self.crop_range)
            crop_pixels = int(min(width, height) * crop_pct)
            
            left = random.randint(0, crop_pixels)
            top = random.randint(0, crop_pixels) 
            right = width - random.randint(0, crop_pixels)
            bottom = height - random.randint(0, crop_pixels)
            
            augmented = augmented.crop((left, top, right, bottom))
            # Resize back to original dimensions
            augmented = augmented.resize((width, height), Image.Resampling.LANCZOS)
        
        # Slight blur (occasionally)
        if random.random() < 0.1:
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return augmented
    
    def _augment_task_description(self, task_description: str, seed: int = None) -> str:
        """Apply text augmentations to task description.
        
        Args:
            task_description: Original task description
            seed: Random seed for reproducible augmentations
            
        Returns:
            Augmented task description
        """
        if seed is not None:
            random.seed(seed)
        
        augmented = task_description
        
        # Apply synonym replacements
        for original, synonyms in self.task_synonyms.items():
            if original in augmented.lower():
                if random.random() < 0.4:  # 40% chance to replace
                    synonym = random.choice(synonyms)
                    # Case-sensitive replacement
                    if original.title() in augmented:
                        augmented = augmented.replace(original.title(), synonym.title())
                    else:
                        augmented = augmented.replace(original, synonym)
        
        # Add context variations
        variations = self._get_context_variations(augmented)
        if variations and random.random() < 0.3:
            augmented = random.choice(variations)
        
        return augmented
    
    def _get_context_variations(self, task_description: str) -> List[str]:
        """Get context variations for task description."""
        variations = [task_description]  # Include original
        
        # Add temporal context
        if "should be" in task_description:
            variations.append(task_description.replace("should be", "needs to be"))
            variations.append(task_description.replace("should be", "is supposed to be"))
        
        # Add urgency context
        if "working on" in task_description.lower():
            variations.append(f"Currently {task_description.lower()}")
            variations.append(f"Focused on {task_description.lower()}")
        
        # Add specificity
        if "ticket" in task_description.lower() and "PROJ-" not in task_description:
            # Don't modify if already has specific ticket ID
            variations.append(task_description.replace("ticket", "ticket PROJ-456"))
        
        return variations
    
    def _get_augmentation_info(self, variant_index: int) -> Dict:
        """Get information about augmentations applied."""
        # This is a simplified version - in practice you'd track actual augmentations
        return {
            "variant_index": variant_index,
            "image_augmented": True,
            "text_augmented": variant_index % 3 == 0,  # Every 3rd variant has text changes
        }
    
    def batch_generate_variants(self, 
                              feedback_items: List[Dict]) -> List[Dict]:
        """Generate variants for multiple feedback items.
        
        Args:
            feedback_items: List of feedback items
            
        Returns:
            List of all generated variants
        """
        all_variants = []
        
        for feedback_item in feedback_items:
            variants = self.generate_variants(feedback_item)
            all_variants.extend(variants)
        
        logger.info(f"Generated {len(all_variants)} total variants from {len(feedback_items)} feedback items")
        
        return all_variants
    
    def export_variants(self, 
                       variants: List[Dict], 
                       output_file: str, 
                       format: str = "parquet") -> str:
        """Export generated variants as training data.
        
        Args:
            variants: List of variant data items
            output_file: Output file path
            format: Export format ('parquet', 'json', 'csv')
            
        Returns:
            Path to exported file
        """
        if not variants:
            logger.warning("No variants to export")
            return ""
        
        try:
            import pandas as pd
            
            df = pd.DataFrame(variants)
            
            if format == "parquet":
                df.to_parquet(output_file)
            elif format == "json":
                df.to_json(output_file, orient="records", indent=2)
            elif format == "csv":
                df.to_csv(output_file, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported {len(variants)} variants to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export variants: {e}")
            return ""
    
    def get_generation_stats(self, variants: List[Dict]) -> Dict:
        """Get statistics on generated variants."""
        if not variants:
            return {"total_variants": 0}
        
        # Count by source feedback
        feedback_ids = {}
        for variant in variants:
            fid = variant.get("original_feedback_id", "unknown")
            feedback_ids[fid] = feedback_ids.get(fid, 0) + 1
        
        # Count by label
        label_counts = {}
        for variant in variants:
            label = variant.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            "total_variants": len(variants),
            "unique_source_feedback": len(feedback_ids),
            "avg_variants_per_feedback": len(variants) / len(feedback_ids) if feedback_ids else 0,
            "label_distribution": label_counts
        }