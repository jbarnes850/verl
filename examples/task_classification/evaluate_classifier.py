#!/usr/bin/env python3
"""Evaluation script for task classification model.

Evaluates trained model performance on validation set and computes metrics.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskClassificationEvaluator:
    """Evaluates task classification model performance."""
    
    def __init__(self, model_dir: str, device: str = "auto"):
        """Initialize evaluator.
        
        Args:
            model_dir: Directory containing trained model
            device: Device for inference
        """
        self.model_dir = Path(model_dir)
        self.device = device
        
        # Will be loaded lazily
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def _load_model(self):
        """Load trained model for evaluation."""
        if self.model is not None:
            return
            
        try:
            # Import here to avoid dependency issues
            from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
            
            logger.info(f"Loading model from {self.model_dir}")
            
            # Load model components
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                torch_dtype="auto",
                device_map=self.device,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_dataset(self, data_file: str, max_samples: int = None) -> Dict:
        """Evaluate model on dataset.
        
        Args:
            data_file: Path to evaluation data (parquet format)
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on dataset: {data_file}")
        
        # Load data
        df = pd.read_parquet(data_file)
        if max_samples:
            df = df.head(max_samples)
        
        logger.info(f"Evaluating {len(df)} samples")
        
        # Load model
        self._load_model()
        
        # Run evaluation
        predictions = []
        ground_truths = []
        confidences = []
        
        for idx, row in df.iterrows():
            try:
                # Get prediction
                pred_label, confidence = self._predict_single(
                    screenshot_path=row.get("screenshot", ""),
                    task_description=row.get("task_description", "")
                )
                
                predictions.append(pred_label)
                confidences.append(confidence)
                ground_truths.append(row.get("label", row.get("ground_truth", "off-task")))
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} samples")
                    
            except Exception as e:
                logger.warning(f"Failed to predict sample {idx}: {e}")
                predictions.append("off-task")  # Conservative default
                confidences.append(0.5)
                ground_truths.append(row.get("label", row.get("ground_truth", "off-task")))
        
        # Compute metrics
        metrics = self._compute_metrics(ground_truths, predictions, confidences)
        
        # Add dataset info
        metrics["dataset_info"] = {
            "data_file": data_file,
            "total_samples": len(df),
            "evaluated_samples": len(predictions)
        }
        
        return metrics
    
    def _predict_single(self, screenshot_path: str, task_description: str) -> Tuple[str, float]:
        """Predict classification for single sample.
        
        Args:
            screenshot_path: Path to screenshot
            task_description: Task description
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        try:
            from PIL import Image
            
            # Create prompt
            prompt = f"""Task: {task_description}

Look at this screenshot and classify if the user is currently on-task or off-task.

Classification:"""
            
            # Load and process image
            if screenshot_path and os.path.exists(screenshot_path):
                image = Image.open(screenshot_path).convert("RGB")
            else:
                # Create dummy image if not found
                image = Image.new("RGB", (224, 224), color="white")
            
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
            
            # Parse classification
            if "on-task" in response or "on_task" in response:
                return "on-task", 0.8
            elif "off-task" in response or "off_task" in response:
                return "off-task", 0.8
            else:
                # Conservative default
                return "off-task", 0.5
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "off-task", 0.5
    
    def _compute_metrics(self, ground_truths: List[str], predictions: List[str], 
                        confidences: List[float]) -> Dict:
        """Compute evaluation metrics.
        
        Args:
            ground_truths: True labels
            predictions: Predicted labels
            confidences: Prediction confidences
            
        Returns:
            Dictionary with metrics
        """
        # Basic accuracy
        accuracy = accuracy_score(ground_truths, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truths, predictions, average=None, labels=["on-task", "off-task"]
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            ground_truths, predictions, average="weighted"
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truths, predictions, labels=["on-task", "off-task"])
        
        # Confidence statistics
        confidence_stats = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences))
        }
        
        # Edge case analysis (low confidence predictions)
        low_confidence_threshold = 0.6
        low_conf_mask = np.array(confidences) < low_confidence_threshold
        if np.any(low_conf_mask):
            low_conf_accuracy = accuracy_score(
                np.array(ground_truths)[low_conf_mask],
                np.array(predictions)[low_conf_mask]
            )
        else:
            low_conf_accuracy = None
        
        metrics = {
            "accuracy": float(accuracy),
            "per_class_metrics": {
                "on-task": {
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "f1": float(f1[0]),
                    "support": int(support[0])
                },
                "off-task": {
                    "precision": float(precision[1]),
                    "recall": float(recall[1]),
                    "f1": float(f1[1]),
                    "support": int(support[1])
                }
            },
            "weighted_averages": {
                "precision": float(precision_weighted),
                "recall": float(recall_weighted),
                "f1": float(f1_weighted)
            },
            "confusion_matrix": cm.tolist(),
            "confidence_stats": confidence_stats,
            "low_confidence_analysis": {
                "threshold": low_confidence_threshold,
                "count": int(np.sum(low_conf_mask)),
                "accuracy": low_conf_accuracy
            }
        }
        
        return metrics
    
    def save_evaluation_report(self, metrics: Dict, output_dir: str):
        """Save evaluation report with visualizations.
        
        Args:
            metrics: Evaluation metrics
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save metrics JSON
        with open(output_path / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        self._create_confusion_matrix_plot(metrics, output_path)
        self._create_metrics_summary_plot(metrics, output_path)
        
        # Create text report
        self._create_text_report(metrics, output_path)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def _create_confusion_matrix_plot(self, metrics: Dict, output_path: Path):
        """Create confusion matrix visualization."""
        cm = np.array(metrics["confusion_matrix"])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=["on-task", "off-task"],
            yticklabels=["on-task", "off-task"]
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_summary_plot(self, metrics: Dict, output_path: Path):
        """Create metrics summary visualization."""
        # Prepare data
        classes = ["on-task", "off-task"]
        precision_scores = [metrics["per_class_metrics"][cls]["precision"] for cls in classes]
        recall_scores = [metrics["per_class_metrics"][cls]["recall"] for cls in classes]
        f1_scores = [metrics["per_class_metrics"][cls]["f1"] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, classes)
        plt.legend()
        plt.ylim(0, 1)
        
        # Add accuracy line
        plt.axhline(y=metrics["accuracy"], color='red', linestyle='--', 
                   label=f'Overall Accuracy: {metrics["accuracy"]:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "metrics_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_text_report(self, metrics: Dict, output_path: Path):
        """Create detailed text report."""
        with open(output_path / "evaluation_report.txt", "w") as f:
            f.write("Task Classification Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall metrics
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n\n")
            
            # Per-class metrics
            f.write("Per-Class Metrics:\n")
            f.write("-" * 20 + "\n")
            for cls in ["on-task", "off-task"]:
                class_metrics = metrics["per_class_metrics"][cls]
                f.write(f"\n{cls.upper()}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {class_metrics['f1']:.4f}\n")
                f.write(f"  Support:   {class_metrics['support']}\n")
            
            # Weighted averages
            f.write(f"\nWeighted Averages:\n")
            f.write("-" * 20 + "\n")
            wa = metrics["weighted_averages"]
            f.write(f"Precision: {wa['precision']:.4f}\n")
            f.write(f"Recall:    {wa['recall']:.4f}\n")
            f.write(f"F1-Score:  {wa['f1']:.4f}\n")
            
            # Confidence analysis
            f.write(f"\nConfidence Analysis:\n")
            f.write("-" * 20 + "\n")
            conf = metrics["confidence_stats"]
            f.write(f"Mean Confidence: {conf['mean']:.4f}\n")
            f.write(f"Std Confidence:  {conf['std']:.4f}\n")
            f.write(f"Min Confidence:  {conf['min']:.4f}\n")
            f.write(f"Max Confidence:  {conf['max']:.4f}\n")
            
            # Low confidence analysis
            low_conf = metrics["low_confidence_analysis"]
            f.write(f"\nLow Confidence Samples (< {low_conf['threshold']}):\n")
            f.write(f"Count: {low_conf['count']}\n")
            if low_conf['accuracy'] is not None:
                f.write(f"Accuracy: {low_conf['accuracy']:.4f}\n")
            else:
                f.write("Accuracy: N/A (no low confidence samples)\n")
            
            # Dataset info
            if "dataset_info" in metrics:
                f.write(f"\nDataset Information:\n")
                f.write("-" * 20 + "\n")
                info = metrics["dataset_info"]
                f.write(f"Data File: {info['data_file']}\n")
                f.write(f"Total Samples: {info['total_samples']}\n")
                f.write(f"Evaluated Samples: {info['evaluated_samples']}\n")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate task classification model")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained model")
    parser.add_argument("--data-file", required=True, help="Evaluation data file (parquet)")
    parser.add_argument("--output-dir", help="Output directory for evaluation report")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--device", default="auto", help="Device for inference")
    
    args = parser.parse_args()
    
    # Set default output directory
    if not args.output_dir:
        args.output_dir = f"{args.model_dir}/evaluation"
    
    # Initialize evaluator
    evaluator = TaskClassificationEvaluator(
        model_dir=args.model_dir,
        device=args.device
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    metrics = evaluator.evaluate_dataset(
        data_file=args.data_file,
        max_samples=args.max_samples
    )
    
    # Print summary
    logger.info("Evaluation Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_averages']['f1']:.4f}")
    
    # Save report
    evaluator.save_evaluation_report(metrics, args.output_dir)
    
    logger.info(f"Evaluation complete! Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()