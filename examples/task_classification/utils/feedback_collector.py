"""Feedback Collector for Task Classification

Handles human feedback collection and integration for continuous learning.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects and manages human feedback for task classification."""
    
    def __init__(self, 
                 feedback_dir: str = "feedback",
                 buffer_size: int = 10000,
                 auto_save: bool = True):
        """Initialize feedback collector.
        
        Args:
            feedback_dir: Directory to store feedback data
            buffer_size: Maximum feedback buffer size
            auto_save: Whether to auto-save feedback to disk
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.buffer_size = buffer_size
        self.auto_save = auto_save
        
        # In-memory feedback buffer
        self.feedback_buffer = []
        
        # Statistics
        self.stats = {
            "total_feedback": 0,
            "corrections_by_label": {"on-task": 0, "off-task": 0},
            "avg_confidence_when_wrong": 0.0
        }
        
        # Load existing feedback if available
        self._load_existing_feedback()
    
    def add_feedback(self,
                    screenshot_path: str,
                    task_description: str,
                    model_prediction: str,
                    correct_label: str,
                    model_confidence: float = 0.0,
                    user_id: str = "anonymous",
                    metadata: Dict = None) -> str:
        """Add human feedback correction.
        
        Args:
            screenshot_path: Path to screenshot
            task_description: Task description
            model_prediction: Model's original prediction
            correct_label: Human-provided correct label
            model_confidence: Model's confidence score
            user_id: User who provided feedback
            metadata: Additional metadata
            
        Returns:
            Feedback ID
        """
        # Create feedback item
        feedback_id = f"feedback_{datetime.now().isoformat()}_{len(self.feedback_buffer)}"
        
        feedback_item = {
            "id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "screenshot_path": screenshot_path,
            "task_description": task_description,
            "model_prediction": model_prediction,
            "correct_label": correct_label.lower(),
            "model_confidence": model_confidence,
            "user_id": user_id,
            "was_correction": model_prediction.lower() != correct_label.lower(),
            "metadata": metadata or {}
        }
        
        # Add to buffer
        self.feedback_buffer.append(feedback_item)
        
        # Maintain buffer size
        if len(self.feedback_buffer) > self.buffer_size:
            self.feedback_buffer.pop(0)
        
        # Update statistics
        self._update_stats(feedback_item)
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_feedback_item(feedback_item)
        
        logger.info(f"Added feedback {feedback_id}: {model_prediction} -> {correct_label}")
        
        return feedback_id
    
    def get_recent_corrections(self, limit: int = 100) -> List[Dict]:
        """Get recent correction feedback.
        
        Args:
            limit: Maximum number of corrections to return
            
        Returns:
            List of recent correction feedback items
        """
        corrections = [
            item for item in self.feedback_buffer 
            if item["was_correction"]
        ]
        
        # Sort by timestamp (most recent first)
        corrections.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return corrections[:limit]
    
    def get_feedback_by_confidence(self, 
                                 min_confidence: float = 0.0,
                                 max_confidence: float = 1.0) -> List[Dict]:
        """Get feedback filtered by model confidence range.
        
        Args:
            min_confidence: Minimum model confidence
            max_confidence: Maximum model confidence
            
        Returns:
            List of feedback items in confidence range
        """
        filtered_feedback = [
            item for item in self.feedback_buffer
            if min_confidence <= item["model_confidence"] <= max_confidence
        ]
        
        return filtered_feedback
    
    def get_priority_samples(self, limit: int = 50) -> List[Dict]:
        """Get high-priority samples for training.
        
        Prioritizes samples where:
        1. Model was confident but wrong
        2. Recent corrections
        3. Diverse task descriptions
        
        Args:
            limit: Maximum number of samples
            
        Returns:
            List of priority feedback samples
        """
        # Get high-confidence mistakes (model confident but wrong)
        high_conf_mistakes = [
            item for item in self.feedback_buffer
            if item["was_correction"] and item["model_confidence"] > 0.7
        ]
        
        # Sort by confidence (highest first)
        high_conf_mistakes.sort(key=lambda x: x["model_confidence"], reverse=True)
        
        # Take top samples
        priority_samples = high_conf_mistakes[:limit]
        
        # Fill remaining slots with recent corrections
        if len(priority_samples) < limit:
            recent_corrections = self.get_recent_corrections(limit - len(priority_samples))
            
            # Avoid duplicates
            existing_ids = {item["id"] for item in priority_samples}
            for correction in recent_corrections:
                if correction["id"] not in existing_ids:
                    priority_samples.append(correction)
                    if len(priority_samples) >= limit:
                        break
        
        return priority_samples
    
    def export_training_data(self, output_file: str, format: str = "parquet") -> str:
        """Export feedback as training data.
        
        Args:
            output_file: Output file path
            format: Export format ('parquet', 'json', 'csv')
            
        Returns:
            Path to exported file
        """
        if not self.feedback_buffer:
            logger.warning("No feedback data to export")
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(self.feedback_buffer)
        
        # Prepare for training format
        training_data = []
        for _, row in df.iterrows():
            training_item = {
                "screenshot": row["screenshot_path"],
                "task_description": row["task_description"],
                "label": row["correct_label"],
                "source": "human_feedback",
                "confidence": 1.0,  # High confidence for human labels
                "metadata": {
                    "original_prediction": row["model_prediction"],
                    "model_confidence": row["model_confidence"],
                    "feedback_timestamp": row["timestamp"]
                }
            }
            training_data.append(training_item)
        
        # Export in requested format
        training_df = pd.DataFrame(training_data)
        
        if format == "parquet":
            training_df.to_parquet(output_file)
        elif format == "json":
            training_df.to_json(output_file, orient="records", indent=2)
        elif format == "csv":
            training_df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(training_data)} feedback samples to {output_file}")
        return output_file
    
    def _update_stats(self, feedback_item: Dict):
        """Update feedback statistics."""
        self.stats["total_feedback"] += 1
        
        # Count corrections by label
        if feedback_item["was_correction"]:
            correct_label = feedback_item["correct_label"]
            self.stats["corrections_by_label"][correct_label] = \
                self.stats["corrections_by_label"].get(correct_label, 0) + 1
            
            # Update average confidence when wrong
            if self.stats["total_feedback"] > 1:
                prev_avg = self.stats["avg_confidence_when_wrong"]
                new_conf = feedback_item["model_confidence"]
                self.stats["avg_confidence_when_wrong"] = \
                    (prev_avg * (self.stats["total_feedback"] - 1) + new_conf) / self.stats["total_feedback"]
            else:
                self.stats["avg_confidence_when_wrong"] = feedback_item["model_confidence"]
    
    def _save_feedback_item(self, feedback_item: Dict):
        """Save individual feedback item to disk."""
        try:
            # Save to daily file
            date_str = datetime.now().strftime("%Y-%m-%d")
            feedback_file = self.feedback_dir / f"feedback_{date_str}.jsonl"
            
            with open(feedback_file, "a") as f:
                f.write(json.dumps(feedback_item) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to save feedback item: {e}")
    
    def _load_existing_feedback(self):
        """Load existing feedback from disk."""
        try:
            feedback_files = list(self.feedback_dir.glob("feedback_*.jsonl"))
            
            for feedback_file in feedback_files:
                with open(feedback_file, "r") as f:
                    for line in f:
                        if line.strip():
                            feedback_item = json.loads(line.strip())
                            self.feedback_buffer.append(feedback_item)
                            self._update_stats(feedback_item)
            
            logger.info(f"Loaded {len(self.feedback_buffer)} existing feedback items")
            
        except Exception as e:
            logger.warning(f"Failed to load existing feedback: {e}")
    
    def get_stats(self) -> Dict:
        """Get feedback statistics."""
        correction_rate = 0.0
        if self.stats["total_feedback"] > 0:
            total_corrections = sum(self.stats["corrections_by_label"].values())
            correction_rate = total_corrections / self.stats["total_feedback"]
        
        return {
            **self.stats,
            "buffer_size": len(self.feedback_buffer),
            "correction_rate": correction_rate
        }