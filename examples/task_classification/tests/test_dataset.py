#!/usr/bin/env python3
"""Unit tests for task classification dataset."""

import unittest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from verl.utils.dataset.task_classification_dataset import TaskClassificationDataset


class TestTaskClassificationDataset(unittest.TestCase):
    """Test cases for task classification dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test data
        self.test_data = [
            {
                "screenshot": "/path/to/screenshot1.png",
                "task_description": "Working on Jira ticket PROJ-123",
                "label": "on-task"
            },
            {
                "screenshot": "/path/to/screenshot2.png", 
                "task_description": "Reviewing code in GitHub",
                "label": "on-task"
            },
            {
                "screenshot": "/path/to/screenshot3.png",
                "task_description": "Writing documentation",
                "label": "off-task"
            }
        ]
        
        # Create temporary data file
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = Path(self.temp_dir) / "test_data.parquet"
        df = pd.DataFrame(self.test_data)
        df.to_parquet(self.data_file)
        
        # Mock tokenizer and processor
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1]]
        }
        
        self.mock_processor = Mock()
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = TaskClassificationDataset(
            data_path=str(self.data_file),
            tokenizer=self.mock_tokenizer,
            processor=self.mock_processor
        )
        
        self.assertEqual(len(dataset.data), 3)
        self.assertEqual(len(dataset), 3)  # No synthetic data initially
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        with patch('PIL.Image.open'), patch('PIL.Image.new') as mock_new:
            mock_image = Mock()
            mock_new.return_value = mock_image
            
            with patch('verl.utils.dataset.vision_utils.process_image') as mock_process:
                mock_process.return_value = "processed_image"
                
                dataset = TaskClassificationDataset(
                    data_path=str(self.data_file),
                    tokenizer=self.mock_tokenizer,
                    processor=self.mock_processor
                )
                
                item = dataset[0]
                
                # Check required keys
                self.assertIn("prompt", item)
                self.assertIn("response", item)
                self.assertIn("input_ids", item)
                self.assertIn("attention_mask", item)
                self.assertIn("images", item)
                self.assertIn("ground_truth", item)
                
                # Check values
                self.assertEqual(item["ground_truth"], "on-task")
                self.assertIn("Jira ticket", item["prompt"])
    
    def test_feedback_integration(self):
        """Test feedback addition and synthetic data generation."""
        with patch('PIL.Image.open'), patch('PIL.Image.new'):
            with patch('verl.utils.dataset.vision_utils.process_image'):
                dataset = TaskClassificationDataset(
                    data_path=str(self.data_file),
                    tokenizer=self.mock_tokenizer,
                    processor=self.mock_processor,
                    synthetic_variants_per_feedback=2  # Reduced for testing
                )
                
                initial_size = len(dataset)
                
                # Add feedback
                dataset.add_feedback(
                    screenshot_path="/tmp/test_screenshot.png",
                    task_description="Test task",
                    correct_label="on-task",
                    model_prediction="off-task",
                    confidence=0.8
                )
                
                # Check feedback buffer
                self.assertEqual(len(dataset.feedback_buffer), 1)
                
                # Check synthetic data generation
                self.assertGreater(len(dataset.synthetic_data), 0)
                self.assertGreater(len(dataset), initial_size)
    
    def test_feedback_stats(self):
        """Test feedback statistics."""
        with patch('PIL.Image.open'), patch('PIL.Image.new'):
            with patch('verl.utils.dataset.vision_utils.process_image'):
                dataset = TaskClassificationDataset(
                    data_path=str(self.data_file),
                    tokenizer=self.mock_tokenizer,
                    processor=self.mock_processor
                )
                
                # Add some feedback
                dataset.add_feedback("/tmp/test1.png", "Task 1", "on-task")
                dataset.add_feedback("/tmp/test2.png", "Task 2", "off-task")
                
                stats = dataset.get_feedback_stats()
                
                self.assertEqual(stats["total_feedback"], 2)
                self.assertIn("label_distribution", stats)
                self.assertIn("synthetic_variants", stats)
    
    def test_prompt_template(self):
        """Test prompt template formatting."""
        dataset = TaskClassificationDataset(
            data_path=str(self.data_file),
            tokenizer=self.mock_tokenizer,
            processor=self.mock_processor
        )
        
        task_desc = "Working on Jira ticket"
        expected_prompt = f"""Task: {task_desc}

Look at this screenshot and classify if the user is currently on-task or off-task.

Classification:"""
        
        formatted = dataset.prompt_template.format(task_description=task_desc)
        self.assertEqual(formatted, expected_prompt)
    
    def test_label_normalization(self):
        """Test that labels are properly normalized."""
        # Test data with mixed case labels
        mixed_case_data = [
            {"screenshot": "/test.png", "task_description": "Test", "label": "ON-TASK"},
            {"screenshot": "/test2.png", "task_description": "Test", "label": "Off-Task"},
            {"screenshot": "/test3.png", "task_description": "Test", "label": "invalid_label"}
        ]
        
        temp_file = Path(self.temp_dir) / "mixed_case.parquet"
        pd.DataFrame(mixed_case_data).to_parquet(temp_file)
        
        with patch('PIL.Image.open'), patch('PIL.Image.new'):
            with patch('verl.utils.dataset.vision_utils.process_image'):
                dataset = TaskClassificationDataset(
                    data_path=str(temp_file),
                    tokenizer=self.mock_tokenizer,
                    processor=self.mock_processor
                )
                
                # Check normalization
                item1 = dataset[0]
                item2 = dataset[1]
                item3 = dataset[2]
                
                self.assertEqual(item1["ground_truth"], "on-task")
                self.assertEqual(item2["ground_truth"], "off-task")
                self.assertEqual(item3["ground_truth"], "off-task")  # Invalid -> off-task
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()