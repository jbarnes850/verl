#!/usr/bin/env python3
"""Integration tests for task classification pipeline."""

import unittest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from prepare_task_data import TaskDataPreparer
from utils.vlm_judge import VLMJudge
from utils.visual_verifier import VisualVerifier
from utils.feedback_collector import FeedbackCollector
from utils.synthetic_generator import SyntheticDataGenerator


class TestTaskClassificationIntegration(unittest.TestCase):
    """Integration tests for the complete task classification pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def test_data_preparation_pipeline(self):
        """Test end-to-end data preparation."""
        # Test with mock VLM judge
        with patch.object(VLMJudge, '__init__', return_value=None):
            with patch.object(VLMJudge, 'judge_classification') as mock_judge:
                mock_judge.return_value = ("on-task", 0.9)
                
                preparer = TaskDataPreparer(output_dir=str(self.temp_path))
                preparer.vlm_judge = Mock()
                preparer.vlm_judge.judge_classification = mock_judge
                
                # Generate small synthetic dataset
                train_file, val_file = preparer.generate_synthetic_dataset(
                    num_samples=10,
                    confidence_threshold=0.8
                )
                
                # Verify files were created
                self.assertTrue(Path(train_file).exists())
                self.assertTrue(Path(val_file).exists())
                
                # Verify data format
                train_df = pd.read_parquet(train_file)
                self.assertGreater(len(train_df), 0)
                self.assertIn("task_description", train_df.columns)
                self.assertIn("label", train_df.columns)
                self.assertIn("screenshot", train_df.columns)
    
    def test_feedback_collection_workflow(self):
        """Test feedback collection and synthetic generation workflow."""
        collector = FeedbackCollector(feedback_dir=str(self.temp_path / "feedback"))
        generator = SyntheticDataGenerator(output_dir=str(self.temp_path / "synthetic"))
        
        # Add sample feedback
        feedback_id = collector.add_feedback(
            screenshot_path="/tmp/test_screenshot.png",
            task_description="Working on code review",
            model_prediction="off-task",
            correct_label="on-task",
            model_confidence=0.7
        )
        
        self.assertIsNotNone(feedback_id)
        self.assertEqual(len(collector.feedback_buffer), 1)
        
        # Test feedback retrieval
        recent_corrections = collector.get_recent_corrections(limit=10)
        self.assertEqual(len(recent_corrections), 1)
        
        priority_samples = collector.get_priority_samples(limit=5)
        self.assertEqual(len(priority_samples), 1)
        
        # Test stats
        stats = collector.get_stats()
        self.assertEqual(stats["total_feedback"], 1)
        self.assertGreater(stats["correction_rate"], 0)
    
    def test_visual_verifier_integration(self):
        """Test visual verifier functionality."""
        verifier = VisualVerifier()
        
        # Test with mock image processing
        with patch.object(verifier, '_extract_text') as mock_extract:
            mock_extract.return_value = "github pull request code review"
            
            # Test on-task verification
            passed, confidence, reason = verifier.verify_classification(
                screenshot_path="/tmp/dummy.png",
                task_description="Reviewing code in GitHub",
                predicted_label="on-task"
            )
            
            # Should pass verification for work-related content
            self.assertTrue(passed)
            self.assertGreater(confidence, 0.5)
            self.assertIn("App:", reason)
    
    def test_config_validation(self):
        """Test training configuration validation."""
        config_file = Path(__file__).parent.parent / "config" / "task_classifier_grpo.yaml"
        
        self.assertTrue(config_file.exists(), "Training configuration file missing")
        
        # Read and validate config structure
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = [
            "model", "data", "algorithm", "actor_rollout_ref", 
            "trainer", "semi_online_learning"
        ]
        
        for section in required_sections:
            self.assertIn(section, config, f"Missing config section: {section}")
        
        # Validate GRPO-specific settings
        self.assertEqual(config["algorithm"]["adv_estimator"], "grpo")
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["n"], 5)
        self.assertEqual(config["semi_online_learning"]["sync_steps"], 10)
    
    def test_training_script_exists(self):
        """Test that training script exists and is executable."""
        script_file = Path(__file__).parent.parent / "run_task_classification.sh"
        
        self.assertTrue(script_file.exists(), "Training script missing")
        
        # Check if script is executable
        import stat
        file_stat = script_file.stat()
        self.assertTrue(file_stat.st_mode & stat.S_IEXEC, "Training script not executable")
        
        # Check script content for required components
        with open(script_file, 'r') as f:
            script_content = f.read()
        
        required_elements = [
            "python3 -m verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            "reward_fn=task_classification",
            "data.image_key=images"
        ]
        
        for element in required_elements:
            self.assertIn(element, script_content, f"Missing in script: {element}")
    
    def test_reward_module_integration(self):
        """Test reward module integration with VERL."""
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        
        try:
            from verl.utils.reward_score.task_classification import TaskClassificationRewardScore
            
            # Test reward model creation
            reward_model = TaskClassificationRewardScore()
            self.assertIsNotNone(reward_model)
            
            # Test basic reward computation
            questions = ["Task: Test task\n\nClassification:"]
            responses = ["on-task"]
            ground_truth = [{"ground_truth": "on-task"}]
            
            rewards = reward_model(questions, responses, ground_truth)
            
            self.assertEqual(len(rewards), 1)
            self.assertIsInstance(rewards[0], (int, float))
            self.assertGreaterEqual(rewards[0], 0.0)
            
        except ImportError as e:
            self.fail(f"Reward module not properly integrated: {e}")
    
    def test_dataset_integration(self):
        """Test dataset integration with VERL."""
        # Create test data
        test_data = [
            {
                "screenshot": "/tmp/test.png",
                "task_description": "Test task", 
                "label": "on-task"
            }
        ]
        
        data_file = self.temp_path / "test_data.parquet"
        pd.DataFrame(test_data).to_parquet(data_file)
        
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent.parent))
            from verl.utils.dataset.task_classification_dataset import TaskClassificationDataset
            
            # Mock tokenizer and processor
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]]
            }
            mock_processor = Mock()
            
            with patch('PIL.Image.open'), patch('PIL.Image.new'):
                with patch('verl.utils.dataset.vision_utils.process_image'):
                    dataset = TaskClassificationDataset(
                        data_path=str(data_file),
                        tokenizer=mock_tokenizer,
                        processor=mock_processor
                    )
                    
                    self.assertEqual(len(dataset), 1)
                    
                    # Test data item structure
                    item = dataset[0]
                    required_keys = [
                        "prompt", "response", "input_ids", "attention_mask",
                        "images", "ground_truth"
                    ]
                    
                    for key in required_keys:
                        self.assertIn(key, item, f"Missing key in dataset item: {key}")
                        
        except ImportError as e:
            self.fail(f"Dataset module not properly integrated: {e}")
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestComputeRequirements(unittest.TestCase):
    """Test compute requirement calculations."""
    
    def test_model_memory_requirements(self):
        """Test memory requirement calculations for models."""
        # Qwen2.5-VL-3B model memory requirements
        # Approximate: 3B params * 2 bytes (fp16) = 6GB base
        # With gradients, optimizer states, activations: ~4x = 24GB
        student_model_memory_gb = 24
        
        # Qwen2.5-VL-72B judge model memory requirements  
        # Approximate: 72B params * 2 bytes (fp16) = 144GB base
        # Inference only (no gradients): ~1.5x = 216GB
        judge_model_memory_gb = 216
        
        # GRPO training with group sampling (n=5)
        # Additional memory for multiple rollouts
        grpo_overhead_gb = 8
        
        total_memory_gb = student_model_memory_gb + judge_model_memory_gb + grpo_overhead_gb
        
        # Verify requirements
        self.assertLess(student_model_memory_gb, 80, "Student model should fit on single A100")
        self.assertGreater(judge_model_memory_gb, 80, "Judge model needs multi-GPU setup")
        self.assertGreater(total_memory_gb, 200, "Total memory requirement is substantial")
        
        print(f"Estimated memory requirements:")
        print(f"  Student model (3B): {student_model_memory_gb}GB")
        print(f"  Judge model (72B): {judge_model_memory_gb}GB") 
        print(f"  GRPO overhead: {grpo_overhead_gb}GB")
        print(f"  Total: {total_memory_gb}GB")


if __name__ == "__main__":
    unittest.main()