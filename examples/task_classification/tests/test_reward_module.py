#!/usr/bin/env python3
"""Unit tests for task classification reward module."""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from verl.utils.reward_score.task_classification import (
    TaskClassificationRewardScore,
    parse_classification_from_response,
    validate_classification_format,
    compute_score
)


class TestTaskClassificationReward(unittest.TestCase):
    """Test cases for task classification reward computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward_model = TaskClassificationRewardScore()
        
        # Sample data
        self.sample_questions = [
            "Task: Working on Jira ticket\n\nLook at this screenshot and classify if the user is currently on-task or off-task.\n\nClassification:"
        ]
        
        self.sample_responses = [
            "on-task",
            "off-task", 
            "Classification: on-task",
            "The user appears to be on-task",
            "invalid response"
        ]
        
        self.sample_ground_truth = [
            {"ground_truth": "on-task"},
            {"ground_truth": "off-task"},
            {"ground_truth": "on-task"},
            {"ground_truth": "on-task"},
            {"ground_truth": "off-task"}
        ]
    
    def test_parse_classification_from_response(self):
        """Test classification parsing from responses."""
        # Test valid responses
        label, conf = parse_classification_from_response("on-task")
        self.assertEqual(label, "on-task")
        self.assertGreater(conf, 0.0)
        
        label, conf = parse_classification_from_response("Classification: off-task")
        self.assertEqual(label, "off-task")
        
        label, conf = parse_classification_from_response("The user is on-task with 90% confidence")
        self.assertEqual(label, "on-task")
        self.assertAlmostEqual(conf, 0.9, places=1)
        
        # Test invalid responses
        label, conf = parse_classification_from_response("invalid")
        self.assertEqual(label, "off-task")  # Should default to off-task
        self.assertEqual(conf, 0.5)
    
    def test_validate_classification_format(self):
        """Test classification format validation."""
        self.assertTrue(validate_classification_format("on-task"))
        self.assertTrue(validate_classification_format("Classification: off-task"))
        self.assertFalse(validate_classification_format("invalid response"))
        self.assertFalse(validate_classification_format(""))
    
    def test_reward_computation(self):
        """Test basic reward computation."""
        # Test correct prediction
        rewards = self.reward_model(
            questions=["test question"],
            responses=["on-task"],
            reward_model=[{"ground_truth": "on-task"}]
        )
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.0)  # Should be positive for correct
        
        # Test incorrect prediction
        rewards = self.reward_model(
            questions=["test question"],
            responses=["on-task"],
            reward_model=[{"ground_truth": "off-task"}]
        )
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # Should be 0 for incorrect
    
    def test_batch_reward_computation(self):
        """Test batch reward computation."""
        responses = ["on-task", "off-task", "on-task"]
        ground_truth = [
            {"ground_truth": "on-task"},   # Correct
            {"ground_truth": "off-task"},  # Correct  
            {"ground_truth": "off-task"}   # Incorrect
        ]
        
        rewards = self.reward_model(
            questions=["test"] * 3,
            responses=responses,
            reward_model=ground_truth
        )
        
        self.assertEqual(len(rewards), 3)
        self.assertGreater(rewards[0], 0.0)  # Correct
        self.assertGreater(rewards[1], 0.0)  # Correct
        self.assertEqual(rewards[2], 0.0)    # Incorrect
    
    def test_compute_score_convenience_function(self):
        """Test convenience function for single sample."""
        score = compute_score("on-task", "on-task")
        self.assertGreater(score, 0.0)
        
        score = compute_score("on-task", "off-task")
        self.assertEqual(score, 0.0)
    
    def test_reward_weights(self):
        """Test custom reward weights."""
        custom_weights = {
            "binary_classification": 1.0,
            "vlm_judge": 0.0  # Disable judge for testing
        }
        
        reward_model = TaskClassificationRewardScore(reward_weights=custom_weights)
        
        # Test that weights are applied
        rewards = reward_model(
            questions=["test"],
            responses=["on-task"],
            reward_model=[{"ground_truth": "on-task"}]
        )
        
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.0)


class TestRewardIntegration(unittest.TestCase):
    """Integration tests for reward system."""
    
    def test_reward_model_creation(self):
        """Test reward model creation with different configurations."""
        # Test default configuration
        reward_model = TaskClassificationRewardScore()
        self.assertIsNotNone(reward_model)
        self.assertIn("binary_classification", reward_model.weights)
        
        # Test custom configuration
        custom_config = {
            "binary_classification": 0.7,
            "vlm_judge": 0.3
        }
        reward_model = TaskClassificationRewardScore(reward_weights=custom_config)
        self.assertEqual(reward_model.weights["binary_classification"], 0.7)
        self.assertEqual(reward_model.weights["vlm_judge"], 0.3)
    
    def test_error_handling(self):
        """Test error handling in reward computation."""
        reward_model = TaskClassificationRewardScore()
        
        # Test with malformed ground truth
        rewards = reward_model(
            questions=["test"],
            responses=["on-task"],
            reward_model=[{"invalid_key": "value"}]
        )
        
        self.assertEqual(len(rewards), 1)
        # Should handle gracefully and return some reward
        self.assertIsInstance(rewards[0], (int, float))


if __name__ == "__main__":
    unittest.main()