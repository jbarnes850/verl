#!/usr/bin/env python3
"""Task Classification Reward Integration for VERL Training

This module provides the main reward function for task classification training.
Integrates with VERL's reward system and training pipeline.
"""

import sys
from pathlib import Path

# Add utils to path for imports
sys.path.append(str(Path(__file__).parent / "utils"))

from verl.utils.reward_score.task_classification import TaskClassificationRewardScore
from utils.vlm_judge import VLMJudge
from utils.visual_verifier import VisualVerifier


def create_reward_model(config=None):
    """Create task classification reward model for VERL training.
    
    Args:
        config: Training configuration
        
    Returns:
        Reward model instance
    """
    # Initialize VLM judge if configured
    vlm_judge = None
    if config and config.get("judge", {}).get("enable", False):
        try:
            judge_model_path = config["judge"].get("model_path", "Qwen/Qwen2.5-VL-72B-Instruct")
            vlm_judge = VLMJudge(model_path=judge_model_path)
        except Exception as e:
            print(f"Warning: Failed to initialize VLM judge: {e}")
    
    # Get reward weights from config
    reward_weights = None
    if config and "algorithm" in config:
        reward_weights = config["algorithm"].get("reward_weights", {
            "binary_classification": 0.5,
            "vlm_judge": 0.5
        })
    
    # Create reward model
    reward_model = TaskClassificationRewardScore(
        reward_weights=reward_weights,
        vlm_judge=vlm_judge
    )
    
    return reward_model


def compute_task_classification_reward(questions, responses, reward_model_data, config=None):
    """Main reward function for task classification.
    
    This function will be called by VERL training pipeline.
    
    Args:
        questions: List of input prompts
        responses: List of model responses
        reward_model_data: Ground truth data
        config: Training configuration
        
    Returns:
        List of reward scores
    """
    # Create reward model
    reward_model = create_reward_model(config)
    
    # Compute rewards
    rewards = reward_model(questions, responses, reward_model_data)
    
    return rewards