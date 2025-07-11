#!/usr/bin/env python3
"""Task Classification Reward Integration for VERL Training

This module provides the main reward function for task classification training.
Integrates with VERL's reward system and training pipeline.
"""

import logging
from typing import Dict, List, Any, Union

from verl.utils.reward_score.task_classification import TaskClassificationRewardScore

logger = logging.getLogger(__name__)


def task_classification_compute_reward(
    data_source: str,
    solution_str: str, 
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
    """Compute task classification reward compatible with VERL's reward system.
    
    This is the main function that VERL's custom_reward_function will call.
    
    Args:
        data_source: Source identifier (should be "task_classification")
        solution_str: Model's response/prediction
        ground_truth: Ground truth label or metadata dict
        extra_info: Additional information for reward computation
        **kwargs: Additional arguments including reward_weights, vlm_judge, etc.
        
    Returns:
        float: Reward score between 0 and 1
    """
    # Extract configuration from kwargs
    reward_weights = kwargs.get("reward_weights", {
        "binary_classification": 0.5,
        "vlm_judge": 0.5,
        "format_compliance": 0.0
    })
    
    vlm_judge = kwargs.get("vlm_judge", None)
    confidence_threshold = kwargs.get("confidence_threshold", 0.8)
    
    # Create reward model instance
    reward_model = TaskClassificationRewardScore(
        reward_weights=reward_weights,
        vlm_judge=vlm_judge,
        confidence_threshold=confidence_threshold
    )
    
    # Handle different ground truth formats
    if isinstance(ground_truth, dict):
        # Full metadata format
        reward_data = [ground_truth]
    else:
        # Simple string label format
        reward_data = [{"ground_truth": str(ground_truth)}]
    
    # Add extra info if provided
    if extra_info:
        reward_data[0].update(extra_info)
    
    # Compute reward
    rewards = reward_model(
        questions=[""],  # Question is embedded in ground_truth metadata
        responses=[solution_str],
        reward_model=reward_data
    )
    
    return float(rewards[0])