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
"""Task Classification reward computation for binary screenshot classification."""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def parse_classification_from_response(response: str) -> Tuple[str, float]:
    """Extract binary classification from model response.
    
    Args:
        response: Model's text response
        
    Returns:
        Tuple of (classification, confidence)
    """
    # Clean response
    response = response.strip().lower()
    
    # Multiple patterns for classification extraction
    patterns = [
        r'classification:\s*(on-task|off-task)',
        r'result:\s*(on-task|off-task)',
        r'answer:\s*(on-task|off-task)',
        r'(on-task|off-task)',  # Simple pattern
    ]
    
    classification = None
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            classification = match.group(1)
            break
    
    # Default to off-task if no valid classification found
    if classification not in ["on-task", "off-task"]:
        classification = "off-task"
    
    # Extract confidence if present
    confidence = 0.5  # Default confidence
    conf_patterns = [
        r'confidence:\s*(\d+\.?\d*)',
        r'conf:\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)%?\s*confident',
    ]
    
    for pattern in conf_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                conf_val = float(match.group(1))
                # Normalize to 0-1 range
                if conf_val > 1.0:
                    conf_val = conf_val / 100.0
                confidence = max(0.0, min(1.0, conf_val))
                break
            except:
                continue
    
    return classification, confidence


def validate_classification_format(response: str) -> bool:
    """Check if response follows expected classification format.
    
    Args:
        response: Model response
        
    Returns:
        True if format is valid
    """
    classification, _ = parse_classification_from_response(response)
    return classification in ["on-task", "off-task"]


class TaskClassificationRewardScore:
    """Binary task classification reward model.
    
    Implements hybrid reward structure:
    1. Binary classification accuracy (core task)
    2. VLM judge verification (quality assurance)
    3. Format compliance (proper response structure)
    """
    
    def __init__(self, 
                 reward_weights: Dict[str, float] = None,
                 vlm_judge = None,
                 confidence_threshold: float = 0.8):
        """Initialize task classification reward model.
        
        Args:
            reward_weights: Weights for reward components
            vlm_judge: VLM judge model for verification
            confidence_threshold: Minimum confidence for high-quality samples
        """
        self.weights = reward_weights or {
            "binary_classification": 0.5,
            "vlm_judge": 0.5,
            "format_compliance": 0.0  # Can be enabled later
        }
        self.vlm_judge = vlm_judge
        self.confidence_threshold = confidence_threshold
        
        # Reward values
        self.correct_reward = 1.0
        self.incorrect_reward = 0.0
        self.format_bonus = 0.1
        
    def __call__(self, questions: List[str], responses: List[str], reward_model: Any) -> List[float]:
        """Compute rewards for a batch of responses.
        
        Args:
            questions: List of prompts (screenshot + task description)
            responses: List of model responses  
            reward_model: Ground truth labels and metadata
            
        Returns:
            List of scalar rewards
        """
        rewards = []
        
        for i, (question, response) in enumerate(zip(questions, responses)):
            # Get ground truth label
            if isinstance(reward_model, list):
                gt_data = reward_model[i]
            else:
                gt_data = reward_model
            
            # Extract ground truth
            if isinstance(gt_data, dict):
                gt_label = gt_data.get("ground_truth", gt_data.get("label", "off-task"))
                screenshot = gt_data.get("screenshot", "")
                task_description = gt_data.get("task_description", "")
            else:
                # Assume string label
                gt_label = str(gt_data).lower()
                screenshot = ""
                task_description = ""
            
            # Ensure ground truth is normalized
            gt_label = gt_label.lower()
            if gt_label not in ["on-task", "off-task"]:
                logger.warning(f"Invalid ground truth label: {gt_label}")
                gt_label = "off-task"
            
            # Parse model output
            pred_label, pred_confidence = parse_classification_from_response(response)
            
            # Compute reward components
            reward = self._compute_composite_reward(
                pred_label=pred_label,
                gt_label=gt_label,
                pred_confidence=pred_confidence,
                response=response,
                question=question,
                screenshot=screenshot,
                task_description=task_description
            )
            
            rewards.append(reward)
        
        return rewards
    
    def _compute_composite_reward(self,
                                  pred_label: str,
                                  gt_label: str,
                                  pred_confidence: float,
                                  response: str,
                                  question: str,
                                  screenshot: str = "",
                                  task_description: str = "") -> float:
        """Compute hybrid reward score.
        
        Args:
            pred_label: Predicted classification
            gt_label: Ground truth classification
            pred_confidence: Model's confidence
            response: Full model response
            question: Input question/prompt
            screenshot: Screenshot path/data
            task_description: Task description
            
        Returns:
            Composite reward score
        """
        # Component 1: Binary classification accuracy
        r_binary = self.correct_reward if pred_label == gt_label else self.incorrect_reward
        
        # Component 2: VLM judge verification
        r_judge = 0.0
        if self.vlm_judge and self.weights.get("vlm_judge", 0) > 0:
            try:
                # Get judge verification
                judge_label, judge_confidence = self._get_judge_verification(
                    screenshot, task_description, pred_label
                )
                
                # Reward based on judge agreement
                if judge_label == pred_label:
                    r_judge = judge_confidence
                else:
                    # Penalize disagreement, but less if judge has low confidence
                    r_judge = -0.5 * judge_confidence
                    
            except Exception as e:
                logger.warning(f"VLM judge failed: {e}")
                r_judge = 0.0
        
        # Component 3: Format compliance
        r_format = 0.0
        if self.weights.get("format_compliance", 0) > 0:
            if validate_classification_format(response):
                r_format = self.format_bonus
        
        # Combine components
        total_reward = (
            self.weights["binary_classification"] * r_binary +
            self.weights.get("vlm_judge", 0) * r_judge +
            self.weights.get("format_compliance", 0) * r_format
        )
        
        return float(total_reward)
    
    def _get_judge_verification(self, screenshot: str, task_description: str, 
                              pred_label: str) -> Tuple[str, float]:
        """Get VLM judge verification of prediction.
        
        Args:
            screenshot: Screenshot path or data
            task_description: Task description
            pred_label: Model's prediction
            
        Returns:
            Tuple of (judge_label, judge_confidence)
        """
        if not self.vlm_judge:
            return pred_label, 0.5
        
        # Create judge prompt
        judge_prompt = f"""
        Screenshot shows a user's screen.
        Task assigned: {task_description}
        Model prediction: {pred_label}
        
        Analyze the screenshot and determine if the user is on-task or off-task.
        Consider:
        - Is the visible application relevant to the assigned task?
        - Are there distracting elements or activities visible?
        - Does the screen content align with the task description?
        
        Respond with:
        Classification: [on-task/off-task]
        Confidence: [0.0-1.0]
        """
        
        try:
            # This would call the actual VLM judge
            # For now, return a placeholder
            judge_response = "Classification: on-task\nConfidence: 0.8"
            
            judge_label, judge_confidence = parse_classification_from_response(judge_response)
            return judge_label, judge_confidence
            
        except Exception as e:
            logger.error(f"Judge verification failed: {e}")
            return pred_label, 0.5


def compute_score(response: str, ground_truth: str,
                  reward_weights: Dict[str, float] = None,
                  vlm_judge = None) -> float:
    """Convenience function to compute task classification reward for a single sample.
    
    Args:
        response: Model response
        ground_truth: Ground truth label ("on-task" or "off-task")
        reward_weights: Optional custom reward weights
        vlm_judge: Optional VLM judge for verification
        
    Returns:
        Reward score
    """
    reward_model = TaskClassificationRewardScore(
        reward_weights=reward_weights,
        vlm_judge=vlm_judge
    )
    
    rewards = reward_model(
        questions=[""],  # Question not used directly in computation
        responses=[response],
        reward_model=[{"ground_truth": ground_truth}]
    )
    
    return rewards[0]