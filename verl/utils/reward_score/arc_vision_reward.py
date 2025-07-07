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
"""Arc Vision RL reward computation for UI element detection with tool learning."""

import json
import re
from typing import Dict, List, Any, Tuple
import numpy as np


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute Intersection over Union between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0 and 1
    """
    # Ensure numpy arrays
    bbox1 = np.array(bbox1, dtype=np.float32)
    bbox2 = np.array(bbox2, dtype=np.float32)
    
    # Compute intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Compute union
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union <= 0:
        return 0.0
    
    return float(intersection / union)


def parse_bbox_from_response(response: str) -> Tuple[np.ndarray, bool]:
    """Extract bounding box from model response.
    
    Args:
        response: Model's text response
        
    Returns:
        Tuple of (bbox array, success flag)
    """
    # Try multiple patterns for bbox extraction
    patterns = [
        r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]',  # [x1, y1, x2, y2]
        r'bbox:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]',  # bbox: [x1, y1, x2, y2]
        r'<bbox>\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]</bbox>',  # <bbox>[x1, y1, x2, y2]</bbox>
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                bbox = np.array([float(x) for x in match.groups()], dtype=np.float32)
                # Validate bbox (coordinates should be normalized 0-1)
                if np.all(bbox >= 0) and np.all(bbox <= 1) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    return bbox, True
            except:
                continue
    
    return np.array([0, 0, 0, 0], dtype=np.float32), False


def parse_tool_usage(response: str) -> Dict[str, Any]:
    """Extract tool usage information from response.
    
    Args:
        response: Model's text response
        
    Returns:
        Dictionary with tool usage info
    """
    tool_info = {
        "tool_used": False,
        "tool_name": None,
        "tool_calls": 0
    }
    
    # Check for tool calls in multiple formats
    tool_patterns = [
        r'<use_tool>(.*?)</use_tool>',  # New prompt format
        r'<tool_call>(.*?)</tool_call>'  # Legacy format
    ]
    
    tool_calls = []
    for pattern in tool_patterns:
        tool_calls.extend(re.findall(pattern, response, re.DOTALL))
    
    if tool_calls:
        tool_info["tool_used"] = True
        tool_info["tool_calls"] = len(tool_calls)
        
        # Extract tool names
        tool_names = []
        for call in tool_calls:
            if "zoom" in call.lower():
                tool_names.append("zoom")
            elif "wait" in call.lower():
                tool_names.append("wait")
            elif "inspect" in call.lower():
                tool_names.append("inspect")
        
        if tool_names:
            tool_info["tool_name"] = tool_names[0]  # First tool used
            tool_info["all_tools"] = tool_names  # All tools used
    
    return tool_info


class ArcVisionRewardScore:
    """Arc Vision RL composite reward model.
    
    Implements the 3-component reward structure:
    1. Task performance (IoU)
    2. Tool effectiveness (based on objective difficulty)
    3. Gating penalty (prevent tool abuse)
    """
    
    def __init__(self, 
                 reward_weights: Dict[str, float] = None,
                 tool_penalties: Dict[str, float] = None):
        """Initialize Arc Vision reward model.
        
        Args:
            reward_weights: Weights for reward components
            tool_penalties: Penalties for different failure modes
        """
        self.weights = reward_weights or {
            "task": 0.6,
            "tool": 0.3,
            "gate": 0.1
        }
        self.penalties = tool_penalties or {
            "unnecessary_tool": -0.5,
            "missed_opportunity": -0.3,
            "ineffective_tool": -0.2,
            "excessive_tools": -0.4
        }
    
    def __call__(self, questions: List[str], responses: List[str], reward_model: Any) -> List[float]:
        """Compute rewards for a batch of responses.
        
        Args:
            questions: List of prompts/questions
            responses: List of model responses
            reward_model: Dictionary containing ground truth and other metadata
            
        Returns:
            List of scalar rewards
        """
        rewards = []
        
        for i, (question, response) in enumerate(zip(questions, responses)):
            # Get ground truth bbox
            if isinstance(reward_model, list):
                gt_data = reward_model[i]
            else:
                gt_data = reward_model
            
            # Extract ground truth bbox
            if "ground_truth" in gt_data:
                gt_bbox = np.array(json.loads(gt_data["ground_truth"]), dtype=np.float32)
            else:
                # Skip if no ground truth
                rewards.append(0.0)
                continue
            
            # Parse model output
            pred_bbox, bbox_success = parse_bbox_from_response(response)
            tool_info = parse_tool_usage(response)
            
            # Compute reward components
            reward = self._compute_composite_reward(
                pred_bbox=pred_bbox,
                gt_bbox=gt_bbox,
                bbox_success=bbox_success,
                tool_info=tool_info
            )
            
            rewards.append(reward)
        
        return rewards
    
    def _compute_composite_reward(self,
                                  pred_bbox: np.ndarray,
                                  gt_bbox: np.ndarray,
                                  bbox_success: bool,
                                  tool_info: Dict[str, Any]) -> float:
        """Compute the 3-component composite reward.
        
        Args:
            pred_bbox: Predicted bounding box
            gt_bbox: Ground truth bounding box
            bbox_success: Whether bbox was successfully parsed
            tool_info: Tool usage information
            
        Returns:
            Composite reward score
        """
        # Component 1: Task performance (IoU-based)
        if bbox_success:
            r_task = compute_iou(pred_bbox, gt_bbox)
        else:
            r_task = 0.0  # Failed to produce valid bbox
        
        # Calculate objective difficulty based on ground truth bbox
        bbox_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
        is_small_object = bbox_area < 0.05  # Less than 5% of screen
        is_edge_object = (gt_bbox[0] < 0.1 or gt_bbox[1] < 0.1 or 
                          gt_bbox[2] > 0.9 or gt_bbox[3] > 0.9)
        
        # Component 2: Tool effectiveness based on objective difficulty
        r_tool = 0.0
        if tool_info["tool_used"]:
            if is_small_object:
                # Tools are necessary for small objects
                r_tool = 0.4 * r_task  # Scale by task success
            elif is_edge_object:
                # Tools helpful for edge objects
                r_tool = 0.2 * r_task
            else:
                # Tools less necessary for large, centered objects
                r_tool = 0.1 * r_task
            
            # Cap maximum tool reward
            r_tool = min(r_tool, 0.4)
            
            # Penalty for excessive tool use
            if tool_info["tool_calls"] > 2:
                r_tool += self.penalties["excessive_tools"] * (tool_info["tool_calls"] - 2)
        else:
            # No tools used - small reward if performed well without tools
            if not is_small_object and r_task > 0.7:
                r_tool = 0.1  # Reward efficiency on easy objects
        
        # Component 3: Gating penalty based on objective criteria
        r_gate = 0.0
        
        # Penalty for missing tools on genuinely hard objects
        if is_small_object and not tool_info["tool_used"] and r_task < 0.5:
            r_gate += self.penalties["missed_opportunity"]
        
        # Penalty for excessive tool use on easy objects
        if not is_small_object and not is_edge_object and tool_info["tool_calls"] > 1:
            r_gate += self.penalties["unnecessary_tool"] * 0.5  # Softer penalty
        
        # Combine components
        total_reward = (
            self.weights["task"] * r_task +
            self.weights["tool"] * r_tool +
            self.weights["gate"] * r_gate
        )
        
        return float(total_reward)


def compute_score(response: str, ground_truth: str, 
                  reward_weights: Dict[str, float] = None) -> float:
    """Convenience function to compute Arc Vision reward for a single sample.
    
    Args:
        response: Model response
        ground_truth: Ground truth bbox as JSON string
        reward_weights: Optional custom reward weights
        
    Returns:
        Reward score
    """
    reward_model = ArcVisionRewardScore(
        reward_weights=reward_weights
    )
    
    rewards = reward_model(
        questions=[""],  # Question not used in computation
        responses=[response],
        reward_model=[{"ground_truth": ground_truth}]
    )
    
    return rewards[0]