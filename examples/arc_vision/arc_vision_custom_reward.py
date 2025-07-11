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
"""Custom reward function for Arc Vision RL training."""

import json
import logging
import time
import os
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
import sys

# Add project root to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


# ==============================================================================
# UTILITY FUNCTIONS TO REDUCE CODE DUPLICATION
# ==============================================================================

def calculate_bbox_area(bbox: List[float]) -> float:
    """Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Area of the bounding box (normalized 0-1)
    """
    x_diff = bbox[2] - bbox[0]
    y_diff = bbox[3] - bbox[1]
    
    # Only fix if clearly invalid (negative or zero area)
    if x_diff <= 0 or y_diff <= 0:
        logger.warning(f"Invalid bbox: {bbox}, using minimum area")
        return 0.01  # 1% minimum area to prevent division issues
    
    return x_diff * y_diff


def classify_object_difficulty(bbox: List[float], bbox_area: float = None) -> Tuple[bool, bool]:
    """Classify object difficulty based on size and position.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        bbox_area: Pre-calculated bbox area (optional)
        
    Returns:
        Tuple of (is_small_object, is_edge_object)
    """
    if bbox_area is None:
        bbox_area = calculate_bbox_area(bbox)
    
    is_small_object = bbox_area < 0.05  # Less than 5% of screen
    is_edge_object = (bbox[0] < 0.1 or bbox[1] < 0.1 or 
                      bbox[2] > 0.9 or bbox[3] > 0.9)
    
    return is_small_object, is_edge_object


def get_performance_category(iou: float) -> str:
    """Categorize detection performance based on IoU.
    
    Args:
        iou: Intersection over Union score
        
    Returns:
        Performance category string
    """
    if iou > 0.8:
        return "excellent"
    elif iou > 0.5:
        return "good"
    elif iou > 0.1:
        return "poor"
    else:
        return "failed"


def extract_tool_usage(response: str) -> Dict[str, Any]:
    """Extract tool usage information from response.
    
    Args:
        response: Model's complete response
        
    Returns:
        Dictionary with tool usage metrics
    """
    tool_metrics = {
        "tool_invocations": 0,
        "tools_used": []
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
        tool_metrics["tool_invocations"] = len(tool_calls)
        
        # Extract tool names
        tool_mapping = {
            "zoom": "zoom",
            "wait": "wait", 
            "inspect": "inspect"
        }
        
        for call in tool_calls:
            call_lower = call.lower()
            for keyword, tool_name in tool_mapping.items():
                if keyword in call_lower:
                    tool_metrics["tools_used"].append(tool_name)
                    break  # Only match first tool per call
    
    return tool_metrics


# ==============================================================================
# DETAILED LOGGING SYSTEM FOR ARC VISION RL MONITORING
# ==============================================================================

def setup_detailed_logging(output_dir: str = "outputs/arc_vision") -> Dict[str, str]:
    """Setup detailed logging directories and return file paths.
    
    TODO: Called once at training start to setup logging infrastructure
    """
    base_dir = Path(output_dir) / "detailed_logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    log_files = {
        "reasoning_traces": str(base_dir / "reasoning_traces.jsonl"),
        "tool_effectiveness": str(base_dir / "tool_effectiveness.jsonl"),
        "tool_patterns": str(base_dir / "tool_patterns.jsonl"),
        "contradictions": str(base_dir / "contradictions.jsonl")
    }
    
    # Initialize log files with headers
    for log_type, file_path in log_files.items():
        if not Path(file_path).exists():
            with open(file_path, 'w') as f:
                f.write(f"# {log_type.upper()} LOG - Arc Vision RL\n")
    
    logger.info(f"Detailed logging setup complete: {base_dir}")
    return log_files


def extract_reasoning_section(response: str) -> str:
    """Extract reasoning section from model response.
    
    TODO: Parses <reasoning>...</reasoning> tags from model output
    """
    reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
    match = re.search(reasoning_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback to <think>...</think> tags
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""


def log_reasoning_trace(prompt_str: str, response_str: str, actual_iou: float, 
                       ground_truth: List[float], log_file: str) -> None:
    """Log detailed reasoning traces for analysis.
    
    TODO: Captures reasoning traces to identify listener disagreement patterns
    """
    try:
        # Extract reasoning and tool information
        reasoning = extract_reasoning_section(response_str)
        tool_metrics = extract_tool_usage(response_str)
        # Confidence tracking removed - not used in reward calculation
        
        trace_data = {
            "timestamp": time.time(),
            "prompt_length": len(prompt_str),
            "response_length": len(response_str),
            "reasoning_text": reasoning,
            "reasoning_length": len(reasoning),
            "tools_used": tool_metrics["tools_used"],
            "tool_invocations": tool_metrics["tool_invocations"],
            "actual_iou": actual_iou,
            "ground_truth_bbox": ground_truth,
            # TODO: Add bbox parsing from response for complete analysis
            "has_reasoning": len(reasoning) > 0,
            "has_tool_calls": tool_metrics["tool_invocations"] > 0
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(trace_data) + '\n')
            
    except Exception as e:
        logger.warning(f"Failed to log reasoning trace: {e}")


def track_tool_effectiveness(tool_used: bool, actual_iou: float, 
                            ground_truth: List[float], response_str: str, log_file: str) -> None:
    """Track tool usage effectiveness based on objective metrics.
    
    TODO: Monitors tool effectiveness for different object sizes and positions
    """
    try:
        tool_metrics = extract_tool_usage(response_str)
        
        # Calculate objective difficulty metrics
        bbox_area = calculate_bbox_area(ground_truth)
        is_small_object, is_edge_object = classify_object_difficulty(ground_truth, bbox_area)
        
        effectiveness_data = {
            "timestamp": time.time(),
            "tool_used": tool_used,
            "tools_used": tool_metrics["tools_used"],
            "tool_count": tool_metrics["tool_invocations"],
            "actual_iou": actual_iou,
            "bbox_area": bbox_area,
            "is_small_object": is_small_object,
            "is_edge_object": is_edge_object,
            "tool_effective": tool_used and actual_iou > 0.5,
            # Performance categories for analysis
            "performance_category": get_performance_category(actual_iou),
            "difficulty_category": (
                "hard" if is_small_object else
                "medium" if is_edge_object else
                "easy"
            )
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(effectiveness_data) + '\n')
            
    except Exception as e:
        logger.warning(f"Failed to track tool effectiveness: {e}")


def monitor_tool_patterns(response_str: str, actual_iou: float, 
                         ground_truth: List[float], log_file: str) -> None:
    """Monitor tool usage patterns for effectiveness analysis.
    
    TODO: Analyzes tool selection patterns and effectiveness
    """
    try:
        tool_metrics = extract_tool_usage(response_str)
        # Confidence tracking removed - not used in reward calculation
        
        # Analyze tool effectiveness based on IoU improvement
        tool_effective = (
            tool_metrics["tool_invocations"] > 0 and 
            actual_iou > 0.5  # Tool effective if it helped achieve decent IoU
        )
        
        pattern_data = {
            "timestamp": time.time(),
            "tools_used": tool_metrics["tools_used"],
            "tool_count": tool_metrics["tool_invocations"],
            # Confidence tracking removed
            "actual_iou": actual_iou,
            "tool_effective": tool_effective,
            "ground_truth_area": calculate_bbox_area(ground_truth) if isinstance(ground_truth, (list, tuple)) and len(ground_truth) >= 4 else 0.0,
            # Tool usage analysis
            "used_zoom": "zoom" in tool_metrics["tools_used"],
            "used_wait": "wait" in tool_metrics["tools_used"],
            "used_inspect": "inspect" in tool_metrics["tools_used"],
            "multiple_tools": len(tool_metrics["tools_used"]) > 1,
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(pattern_data) + '\n')
            
    except Exception as e:
        logger.warning(f"Failed to monitor tool patterns: {e}")


def detect_tool_contradictions(response_str: str, actual_iou: float, 
                              ground_truth: List[float], log_file: str) -> None:
    """Detect contradictions between tool decisions and outcomes.
    
    TODO: Identifies listener disagreement patterns and reasoning contradictions
    """
    try:
        tool_metrics = extract_tool_usage(response_str)
        # Confidence tracking removed - not used in reward calculation
        reasoning = extract_reasoning_section(response_str)
        
        # Detect various contradiction patterns
        contradictions = {
            # Tool used but high performance without need
            "unnecessary_tool_use": (
                tool_metrics["tool_invocations"] > 0 and 
                actual_iou > 0.8
            ),
            # No tool used but poor performance suggests need
            "missed_tool_opportunity": (
                tool_metrics["tool_invocations"] == 0 and 
                actual_iou < 0.3 and
                True  # Always check for missed opportunities
            ),
            # Tool used but no confidence improvement
            "ineffective_tool_use": (
                tool_metrics["tool_invocations"] > 0 and
                actual_iou < 0.5  # Tool ineffective if poor result
            ),
            # High confidence but poor performance
            # Confidence-based contradictions removed
            # Reasoning suggests uncertainty but no tools used
            "reasoning_tool_mismatch": (
                len(reasoning) > 0 and
                any(phrase in reasoning.lower() for phrase in 
                    ["unclear", "difficult", "hard to see", "not sure"]) and
                tool_metrics["tool_invocations"] == 0
            )
        }
        
        # Only log if contradictions detected
        if any(contradictions.values()):
            contradiction_data = {
                "timestamp": time.time(),
                "actual_iou": actual_iou,
                # Confidence tracking removed
                "tools_used": tool_metrics["tools_used"],
                "tool_count": tool_metrics["tool_invocations"],
                "reasoning_length": len(reasoning),
                "contradictions": contradictions,
                "contradiction_count": sum(contradictions.values()),
                "reasoning_snippet": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(contradiction_data) + '\n')
                
    except Exception as e:
        logger.warning(f"Failed to detect contradictions: {e}")


# Global variable to store log file paths (initialized once)
_LOG_FILES = None


def get_log_files() -> Dict[str, str]:
    """Get or initialize log file paths."""
    global _LOG_FILES
    if _LOG_FILES is None:
        _LOG_FILES = setup_detailed_logging()
    return _LOG_FILES


def arc_vision_compute_reward(data_source: str, 
                            solution_str: str, 
                            ground_truth: Any, 
                            extra_info: Dict = None,
                            reward_weights: Dict[str, float] = None,
                            tool_penalties: Dict[str, float] = None,
                            **kwargs):
    """Custom reward function for Arc Vision that integrates with VERL's reward manager.
    
    Implements the complete 3-component reward function from the blog post:
    R(s,a,t) = α*R_task + β*R_tool + γ*R_gate
    
    Where:
    - R_task: IoU-based detection accuracy
    - R_tool: Tool effectiveness based on objective difficulty (bbox size, position)
    - R_gate: Penalties for tool misuse
    
    Args:
        data_source: Dataset identifier (should be "arc_vision" or "screenspot")
        solution_str: Model response string containing reasoning and tool usage
        ground_truth: Ground truth bounding box coordinates [x1, y1, x2, y2]
        extra_info: Additional information (unused for Arc Vision)
        # confidence_threshold removed - using objective metrics instead
        reward_weights: Weights for reward components (default: α=0.6, β=0.3, γ=0.1)
        tool_penalties: Penalties for different tool usage failure modes
        **kwargs: Additional keyword arguments
        
    Returns:
        Dict: Reward score with detailed metrics
    """
    # Verify this is an Arc Vision request
    # Handle case where data_source might be passed as array/list
    if hasattr(data_source, '__len__') and not isinstance(data_source, str):
        # If it's an array/list, take the first element
        data_source = data_source[0] if len(data_source) > 0 else "unknown"
    
    if data_source not in ["arc_vision", "screenspot", "rootsautomation/ScreenSpot"]:
        logger.warning(f"Arc Vision reward function called with data_source: {data_source}")
    
    # Handle JSON string extra_info (from parquet storage)
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except (json.JSONDecodeError, TypeError):
            extra_info = {}
    
    # Use default parameters from blog post if not provided
    if reward_weights is None:
        reward_weights = {"task": 0.6, "tool": 0.3, "gate": 0.1}
    if tool_penalties is None:
        tool_penalties = {
            "unnecessary_tool": -0.5,
            "missed_opportunity": -0.3,
            "ineffective_tool": -0.2,
            "excessive_tools": -0.4
        }
    
    # Initialize reward components
    r_task = 0.0
    r_tool = 0.0
    r_gate = 0.0
    
    # ==============================================================================
    # Multi-Turn Support: Detect response type
    # ==============================================================================
    # Extract tool usage early to check response type
    tool_metrics = extract_tool_usage(solution_str)
    
    # Check if this is a tool-only response (multi-turn first turn)
    # Look for presence of tools and absence of bbox
    has_tools = tool_metrics["tool_invocations"] > 0
    
    # Quick check for bbox presence
    bbox_pattern = r'<bbox>.*?</bbox>|\[\s*[\d\.\s,\-]+\s*\]'
    has_bbox = bool(re.search(bbox_pattern, solution_str, re.IGNORECASE))
    
    # If tool-only response (no bbox), return appropriate reward
    if has_tools and not has_bbox:
        logger.info(f"Tool-only response detected with {tool_metrics['tool_invocations']} tools")
        
        # Calculate objective difficulty from ground truth
        bbox_area = calculate_bbox_area(ground_truth)
        is_small_object, is_edge_object = classify_object_difficulty(ground_truth, bbox_area)
        
        # Give small positive reward for appropriate tool use
        if is_small_object or is_edge_object:
            # Tools are appropriate for difficult objects
            tool_reward = 0.1  # Small positive reward
        else:
            # Tools might be unnecessary for easy objects
            tool_reward = 0.0  # Neutral reward
        
        return {
            "score": float(tool_reward),
            "r_task": 0.0,  # No task completion yet
            "r_tool": float(tool_reward),
            "r_gate": 0.0,
            "iou": 0.0,
            "is_small_object": float(is_small_object),
            "is_edge_object": float(is_edge_object),
            "bbox_area": float(bbox_area),
            "tool_invocations": int(tool_metrics["tool_invocations"]),
            "num_tools_used": len(tool_metrics["tools_used"]),
            "num_gate_penalties": 0,
            "total_gate_penalty": 0.0,
            "response_type": "tool_only"  # Indicator for debugging
        }
    
    # ==============================================================================
    # 1. R_task: IoU-based Detection Accuracy
    # ==============================================================================
    
    # Extract predicted bbox from solution - try multiple formats
    predicted_bbox = None
    iou = 0.0
    
    # Try multiple bbox patterns - Qwen2.5-VL supports various formats
    patterns = [
        r'<bbox>\s*\[([\d\.\s,\-]+)\]\s*</bbox>',  # Tagged format (PREFERRED)
        r'(?:^|\s)\[([\d\.\s,\-]+)\](?:\s|$)',  # Standalone bracket format
        r'"bbox":\s*\[([\d\s,\-]+)\]',  # JSON format
        r'"bbox_2d":\s*\[([\d\s,\-]+)\]',  # JSON format with pixels
        r'bbox:\s*\[([\d\.\s,\-]+)\]',  # Colon format
        r'coordinates[:\s]+\[([\d\.\s,\-]+)\]',  # Alternative wording
        r'box:\s*\[([\d\.\s,\-]+)\]',  # Shortened version
        r'bounding box:\s*\[([\d\.\s,\-]+)\]',  # Full phrase
        r'<click>([\d\s,\-]+)</click>',  # Qwen2.5-VL click format
        r'<point>([\d\s,\-]+)</point>',  # Qwen2.5-VL point format
        r'<box>([\d\.\s,\-]+)</box>',  # Alternative box tag
        r'(?:detect|found|located).*?\[([\d\.\s,\-]+)\]',  # With context words
        r'(?:x1.*?y1.*?x2.*?y2.*?)[\s:=]*([\d\.\s,\-]+)',  # Explicit coordinate names
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.IGNORECASE)
        if match:
            try:
                coords = [float(x.strip()) for x in match.group(1).split(',')]
                
                # Ensure we have exactly 4 coordinates
                if len(coords) != 4:
                    continue
                
                # Fix coordinate ordering if needed (ensure x1 < x2, y1 < y2)
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                
                # Convert pixel coords to normalized if values > 1
                if any(c > 1.0 for c in coords):
                    # Try to infer image dimensions from the coordinates
                    # Most screens are 16:9, so we'll try common resolutions
                    max_x = max(coords[0], coords[2])
                    max_y = max(coords[1], coords[3])
                    
                    # Common resolutions to try
                    resolutions = [
                        (1920, 1080),  # Full HD
                        (1280, 720),   # HD
                        (2560, 1440),  # 2K
                        (3840, 2160),  # 4K
                        (1366, 768),   # Common laptop
                        (1440, 900),   # Common monitor
                        (1600, 900),   # Widescreen
                    ]
                    
                    # Find best matching resolution
                    best_res = (1920, 1080)  # Default
                    for res_x, res_y in resolutions:
                        if max_x <= res_x and max_y <= res_y:
                            best_res = (res_x, res_y)
                            break
                    
                    # Normalize using best resolution
                    predicted_bbox = [
                        coords[0] / best_res[0],
                        coords[1] / best_res[1],
                        coords[2] / best_res[0],
                        coords[3] / best_res[1]
                    ]
                else:
                    predicted_bbox = coords
                break
            except:
                continue
    
    if predicted_bbox:
        try:
            # Validate bbox values are in [0, 1] range
            predicted_bbox = [max(0.0, min(1.0, x)) for x in predicted_bbox]
            
            # Log coordinate conversion for debugging
            logger.debug(f"Original coords from model: {match.group(1) if 'match' in locals() else 'N/A'}")
            logger.debug(f"Converted normalized bbox: {predicted_bbox}")
            logger.debug(f"Ground truth bbox: {ground_truth}")
            
            # Calculate IoU
            x1 = max(predicted_bbox[0], ground_truth[0])
            y1 = max(predicted_bbox[1], ground_truth[1])
            x2 = min(predicted_bbox[2], ground_truth[2])
            y2 = min(predicted_bbox[3], ground_truth[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area_pred = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
                area_gt = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
                union = area_pred + area_gt - intersection
                iou = intersection / union if union > 0 else 0.0
            else:
                iou = 0.0
            
            r_task = iou
        except Exception as e:
            logger.warning(f"Failed to calculate IoU: {e}")
            logger.warning(f"Predicted bbox: {predicted_bbox}, Ground truth: {ground_truth}")
            r_task = 0.0
    else:
        # No bbox found - this might be expected for some responses
        logger.info(f"No bbox found in response (length: {len(solution_str)})")
        # Only warn if there are no tools either (completely empty response)
        if not has_tools:
            logger.warning(f"Response has neither bbox nor tools. First 500 chars: {solution_str[:500]}")
            # Log if response contains any coordinate-like patterns
            any_numbers = re.findall(r'\d+\.?\d*', solution_str)
            if any_numbers:
                logger.warning(f"Found numbers in response: {any_numbers[:10]}")
        r_task = 0.0
    
    # ==============================================================================
    # 2. Calculate Objective Task Properties (tool_metrics already extracted above)
    # ==============================================================================
    # REMOVED confidence extraction - always returns artificial values
    # Instead, calculate objective difficulty from ground truth
    
    # Calculate objective difficulty from ground truth
    bbox_area = calculate_bbox_area(ground_truth)
    is_small_object, is_edge_object = classify_object_difficulty(ground_truth, bbox_area)
    
    # ==============================================================================
    # 3. R_tool: Tool Effectiveness Based on Task Difficulty and Success
    # ==============================================================================
    if tool_metrics["tool_invocations"] > 0:
        # CHANGED: Reward tools based on objective difficulty, not fake confidence
        # Research shows: small objects genuinely need tools, large objects don't
        if is_small_object:
            # Tools are necessary for small objects - strong reward
            r_tool = 0.4 * iou  # High multiplier for appropriate tool use
        elif is_edge_object:
            # Tools helpful for edge objects - moderate reward  
            r_tool = 0.2 * iou
        else:
            # Tools less necessary for large, centered objects - small reward
            r_tool = 0.1 * iou  # Still some reward if it helped (high IoU)
        
        # Cap maximum tool reward
        r_tool = min(r_tool, 0.4)
    else:
        # No tools used - small reward if performed well without tools
        if not is_small_object and iou > 0.7:
            r_tool = 0.1  # Reward efficiency on easy objects
        else:
            r_tool = 0.0
    
    # ==============================================================================
    # 4. R_gate: Penalties for Tool Misuse Based on Objective Criteria
    # ==============================================================================
    gate_penalties = []
    
    # Penalty 1: Missed tools on genuinely hard objects
    # CHANGED: Use objective difficulty instead of fake confidence
    if is_small_object and tool_metrics["tool_invocations"] == 0 and iou < 0.5:
        # Small object, no tools, poor performance = bad decision
        penalty = tool_penalties["missed_opportunity"]
        gate_penalties.append(("missed_hard_object", penalty))
    
    # Penalty 2: Excessive tool use on easy objects  
    # CHANGED: Penalize based on object size/difficulty, not confidence
    if not is_small_object and not is_edge_object and tool_metrics["tool_invocations"] > 1:
        # Large centered object but multiple tools = inefficient
        penalty = tool_penalties["unnecessary_tool"] * 0.5  # Softer penalty
        gate_penalties.append(("excessive_easy_object", penalty))
    
    # Penalty 3: Wrong tool selection
    # NEW: Penalize using wrong tool for the situation
    if is_small_object and "wait" in tool_metrics["tools_used"] and "zoom" not in tool_metrics["tools_used"]:
        # Small object needs zoom, not wait
        penalty = -0.05
        gate_penalties.append(("wrong_tool_choice", penalty))
    
    # Penalty 4: Excessive tool use (more than 2 tools)
    # KEPT: This is still valid - too many tools is inefficient
    if tool_metrics["tool_invocations"] > 2:
        penalty = tool_penalties["excessive_tools"]
        gate_penalties.append(("excessive_tools", penalty))
    
    # Sum all gate penalties (they are negative values)
    r_gate = sum(penalty for _, penalty in gate_penalties)
    
    # ==============================================================================
    # 5. Compute Final Reward: R(s,a,t) = α*R_task + β*R_tool + γ*R_gate
    # ==============================================================================
    final_reward = (
        reward_weights["task"] * r_task + 
        reward_weights["tool"] * r_tool + 
        reward_weights["gate"] * r_gate
    )
    
    # Do not clamp - allow negative rewards to propagate for proper RL signal
    
    # Safety check: Ensure final reward is finite
    if not np.isfinite(final_reward):
        logger.error(f"Non-finite reward detected! Components: r_task={r_task}, r_tool={r_tool}, r_gate={r_gate}")
        logger.error(f"Weights: {reward_weights}, Ground truth: {ground_truth}")
        final_reward = -1.0  # Penalize but keep training stable
    
    
    # Log reward statistics for debugging
    logger.info(f"Arc Vision reward breakdown - Task: {r_task:.3f}, Tool: {r_tool:.3f}, Gate: {r_gate:.3f}")
    logger.info(f"Object: {'small' if is_small_object else 'large'}, {'edge' if is_edge_object else 'center'}, Tools: {tool_metrics['tool_invocations']}")
    logger.info(f"Final reward: {final_reward:.3f} (IoU: {iou:.3f})")
    
    # Return detailed reward information
    # VERL's reward manager expects a dict with at least a 'score' key
    # Note: All values must be numeric for validation metrics computation
    return {
        "score": float(final_reward),
        # Additional metrics for analysis
        "r_task": float(r_task),
        "r_tool": float(r_tool),
        "r_gate": float(r_gate),
        "iou": float(iou),
        "is_small_object": float(is_small_object),  # 1.0 or 0.0
        "is_edge_object": float(is_edge_object),    # 1.0 or 0.0
        "bbox_area": float(bbox_area),               # Object size
        "tool_invocations": int(tool_metrics["tool_invocations"]),
        # Convert non-numeric values to counts/flags for metrics
        "num_tools_used": len(tool_metrics["tools_used"]),
        "num_gate_penalties": len(gate_penalties),
        "total_gate_penalty": float(r_gate),  # Sum of all penalties
        # Keep these for logging but not in metrics
        # "tools_used": tool_metrics["tools_used"],
        # "gate_penalties": gate_penalties,
        # "predicted_bbox": predicted_bbox,
        # "ground_truth": ground_truth
        "response_type": "bbox_response"  # Indicator for debugging
    }


def create_arc_vision_compute_score_fn(reward_weights: Dict[str, float] = None,
                                      tool_penalties: Dict[str, float] = None):
    """Create a compute_score function configured for Arc Vision.
    
    This factory function creates a compute_score function with the Arc Vision
    parameters pre-configured. This allows VERL's reward manager to use it
    directly without needing to pass custom parameters.
    
    Args:
        reward_weights: Weights for reward components (task, tool, gate)
        tool_penalties: Penalties for different tool usage failure modes
        
    Returns:
        Function that matches VERL's compute_score interface
    """
    def compute_score(data_source: str, solution_str: str, ground_truth: Any, 
                     extra_info: Dict = None, **kwargs) -> Dict:
        return arc_vision_compute_reward(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_weights=reward_weights,
            tool_penalties=tool_penalties,
            **kwargs
        )
    
    return compute_score


# Create the default Arc Vision compute score function
arc_vision_compute_score_fn = create_arc_vision_compute_score_fn(
    reward_weights={"task": 0.6, "tool": 0.3, "gate": 0.1},
    tool_penalties={
        "unnecessary_tool": -0.5,
        "missed_opportunity": -0.3,
        "ineffective_tool": -0.2,
        "excessive_tools": -0.4
    }
)