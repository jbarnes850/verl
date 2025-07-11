"""
Minimal custom reward function for Arc Vision POC.
This wraps the arc_vision_reward compute_score function to match VERL's expected interface.
"""

from typing import Union, List, Dict, Any
import json
import os
import sys
import numpy as np

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from verl.utils.reward_score.arc_vision_reward import compute_score


def arc_vision_compute_reward(
    prompt: Union[str, List[Dict[str, Any]]], 
    response: str,
    data_source: str = None,
    **kwargs
) -> float:
    """
    Compute reward for Arc Vision UI detection task.
    
    This function wraps the arc_vision_reward.compute_score to match
    the interface expected by VERL's custom reward function loader.
    
    Args:
        prompt: The prompt (can be string or message list)
        response: The model's response
        data_source: Data source identifier (contains ground truth)
        **kwargs: Additional arguments (may include ground_truth, etc.)
    
    Returns:
        Reward score between 0 and 1
    """
    
    # Default reward if something goes wrong
    default_reward = 0.0
    
    try:
        # Debug: print what we receive
        if os.getenv('DEBUG_REWARD', '').lower() == 'true':
            print(f"[REWARD DEBUG] prompt type: {type(prompt)}")
            print(f"[REWARD DEBUG] response: {response[:100]}...")
            print(f"[REWARD DEBUG] data_source: {data_source}")
            print(f"[REWARD DEBUG] kwargs keys: {list(kwargs.keys())}")
        
        # Extract ground truth from various possible locations
        ground_truth = kwargs.get('ground_truth')
        
        # Try to get from reward_model dict
        if ground_truth is None and 'reward_model' in kwargs:
            reward_model = kwargs['reward_model']
            if isinstance(reward_model, dict):
                ground_truth = reward_model.get('ground_truth')
            elif isinstance(reward_model, str):
                try:
                    reward_model_dict = json.loads(reward_model)
                    ground_truth = reward_model_dict.get('ground_truth')
                except:
                    pass
        
        # Try to parse from data_source if it's a JSON string
        if ground_truth is None and isinstance(data_source, str):
            try:
                data = json.loads(data_source)
                ground_truth = data.get('ground_truth')
            except:
                pass
        
        if ground_truth is None:
            # No ground truth available
            if os.getenv('DEBUG_REWARD', '').lower() == 'true':
                print("[REWARD DEBUG] No ground truth found!")
            return default_reward
        
        # Handle numpy array ground truth (from our ScreenSpot data)
        if isinstance(ground_truth, np.ndarray):
            ground_truth = ground_truth.tolist()
        
        # Convert ground truth to JSON string if it's a list
        if isinstance(ground_truth, list):
            ground_truth_json = json.dumps(ground_truth)
        else:
            ground_truth_json = ground_truth
        
        # Debug ground truth
        if os.getenv('DEBUG_REWARD', '').lower() == 'true':
            print(f"[REWARD DEBUG] Final ground truth: {ground_truth_json}")
        
        # Compute score using the Arc Vision reward function
        score = compute_score(
            response=response,
            ground_truth=ground_truth_json,
            reward_weights=kwargs.get('reward_weights')
        )
        
        # Ensure score is a float
        if hasattr(score, 'item'):
            score = score.item()
        
        return float(score)
        
    except Exception as e:
        print(f"Error in arc_vision_compute_reward: {e}")
        if os.getenv('DEBUG_REWARD', '').lower() == 'true':
            import traceback
            traceback.print_exc()
        return default_reward


# For testing
if __name__ == "__main__":
    # Test the reward function
    test_cases = [
        {
            "prompt": "Find the button",
            "response": "The button is at [0.1, 0.2, 0.3, 0.4]",
            "ground_truth": [0.1, 0.2, 0.3, 0.4]
        },
        {
            "prompt": "Find the button",
            "response": "<confidence>0.8</confidence> <bbox>[0.1, 0.2, 0.3, 0.4]</bbox>",
            "ground_truth": np.array([0.1, 0.2, 0.3, 0.4])  # Test numpy array
        },
        {
            "prompt": "Find the button",
            "response": "No button found",
            "ground_truth": [0.1, 0.2, 0.3, 0.4]
        }
    ]
    
    print("Testing Arc Vision reward function wrapper:")
    for i, test in enumerate(test_cases):
        reward = arc_vision_compute_reward(
            prompt=test["prompt"],
            response=test["response"],
            data_source=None,
            ground_truth=test["ground_truth"]
        )
        print(f"Test {i+1}: {test['response'][:50]}... → Reward: {reward:.3f}")