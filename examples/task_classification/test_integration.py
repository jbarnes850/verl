#!/usr/bin/env python3
"""Test script to verify task classification integration with VERL."""

import sys
import os
from pathlib import Path

# Add VERL to path
sys.path.insert(0, str(Path(__file__).parents[2]))

def test_reward_module():
    """Test if reward module is properly integrated."""
    print("Testing reward module integration...")
    try:
        from verl.utils.reward_score import default_compute_score
        from verl.utils.reward_score.task_classification import TaskClassificationRewardScore, compute_score
        
        # Test compute_score function
        response = "Classification: on-task\nConfidence: 0.9"
        ground_truth = "on-task"
        
        score = compute_score(response, ground_truth)
        print(f"✓ Reward score computed: {score}")
        
        # Test with default_compute_score
        score2 = default_compute_score("task_classification", response, ground_truth)
        print(f"✓ Default compute_score works: {score2}")
        
        # Test reward model class
        reward_model = TaskClassificationRewardScore()
        rewards = reward_model(
            questions=["Test question"],
            responses=[response],
            reward_model=[{"ground_truth": ground_truth}]
        )
        print(f"✓ Reward model class works: {rewards[0]}")
        
        return True
    except Exception as e:
        print(f"✗ Reward module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_module():
    """Test if dataset module is properly integrated."""
    print("\nTesting dataset module integration...")
    try:
        from verl.utils.dataset import TaskClassificationDataset
        from transformers import AutoTokenizer, AutoProcessor
        
        # Create dummy tokenizer and processor
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"Loading tokenizer and processor from {model_path}...")
        
        # Create a dummy data file
        import json
        import tempfile
        import pandas as pd
        
        dummy_data = [
            {
                "prompt": [{"role": "user", "content": "Task: Write a Python script\nClassify if the user is on-task or off-task."}],
                "screenshot": "/path/to/image1.png",
                "task_description": "Write a Python script",
                "label": "on-task",
                "data_source": "task_classification",
                "ground_truth": "on-task"
            },
            {
                "prompt": [{"role": "user", "content": "Task: Review code documentation\nClassify if the user is on-task or off-task."}],
                "screenshot": "/path/to/image2.png", 
                "task_description": "Review code documentation",
                "label": "off-task",
                "data_source": "task_classification",
                "ground_truth": "off-task"
            }
        ]
        
        # Create parquet file
        df = pd.DataFrame(dummy_data)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_file = f.name
            df.to_parquet(temp_file)
        
        try:
            # Mock tokenizer and processor for testing
            class MockTokenizer:
                def __call__(self, text, **kwargs):
                    import torch
                    return {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.tensor([[1, 1, 1]])
                    }
            
            class MockProcessor:
                pass
            
            tokenizer = MockTokenizer()
            processor = MockProcessor()
            
            # Create dataset
            dataset = TaskClassificationDataset(
                data_path=temp_file,
                tokenizer=tokenizer,
                processor=processor,
                max_prompt_length=256,
                max_response_length=16
            )
            
            print(f"✓ Dataset created with {len(dataset)} samples")
            
            # Test getting an item
            item = dataset[0]
            print(f"✓ Dataset item retrieved successfully")
            print(f"  - Prompt: {item['prompt'][:50]}...")
            print(f"  - Response: {item['response']}")
            print(f"  - Ground truth: {item['ground_truth']}")
            
            return True
            
        finally:
            # Clean up
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"✗ Dataset module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_reward_function():
    """Test custom reward function for VERL integration."""
    print("\nTesting custom reward function...")
    try:
        # Import the custom reward function
        sys.path.append(str(Path(__file__).parent))
        from task_classification_reward import task_classification_compute_reward
        
        # Test the function
        reward = task_classification_compute_reward(
            data_source="task_classification",
            solution_str="Classification: on-task",
            ground_truth="on-task",
            reward_weights={
                "binary_classification": 1.0,
                "vlm_judge": 0.0,
                "format_compliance": 0.0
            }
        )
        
        print(f"✓ Custom reward function works: {reward}")
        
        return True
        
    except Exception as e:
        print(f"✗ Custom reward function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Task Classification VERL Integration Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test reward module
    if not test_reward_module():
        all_passed = False
    
    # Test dataset module  
    if not test_dataset_module():
        all_passed = False
    
    # Test custom reward function
    if not test_custom_reward_function():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All integration tests passed!")
        print("The task classification example is properly integrated with VERL.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()