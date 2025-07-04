#!/usr/bin/env python3
"""Quick test script to debug data pipeline hanging."""

import os
import sys
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_loading():
    """Test if dataset loading works."""
    logger.info("Testing dataset loading...")
    try:
        dataset = load_dataset("weizhiwang/Open-Qwen2VL-Data", split="train", streaming=True)
        logger.info("✅ Dataset loaded successfully")
        
        # Test iteration
        logger.info("Testing iteration (first 5 items)...")
        for i, item in enumerate(dataset):
            if i >= 5:
                break
            logger.info(f"Item {i}: url={item.get('url', 'N/A')[:50]}...")
            
        logger.info("✅ Dataset iteration works")
        return True
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return False

def test_hf_api():
    """Test if HF API works."""
    logger.info("\nTesting HF Inference API...")
    
    if not os.getenv("HF_TOKEN"):
        logger.error("❌ HF_TOKEN not set")
        return False
        
    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(
            provider="auto",
            api_key=os.getenv("HF_TOKEN"),
        )
        
        logger.info("✅ HF client created")
        
        # Test simple text completion
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Say 'test' and nothing else."}
                    ]
                }
            ],
            stream=False,
            max_tokens=10
        )
        
        logger.info(f"✅ API response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        logger.error(f"❌ HF API test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== Data Pipeline Debug Test ===")
    
    # Test dataset
    dataset_ok = test_dataset_loading()
    
    # Test API
    api_ok = test_hf_api()
    
    if dataset_ok and api_ok:
        logger.info("\n✅ All tests passed! The issue might be with BLIP or processing logic.")
    else:
        logger.info("\n❌ Some tests failed. Check the errors above.")