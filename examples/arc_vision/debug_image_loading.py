#!/usr/bin/env python3
"""
Debug script to understand why Qwen2.5-VL isn't seeing images
"""

import os
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor

def main():
    print("Debug Image Loading for Qwen2.5-VL")
    print("=" * 50)
    
    # Load dataset
    dataset = load_dataset(
        'parquet', 
        data_files=os.path.expanduser('~/data/arc_vision/screenspot/train.parquet'), 
        split='train'
    )
    
    # Check first few samples
    for i in range(3):
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]
        
        # Check image path
        image_path = sample['images'][0]['image']
        print(f"Image path: {image_path}")
        print(f"Image exists: {os.path.exists(image_path)}")
        
        # Check prompt content
        prompt = sample['prompt'][0]['content']
        print(f"\nPrompt preview: {prompt[:200]}...")
        print(f"Contains <|image_pad|>: {'<|image_pad|>' in prompt}")
        print(f"Contains <image>: {'<image>' in prompt}")
        
        # Try to load image
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                print(f"Image loaded successfully: {img.size}")
            except Exception as e:
                print(f"Error loading image: {e}")
        
        # Test processor
        try:
            processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)
            
            # Check what the processor expects
            print(f"\nProcessor image token: {processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')}")
            print(f"Processor vision start token: {processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')}")
            print(f"Processor vision end token: {processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')}")
            
            # Try processing with the prompt as-is
            messages = sample['prompt'].copy()
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"\nChat template output preview: {text[:200]}...")
            
            # Check if we need to manually replace tokens
            if '<image>' in messages[0]['content'] and '<|image_pad|>' not in messages[0]['content']:
                messages[0]['content'] = messages[0]['content'].replace('<image>', '<|image_pad|>')
                text2 = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                print(f"\nAfter token replacement: {text2[:200]}...")
                
        except Exception as e:
            print(f"Error with processor: {e}")
            
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()