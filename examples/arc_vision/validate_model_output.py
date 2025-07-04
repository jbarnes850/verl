#!/usr/bin/env python3
"""
Validate Arc Vision model output format before training.

This script tests that:
1. The dataset loads correctly with normalized coordinates
2. The model can process the updated prompt format
3. The model outputs coordinates in the expected normalized format (0-1 range)
4. IoU calculation works with the normalized coordinates
"""

import os
import re
import torch
from datasets import load_dataset
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from PIL import Image


def calculate_iou(pred, gt):
    """Calculate IoU between predicted and ground truth bounding boxes."""
    x1 = max(pred[0], gt[0])
    y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2])
    y2 = min(pred[3], gt[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    union = pred_area + gt_area - intersection
    
    return intersection / union if union > 0 else 0


def main():
    print("Arc Vision Model Output Validation")
    print("=" * 50)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = load_dataset(
        'parquet', 
        data_files=os.path.expanduser('~/data/arc_vision/screenspot/train.parquet'), 
        split='train'
    )
    print(f"✓ Dataset loaded with {len(dataset)} samples")
    
    # Check sample format
    sample = dataset[0]
    print(f"\n2. Checking data format...")
    print(f"Ground truth (normalized): {sample['ground_truth']}")
    print(f"Prompt preview: {sample['prompt'][0]['content'][:100]}...")
    
    # Load model and processor
    print("\n3. Loading Qwen2.5-VL-3B model...")
    model_path = 'Qwen/Qwen2.5-VL-3B-Instruct'
    
    # Load the model using Qwen2_5_VLForConditionalGeneration
    # Check if CUDA is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device if torch.cuda.is_available() else None
    )
    
    # Load the processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("✓ Model loaded successfully")
    
    # Test inference
    print("\n4. Testing model inference...")
    image = Image.open(sample['images'][0]['image'])
    
    # The data uses <image> placeholder but Qwen2.5-VL expects <|image_pad|>
    messages = sample['prompt'].copy()
    
    # Replace <image> with the actual image token that Qwen2.5-VL expects
    messages[0]["content"] = messages[0]["content"].replace("<image>", "<|image_pad|>")
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process with the image
    inputs = processor(
        text=text, 
        images=[image], 
        return_tensors='pt'
    ).to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    response = processor.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    print(f"Model response: {response[:200]}...")
    
    # Extract and validate coordinates
    print("\n5. Validating output format...")
    coord_pattern = r'\[[\d.]+,\s*[\d.]+,\s*[\d.]+,\s*[\d.]+\]'
    matches = re.findall(coord_pattern, response)
    
    if matches:
        coords_str = matches[0]
        coords = eval(coords_str)
        print(f"Extracted coordinates: {coords}")
        print(f"Ground truth: {sample['ground_truth']}")
        
        # Check if normalized
        if all(0 <= c <= 1 for c in coords):
            print("✓ Coordinates are normalized (0-1 range)")
            normalized = True
        else:
            print("✗ Warning: Coordinates NOT in normalized range!")
            normalized = False
        
        # Calculate IoU
        iou = calculate_iou(coords, sample['ground_truth'])
        print(f"IoU: {iou:.3f}")
        
        # Summary
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY:")
        print(f"- Dataset format: ✓ OK")
        print(f"- Model loading: ✓ OK")
        print(f"- Coordinate extraction: ✓ OK")
        print(f"- Coordinate normalization: {'✓ OK' if normalized else '✗ FAIL'}")
        print(f"- IoU calculation: ✓ OK ({iou:.3f})")
        
        if normalized and iou > 0:
            print("\n✓ READY FOR TRAINING")
        else:
            print("\n✗ Issues detected - review output above")
    else:
        print("✗ No coordinates found in model response!")
        print("\nPossible issues:")
        print("- Model may need more specific prompting")
        print("- Check if model is following the coordinate format")
        print("- Review the full response above")
    
    print("\nValidation complete.")


if __name__ == "__main__":
    main()