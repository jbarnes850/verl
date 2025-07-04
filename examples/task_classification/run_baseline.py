#!/usr/bin/env python3
"""Run zero-shot baseline evaluation on labeled dataset.

This establishes the baseline performance before GRPO training,
allowing us to measure actual performance gains.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_baseline(data_path: str, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """Evaluate zero-shot baseline performance."""
    
    logger.info(f"Loading baseline model: {model_path}")
    
    # Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    logger.info(f"Model loaded on {device}")
    
    # Load validation data
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")
    
    # Evaluate each sample
    correct = 0
    total = 0
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            # Load image
            image = Image.open(row['screenshot'])
            
            # Create zero-shot prompt
            prompt = f"""You are viewing a screenshot of someone's computer screen.

Task: "{row['task_description']}"

Based on what you see in the image, would this activity be considered on-task or off-task for the given work assignment?

Answer with ONLY "on-task" or "off-task":"""
            
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, images=image, return_tensors="pt").to(device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode response
            response = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = response.strip().lower()
            
            # Parse prediction
            if "on-task" in response:
                pred_label = "on-task"
            elif "off-task" in response:
                pred_label = "off-task"
            else:
                pred_label = "unknown"
                logger.warning(f"Unexpected response: {response}")
            
            # Check correctness
            true_label = row['label']
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            predictions.append({
                'idx': idx,
                'task': row['task_description'],
                'true_label': true_label,
                'pred_label': pred_label,
                'correct': is_correct
            })
            
            # Log progress
            if total % 10 == 0:
                current_acc = correct / total * 100
                logger.info(f"Progress: {total}/{len(df)}, Accuracy: {current_acc:.2f}%")
                
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue
    
    # Final results
    accuracy = correct / total * 100 if total > 0 else 0
    
    logger.info("\n" + "="*50)
    logger.info("ZERO-SHOT BASELINE RESULTS")
    logger.info("="*50)
    logger.info(f"Model: {model_path}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Correct predictions: {correct}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info("="*50)
    
    # Save detailed results
    results = {
        'model': model_path,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'predictions': predictions
    }
    
    output_path = Path(data_path).parent / "baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to: {output_path}")
    
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to validation parquet file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model to evaluate")
    args = parser.parse_args()
    
    evaluate_baseline(args.data_path, args.model)