# Task Classification Setup for Remote GPU

## Quick Start

### 1. Environment Setup
```bash
# Set HuggingFace token for VLM judge
export HF_TOKEN=your_hf_token_here

# Install dependencies (if not already done)
pip install -e ".[all]"
```

### 2. Generate Training Data
```bash
# Generate 1000+ samples with real images + BLIP + VLM judge
python prepare_task_data.py --num-samples 1000 --output-dir training_data

# Quick test with 10 samples
python prepare_task_data.py --num-samples 10 --output-dir test_data
```

### 3. Train Model
```bash
# Train with GRPO following plan.md
bash run_task_classification.sh

# Or manual training
python3 -m verl.trainer.main_ppo \
  data.train_files=training_data/real_train.parquet \
  data.val_files=training_data/real_val.parquet \
  algorithm.adv_estimator=grpo \
  reward_fn=task_classification
```

### 4. Evaluate
```bash
python evaluate_classifier.py --model-path outputs/final_model
```

## Pipeline Components

- **Real Images**: Open-Qwen2VL-Data (Flickr photos)
- **BLIP**: Enhanced image captioning
- **VLM Judge**: Qwen2.5-VL-72B via HuggingFace Inference API
- **Training**: GRPO with group sampling (n=5)

## Expected Performance (plan.md)

- Day 2 (VLM judge bootstrap): 80-85%
- Day 4 (GRPO training): 88-90%  
- Day 7 (semi-online learning): 92-95%

## Files

- `prepare_task_data.py`: Data generation pipeline
- `run_task_classification.sh`: Training script
- `config/task_classifier_grpo.yaml`: GRPO configuration
- `utils/vlm_judge.py`: VLM judge implementation
- `plan.md`: Full implementation plan