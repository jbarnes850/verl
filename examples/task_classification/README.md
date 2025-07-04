# Task Classification with VLM Judge

A binary screenshot classifier that determines if users are "on-task" or "off-task" based on task descriptions. This implementation uses GRPO training with VLM judge verification to outperform prompt engineering approaches.

## Overview

- **Input**: Screenshot + Task description
- **Output**: Binary classification ("on-task" or "off-task") with confidence score
- **Training**: GRPO with Qwen2.5-VL-3B student model
- **Judge**: Qwen2.5-VL-72B for training data labeling and verification

## Quick Start

```bash
# 1. Prepare initial dataset
python prepare_task_data.py

# 2. Train the classifier
bash run_task_classification.sh

# 3. Evaluate performance
python evaluate_classifier.py

# 4. Serve for inference
python serve_classifier.py
```

## Performance Targets

- **Accuracy**: >92% (vs 70-75% prompt engineering baseline)
- **Latency**: <100ms inference
- **Convergence**: Meaningful improvement within 20-50 feedback samples

## Architecture

- **Student Model**: Qwen2.5-VL-3B-Instruct (fast inference)
- **Judge Model**: Qwen2.5-VL-72B-Instruct (high-quality labeling)
- **Algorithm**: GRPO with group sampling (n=5)
- **Learning**: Semi-online with 10-step sync
- **Rewards**: Hybrid (binary classification + VLM judge + visual verifier)

## Key Features

1. **VLM Judge Bootstrapping**: High-quality initial training data
2. **Synthetic Data Generation**: 10 variants per correction
3. **Real-time Feedback Integration**: Continuous improvement
4. **Visual Verification**: Rule-based UI element detection
5. **Entropy Regularization**: Prevents mode collapse

## Configuration

See `config/task_classifier_grpo.yaml` for full training configuration.