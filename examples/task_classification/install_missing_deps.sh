#!/bin/bash
# Install missing dependencies for task classification training

echo "Installing missing dependencies..."

# Install vLLM (core inference engine - CRITICAL)
pip install vllm

# Install torchvision (required for video processing)
pip install torchvision

# Install matplotlib (required for evaluation plots)
pip install matplotlib

# Install msgspec (required for VERL training)
pip install msgspec

# Install other commonly missing dependencies
pip install seaborn plotly

# Install Ray if not already installed (required for distributed training)
pip install ray[default]

echo "Dependencies installed successfully!"
echo "You can now run the training pipeline."