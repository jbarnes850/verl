#!/bin/bash
# Install missing dependencies for task classification training

echo "Installing missing dependencies..."

# CRITICAL: Install correct transformers version (must be < 4.53)
echo "Installing transformers < 4.53 (required by VERL)..."
pip install "transformers<4.53"

# Install vLLM (core inference engine - CRITICAL)
pip install vllm

# Install prerequisites for flash-attention
echo "Installing build prerequisites..."
pip install wheel setuptools

# Install flash-attention (CRITICAL for VERL training)
# Note: This requires CUDA and may take time to compile
echo "Installing flash-attention (this may take several minutes)..."
pip install flash-attn --no-build-isolation

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

# Install other potential missing deps
pip install ninja packaging

echo "Dependencies installed successfully!"
echo "You can now run the training pipeline."