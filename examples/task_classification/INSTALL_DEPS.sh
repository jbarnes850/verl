#!/bin/bash
# Install additional dependencies for task classification

echo "Installing task classification dependencies..."

# Core dependencies
pip install Pillow  # For PIL/Image
pip install transformers>=4.40.0  # For VLM models
pip install datasets  # For HuggingFace datasets
pip install huggingface-hub  # For HF API access

# Optional but recommended
pip install accelerate  # For faster model loading
pip install sentencepiece  # For tokenization
pip install protobuf  # For some models
pip install torchvision  # Required for Qwen2.5-VL
pip install matplotlib  # For evaluation plots
pip install scikit-learn  # For metrics

echo "Dependencies installed!"
echo "Don't forget to export your HF_TOKEN:"
echo "export HF_TOKEN=your_huggingface_token_here"