#!/bin/bash
# Install missing dependencies for task classification training

echo "Installing missing dependencies..."

# Install torchvision (required for video processing)
pip install torchvision

# Install matplotlib (required for evaluation plots)
pip install matplotlib

# Install other commonly missing dependencies
pip install seaborn plotly

echo "Dependencies installed successfully!"
echo "You can now run the training pipeline."