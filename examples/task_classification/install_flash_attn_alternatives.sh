#!/bin/bash
# Alternative methods to install flash-attention

echo "Attempting to install flash-attention..."

# Method 1: Try with wheel and setuptools
pip install wheel setuptools
pip install flash-attn --no-build-isolation

if [ $? -ne 0 ]; then
    echo "Method 1 failed. Trying alternative methods..."
    
    # Method 2: Try specific version that might have pre-built wheels
    echo "Trying flash-attn 2.5.8 (stable version)..."
    pip install flash-attn==2.5.8
    
    if [ $? -ne 0 ]; then
        echo "Method 2 failed. Trying from source with different approach..."
        
        # Method 3: Clone and build from source
        echo "Building from source..."
        git clone https://github.com/Dao-AILab/flash-attention.git
        cd flash-attention
        pip install .
        cd ..
        
        if [ $? -ne 0 ]; then
            echo "All methods failed. Flash-attention installation requires:"
            echo "1. CUDA toolkit installed"
            echo "2. Compatible PyTorch version"
            echo "3. C++ compiler"
            echo ""
            echo "You can try:"
            echo "- Check CUDA version: nvidia-smi"
            echo "- Check PyTorch CUDA: python -c 'import torch; print(torch.cuda.is_available())'"
            echo "- Install CUDA development tools if missing"
        fi
    fi
fi

echo "Testing flash-attention import..."
python -c "import flash_attn; print('Flash-attention installed successfully!')" || echo "Flash-attention import failed"