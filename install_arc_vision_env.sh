#!/bin/bash
# Arc Vision Environment Installation Script
# This script installs all dependencies in the correct order to avoid conflicts

set -e  # Exit on error

echo "=================================================="
echo "Arc Vision Environment Installer"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.12"

if [[ ! "$python_version" == "$required_version"* ]]; then
    echo "Error: Python $required_version is required, but $python_version is installed"
    echo "Please install Python 3.12 first"
    exit 1
fi

echo "✓ Python $python_version detected"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv arc
source arc/bin/activate

# Upgrade core tools
echo ""
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install build dependencies first
echo ""
echo "Installing build dependencies..."
pip install numpy==2.3.1
pip install pybind11
pip install packaging>=20.0

# Install PyTorch ecosystem (order matters!)
echo ""
echo "Installing PyTorch and related packages..."
pip install torch==2.7.1
pip install torchvision==0.22.1
pip install torchaudio==2.7.1
pip install torchdata==0.11.0

# Install CUDA-related packages
echo ""
echo "Installing CUDA packages..."
pip install cuda-python==12.9.0

# Install flash-attn with special flags
echo ""
echo "Installing flash-attn (this may take a while)..."
pip install flash-attn==2.8.0.post2 --no-build-isolation

# Install other packages that need special handling
echo ""
echo "Installing flashinfer..."
pip install ninja  # Required for flashinfer
pip install flashinfer-python==0.2.6.post1

# Install transformers and related ML packages
echo ""
echo "Installing ML frameworks..."
pip install transformers==4.52.3
pip install accelerate==1.8.1
pip install peft==0.15.2
pip install tensordict==0.6.2

# Install inference engines
echo ""
echo "Installing inference engines..."
pip install sglang==0.4.7.post1
pip install sgl-kernel==0.1.9
pip install xgrammar==0.1.19

# Install data processing packages
echo ""
echo "Installing data processing packages..."
pip install datasets==3.6.0
pip install pandas==2.3.0
pip install pyarrow==20.0.0
pip install scipy==1.16.0

# Install vision and media packages
echo ""
echo "Installing vision packages..."
pip install pillow==11.3.0
pip install qwen-vl-utils==0.0.11
pip install einops==0.8.1
pip install decord==0.6.0
pip install soundfile==0.13.1

# Install distributed computing
echo ""
echo "Installing distributed computing packages..."
pip install ray==2.47.1

# Install utilities
echo ""
echo "Installing utilities..."
pip install hydra-core==1.3.2
pip install msgspec==0.19.0
pip install orjson==3.10.18
pip install tiktoken==0.9.0
pip install openai==1.93.0
pip install wandb==0.21.0
pip install codetiming
pip install dill
pip install pylatexenc

# Install optimization packages
echo ""
echo "Installing optimization packages..."
pip install torch_memory_saver==0.0.8
pip install torchao==0.9.0

# Install web server packages
echo ""
echo "Installing server packages..."
pip install fastapi==0.115.14
pip install uvicorn==0.35.0
pip install uvloop==0.21.0

# Install VERL in development mode
echo ""
echo "Installing VERL..."
if [ -d "verl" ] && [ -f "verl/setup.py" ]; then
    cd verl
    pip install -e .
    cd ..
else
    echo "Warning: verl directory not found. Skipping VERL installation."
    echo "Make sure to run this script from the parent directory of verl/"
fi

# Verification
echo ""
echo "=================================================="
echo "Verifying installation..."
echo "=================================================="

python -c "
import torch
import transformers
import accelerate
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ Transformers {transformers.__version__}')
print(f'✓ Accelerate {accelerate.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"

# Create a requirements file for reference
echo ""
echo "Creating requirements snapshot..."
pip freeze > arc_environment_snapshot.txt

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "To activate the environment, run:"
echo "  source arc/bin/activate"
echo ""
echo "To start Arc Vision training:"
echo "  cd verl"
echo "  python examples/arc_vision/prepare_screenspot_data.py --local_dir ~/data/arc_vision/screenspot --max_samples 1200 --split_test_data"
echo "  N_GPUS=2 bash examples/arc_vision/run_arc_vision_grpo.sh"
echo ""
echo "Package list saved to: arc_environment_snapshot.txt"
echo "==================================================