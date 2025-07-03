#!/bin/bash
# Script to check Arc Vision data setup on remote instance

echo "=========================================="
echo "Arc Vision Data Setup Checker"
echo "=========================================="

# Check current directory
echo -e "\n1. Current directory:"
pwd

# Check if we're in a container or regular system
echo -e "\n2. System info:"
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
else
    echo "Running on host system"
fi

# Check default data paths from the training script
echo -e "\n3. Checking default data paths from run_arc_vision_grpo.sh:"
echo "Default TRAIN_DATA: /root/data/arc_vision/screenspot/train.parquet"
echo "Default VAL_DATA: /root/data/arc_vision/screenspot/validation.parquet"

# Check if data exists at default location
echo -e "\n4. Checking if data exists at default location:"
if [ -d "/root/data/arc_vision/screenspot" ]; then
    echo "✓ Data directory exists at /root/data/arc_vision/screenspot"
    echo "Contents:"
    ls -la /root/data/arc_vision/screenspot/
    
    # Check file sizes
    echo -e "\nFile sizes:"
    if [ -f "/root/data/arc_vision/screenspot/train.parquet" ]; then
        echo "train.parquet: $(du -h /root/data/arc_vision/screenspot/train.parquet | cut -f1)"
    fi
    if [ -f "/root/data/arc_vision/screenspot/validation.parquet" ]; then
        echo "validation.parquet: $(du -h /root/data/arc_vision/screenspot/validation.parquet | cut -f1)"
    fi
    if [ -f "/root/data/arc_vision/screenspot/test.parquet" ]; then
        echo "test.parquet: $(du -h /root/data/arc_vision/screenspot/test.parquet | cut -f1)"
    fi
else
    echo "✗ Data directory NOT found at /root/data/arc_vision/screenspot"
fi

# Check alternative common locations
echo -e "\n5. Checking alternative data locations:"
ALT_LOCATIONS=(
    "$HOME/data/arc_vision/screenspot"
    "/data/arc_vision/screenspot"
    "/workspace/data/arc_vision/screenspot"
    "./data/arc_vision/screenspot"
    "../data/arc_vision/screenspot"
)

for loc in "${ALT_LOCATIONS[@]}"; do
    if [ -d "$loc" ]; then
        echo "✓ Found data at: $loc"
        ls -la "$loc/" | grep -E "\.parquet$"
    fi
done

# Check if prepare_screenspot_data.py has been run before
echo -e "\n6. Checking for image directories (indicates data was prepared):"
if [ -d "/root/data/arc_vision/screenspot/train_images" ]; then
    echo "✓ train_images directory exists"
    echo "  Number of images: $(find /root/data/arc_vision/screenspot/train_images -name "*.png" | wc -l)"
fi
if [ -d "/root/data/arc_vision/screenspot/validation_images" ]; then
    echo "✓ validation_images directory exists"
    echo "  Number of images: $(find /root/data/arc_vision/screenspot/validation_images -name "*.png" | wc -l)"
fi
if [ -d "/root/data/arc_vision/screenspot/test_images" ]; then
    echo "✓ test_images directory exists"
    echo "  Number of images: $(find /root/data/arc_vision/screenspot/test_images -name "*.png" | wc -l)"
fi

# Suggest next steps
echo -e "\n=========================================="
echo "NEXT STEPS:"
echo "=========================================="

if [ -f "/root/data/arc_vision/screenspot/train.parquet" ] && [ -f "/root/data/arc_vision/screenspot/validation.parquet" ]; then
    echo "✓ Data appears to be ready!"
    echo ""
    echo "To use the existing data:"
    echo "  N_GPUS=2 bash examples/arc_vision/run_arc_vision_grpo.sh"
    echo ""
    echo "To re-prepare with updated prompts:"
    echo "  cd examples/arc_vision"
    echo "  python prepare_screenspot_data.py"
    echo "  cd ../.."
    echo "  N_GPUS=2 bash examples/arc_vision/run_arc_vision_grpo.sh"
else
    echo "✗ Data needs to be prepared!"
    echo ""
    echo "To prepare the data:"
    echo "  cd examples/arc_vision"
    echo "  python prepare_screenspot_data.py --local_dir /root/data/arc_vision/screenspot"
    echo ""
    echo "If you want to use a different directory, update the paths in run_arc_vision_grpo.sh:"
    echo "  TRAIN_DATA=/your/path/train.parquet"
    echo "  VAL_DATA=/your/path/validation.parquet"
fi

echo -e "\n=========================================="