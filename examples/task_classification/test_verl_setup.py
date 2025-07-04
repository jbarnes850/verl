#!/usr/bin/env python3
"""Test basic VERL setup and dependencies."""

print("Testing VERL setup...")

# Test basic imports
try:
    import verl
    print("✓ VERL imported successfully")
except ImportError as e:
    print(f"✗ Failed to import VERL: {e}")

try:
    import vllm
    print("✓ vLLM imported successfully")
except ImportError as e:
    print(f"✗ Failed to import vLLM: {e}")

try:
    import ray
    print("✓ Ray imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Ray: {e}")

try:
    import torch
    print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers imported successfully (version: {transformers.__version__})")
except ImportError as e:
    print(f"✗ Failed to import Transformers: {e}")

try:
    import msgspec
    print("✓ msgspec imported successfully")
except ImportError as e:
    print(f"✗ Failed to import msgspec: {e}")

# Test data loading
try:
    import pandas as pd
    df = pd.read_parquet("desktop_task_data_balanced/desktop_train.parquet")
    print(f"✓ Training data loaded: {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")
    print(f"  On-task ratio: {(df['label'] == 'on-task').mean():.2%}")
except Exception as e:
    print(f"✗ Failed to load training data: {e}")

print("\nSetup test complete!")