#!/usr/bin/env python3
"""Test which image format works with VERL's vision utils."""

from verl.utils.dataset.vision_utils import process_image
from PIL import Image
import tempfile
import os

print("Testing VERL image format compatibility...\n")

# Create a dummy image
dummy_image = Image.new('RGB', (100, 100), color='red')
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
    dummy_image.save(tmp.name)
    temp_path = tmp.name

print(f"Created temporary image at: {temp_path}")

# Test different formats
test_formats = [
    {"image": temp_path},
    {"image_url": f"file://{temp_path}"},
    {"image_url": temp_path},
    {"path": temp_path},
    {"file": temp_path},
]

for i, format_dict in enumerate(test_formats):
    print(f"\nTest {i+1}: {format_dict}")
    try:
        result = process_image(format_dict)
        print(f"  ✓ Success! Result type: {type(result)}")
        if hasattr(result, 'size'):
            print(f"  Image size: {result.size}")
    except Exception as e:
        print(f"  ✗ Failed with error: {type(e).__name__}: {e}")

# Also test with PIL Image directly
print(f"\nTest with PIL Image directly:")
try:
    result = process_image(dummy_image)
    print(f"  ✓ Success! Result type: {type(result)}")
except Exception as e:
    print(f"  ✗ Failed with error: {type(e).__name__}: {e}")

# Clean up
os.unlink(temp_path)
print("\nTest complete.")