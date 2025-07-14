#!/usr/bin/env python3
"""
Script to prepare VERL checkpoint for use as an advisor model in CRMArena.
Handles FSDP checkpoint consolidation and vLLM server setup.
"""

import os
import sys
import json
import torch
import argparse
import subprocess
from pathlib import Path

def consolidate_fsdp_checkpoint(checkpoint_dir, output_dir, base_model="Qwen/Qwen3-4B"):
    """Consolidate FSDP sharded checkpoints using VERL's built-in model merger."""
    print(f"Consolidating FSDP checkpoint from {checkpoint_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use VERL's model merger to consolidate the checkpoint
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(checkpoint_dir),
        "--target_dir", str(output_dir)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error merging model: {result.stderr}")
        raise RuntimeError("Failed to merge FSDP checkpoint")
    
    print(result.stdout)
    
    # Save model info for reference
    info = {
        "base_model": base_model,
        "checkpoint_dir": str(checkpoint_dir),
        "consolidation_method": "verl_model_merger",
        "merge_command": " ".join(cmd)
    }
    with open(os.path.join(output_dir, "consolidation_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print("Model consolidation complete!")
    return output_dir

def test_model_loading(model_path):
    """Quick test to ensure model loads correctly."""
    print(f"\nTesting model loading from {model_path}")
    try:
        # Import transformers here to avoid issues if not installed
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("Transformers not installed, skipping model test")
            return True
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Test generation
        inputs = tokenizer("Query: Find all leads with status", return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test generation successful: {response[:100]}...")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

def create_vllm_launch_script(model_path, script_path="launch_advisor_server.sh"):
    """Create a launch script for vLLM server."""
    script_content = f"""#!/bin/bash
# Launch script for vLLM advisor server

MODEL_PATH="{model_path}"
PORT="${{1:-8000}}"

echo "Starting vLLM server for advisor model..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"

python -m vllm.entrypoints.openai.api_server \\
    --model "$MODEL_PATH" \\
    --port "$PORT" \\
    --tensor-parallel-size 2 \\
    --max-model-len 4096 \\
    --gpu-memory-utilization 0.8 \\
    --dtype bfloat16 \\
    --enable-prefix-caching \\
    --max-num-seqs 256 \\
    --disable-log-stats
"""
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"\nCreated vLLM launch script: {script_path}")
    print(f"Usage: ./{script_path} [port]")

def create_test_script(model_name, script_path="test_advisor.py"):
    """Create a test script for the advisor functionality."""
    script_content = f'''#!/usr/bin/env python3
"""Test script for advisor model."""

import requests
import json

def test_advisor(endpoint="http://localhost:8000/v1", model="{model_name}"):
    """Test the advisor model with a sample CRM query."""
    
    # Test completions endpoint
    print("Testing advisor model...")
    
    messages = [
        {{"role": "system", "content": "You are a CRM expert advisor. Suggest the best SOQL query approach."}},
        {{"role": "user", "content": "I need to find all opportunities that closed last quarter with amount > $50,000"}}
    ]
    
    response = requests.post(
        f"{{endpoint}}/chat/completions",
        json={{
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 200
        }}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\\nAdvisor suggestion:")
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"Error: {{response.status_code}}")
        print(response.text)

if __name__ == "__main__":
    test_advisor()
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"Created test script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare VERL model for advisor use")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoint_qwen3_4b_200steps/global_step_20/actor",
        help="Path to FSDP checkpoint directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qwen3_4b_sec_advisor",
        help="Output directory for consolidated model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Base model name from HuggingFace"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip model loading test"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)
    
    # Consolidate checkpoint
    output_dir = consolidate_fsdp_checkpoint(
        args.checkpoint_dir,
        args.output_dir,
        args.base_model
    )
    
    # Test model loading
    if not args.skip_test:
        success = test_model_loading(output_dir)
        if not success:
            print("Warning: Model test failed, but files were saved.")
    
    # Create helper scripts
    create_vllm_launch_script(output_dir)
    create_test_script(Path(output_dir).name)
    
    print("\n" + "="*50)
    print("Model preparation complete!")
    print("="*50)
    print(f"\nNext steps:")
    print(f"1. Start vLLM server: ./launch_advisor_server.sh")
    print(f"2. Test advisor: python test_advisor.py")
    print(f"3. Use in CRMArena evaluation:")
    print(f"   python run_tasks.py \\")
    print(f"     --model gpt-4o \\")
    print(f"     --advisor_model {Path(output_dir).name} \\")
    print(f"     --advisor_endpoint http://localhost:8000/v1 \\")
    print(f"     --task_category all")

if __name__ == "__main__":
    main()