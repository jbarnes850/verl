#!/bin/bash
# Launch script for vLLM advisor server

MODEL_PATH="qwen3_4b_sec_advisor_step80"
PORT="${1:-8000}"

echo "Starting vLLM server for advisor model..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"

# Run without custom chat template to use model's default
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --disable-log-stats
