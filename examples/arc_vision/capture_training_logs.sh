#!/bin/bash
# Capture VERL training logs to a file for analysis
# This script helps save console output for later parsing

OUTPUT_DIR="${OUTPUT_DIR:-/root/verl/outputs/arc_vision}"
LOG_FILE="$OUTPUT_DIR/console_training_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================="
echo "VERL Training Log Capture"
echo "==========================================="
echo ""
echo "This will capture your training output to:"
echo "  $LOG_FILE"
echo ""
echo "You can then:"
echo "1. Monitor in real-time with: tail -f $LOG_FILE | python monitor_verl_console.py"
echo "2. Parse afterwards with: ./parse_verl_logs.sh $LOG_FILE"
echo ""
echo "Starting training with log capture..."
echo "==========================================="

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run training and capture all output
N_GPUS=${N_GPUS:-2} bash examples/arc_vision/run_arc_vision_grpo.sh 2>&1 | tee "$LOG_FILE"

echo ""
echo "==========================================="
echo "Training complete! Logs saved to:"
echo "  $LOG_FILE"
echo ""
echo "To analyze the logs, run:"
echo "  ./parse_verl_logs.sh $LOG_FILE"