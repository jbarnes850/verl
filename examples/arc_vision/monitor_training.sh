#!/bin/bash
# Monitor Arc Vision training progress

OUTPUT_DIR="${OUTPUT_DIR:-/root/verl/outputs/arc_vision}"
LOGS_DIR="$OUTPUT_DIR/detailed_logs"

echo "=========================================="
echo "Arc Vision Training Monitor"
echo "=========================================="

# Function to calculate average from jsonl
calc_avg() {
    local file=$1
    local field=$2
    if [ -f "$file" ]; then
        tail -100 "$file" | jq -r ".$field" | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "N/A"}'
    else
        echo "N/A"
    fi
}

# Function to count occurrences
count_field() {
    local file=$1
    local field=$2
    local value=$3
    if [ -f "$file" ]; then
        tail -100 "$file" | jq -r ".$field" | grep -c "$value"
    else
        echo "0"
    fi
}

while true; do
    clear
    echo "Arc Vision Training Progress - $(date)"
    echo "=========================================="
    
    # Check if logs exist
    if [ ! -d "$LOGS_DIR" ]; then
        echo "Waiting for training to start..."
        echo "Logs directory not found: $LOGS_DIR"
        sleep 5
        continue
    fi
    
    # Display metrics from last 100 samples
    echo -e "\nPerformance Metrics (last 100 samples):"
    echo "- Average IoU: $(calc_avg $LOGS_DIR/reasoning_traces.jsonl actual_iou)"
    echo "- Average Confidence Before: $(calc_avg $LOGS_DIR/confidence_calibration.jsonl predicted_confidence)"
    echo "- Average Confidence After: $(calc_avg $LOGS_DIR/confidence_calibration.jsonl actual_iou)"
    
    echo -e "\nTool Usage (last 100 samples):"
    echo "- Samples using tools: $(count_field $LOGS_DIR/reasoning_traces.jsonl has_tool_calls true)"
    echo "- Average tool invocations: $(calc_avg $LOGS_DIR/reasoning_traces.jsonl tool_invocations)"
    
    echo -e "\nReward Components (if available):"
    if [ -f "$LOGS_DIR/reasoning_traces.jsonl" ]; then
        # Get latest reward info from training logs
        echo "Check main training logs for reward breakdown"
    fi
    
    echo -e "\nLatest Contradictions Detected:"
    if [ -f "$LOGS_DIR/contradictions.jsonl" ]; then
        tail -3 "$LOGS_DIR/contradictions.jsonl" | jq -r '.contradiction_count' | while read count; do
            echo "- Contradictions: $count"
        done
    fi
    
    echo -e "\n=========================================="
    echo "Press Ctrl+C to exit"
    echo "Refreshing in 10 seconds..."
    
    sleep 10
done