#!/bin/bash
# Monitor Arc Vision training progress - works without jq

OUTPUT_DIR="${OUTPUT_DIR:-/root/verl/outputs/arc_vision}"
LOGS_DIR="$OUTPUT_DIR/detailed_logs"

echo "=========================================="
echo "Arc Vision Training Monitor (5 epochs)"
echo "=========================================="

# Function to extract JSON field using grep and awk
extract_field() {
    local file=$1
    local field=$2
    grep -o "\"$field\":[^,}]*" "$file" | cut -d':' -f2 | tr -d ' "'
}

# Function to calculate average from jsonl using pure bash
calc_avg() {
    local file=$1
    local field=$2
    if [ -f "$file" ]; then
        # Extract field values and calculate average
        tail -100 "$file" 2>/dev/null | grep -o "\"$field\": *[0-9.]*" | awk -F': ' '{sum+=$2; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}'
    else
        echo "N/A"
    fi
}

# Function to count true/false occurrences
count_bool() {
    local file=$1
    local field=$2
    local value=$3
    if [ -f "$file" ]; then
        tail -100 "$file" 2>/dev/null | grep -c "\"$field\": *$value"
    else
        echo "0"
    fi
}

# Function to get latest values from console output
get_latest_from_console() {
    # Look for the training output patterns
    if [ -f "$OUTPUT_DIR/training.log" ]; then
        tail -50 "$OUTPUT_DIR/training.log" | grep -E "\[score\]|\[r_task\]|\[r_tool\]|\[r_gate\]|\[iou\]" | tail -5
    else
        # If no log file, suggest checking console output
        echo "Monitor console output for real-time metrics"
    fi
}

while true; do
    clear
    echo "Arc Vision Training Progress - $(date)"
    echo "==========================================
Training Configuration: 5 epochs total
Checkpoints saved every 25 steps
=========================================="
    
    # Check if logs exist
    if [ ! -d "$LOGS_DIR" ]; then
        echo "Waiting for training to start..."
        echo "Logs directory not found: $LOGS_DIR"
        echo ""
        echo "Make sure training is running with:"
        echo "N_GPUS=2 bash examples/arc_vision/run_arc_vision_grpo.sh"
        sleep 5
        continue
    fi
    
    # Display metrics from last 100 samples
    echo -e "\n📊 Performance Metrics (last 100 samples):"
    echo "----------------------------------------"
    
    if [ -f "$LOGS_DIR/reasoning_traces.jsonl" ]; then
        echo "- Average IoU: $(calc_avg $LOGS_DIR/reasoning_traces.jsonl actual_iou)"
        echo "- Samples with tools: $(count_bool $LOGS_DIR/reasoning_traces.jsonl has_tool_calls true)/100"
        echo "- Tool invocations avg: $(calc_avg $LOGS_DIR/reasoning_traces.jsonl tool_invocations)"
        
        # Count samples with non-zero IoU
        NON_ZERO=$(tail -100 "$LOGS_DIR/reasoning_traces.jsonl" 2>/dev/null | grep -c '"actual_iou": *[1-9]')
        echo "- Non-zero IoU samples: $NON_ZERO/100"
    else
        echo "- No reasoning traces yet..."
    fi
    
    echo -e "\n🎯 Confidence Calibration:"
    echo "----------------------------------------"
    if [ -f "$LOGS_DIR/confidence_calibration.jsonl" ]; then
        echo "- Avg predicted confidence: $(calc_avg $LOGS_DIR/confidence_calibration.jsonl predicted_confidence)"
        echo "- Avg actual IoU: $(calc_avg $LOGS_DIR/confidence_calibration.jsonl actual_iou)"
        echo "- Overconfident cases: $(count_bool $LOGS_DIR/confidence_calibration.jsonl overconfident true)"
        echo "- Underconfident cases: $(count_bool $LOGS_DIR/confidence_calibration.jsonl underconfident true)"
    else
        echo "- No calibration data yet..."
    fi
    
    echo -e "\n🔧 Tool Usage Patterns:"
    echo "----------------------------------------"
    if [ -f "$LOGS_DIR/tool_patterns.jsonl" ]; then
        echo "- Zoom tool used: $(count_bool $LOGS_DIR/tool_patterns.jsonl used_zoom true) times"
        echo "- Wait tool used: $(count_bool $LOGS_DIR/tool_patterns.jsonl used_wait true) times"
        echo "- Inspect tool used: $(count_bool $LOGS_DIR/tool_patterns.jsonl used_inspect true) times"
        echo "- Avg confidence gain: $(calc_avg $LOGS_DIR/tool_patterns.jsonl confidence_gain)"
    else
        echo "- No tool usage data yet..."
    fi
    
    echo -e "\n⚠️  Contradictions (last 10):"
    echo "----------------------------------------"
    if [ -f "$LOGS_DIR/contradictions.jsonl" ]; then
        TOTAL_CONTRADICTIONS=$(wc -l < "$LOGS_DIR/contradictions.jsonl")
        echo "- Total contradictions logged: $TOTAL_CONTRADICTIONS"
        
        # Show types of recent contradictions
        RECENT=$(tail -10 "$LOGS_DIR/contradictions.jsonl" 2>/dev/null)
        if [ -n "$RECENT" ]; then
            echo "- Unnecessary tool: $(echo "$RECENT" | grep -c '"unnecessary_tool_use": true')"
            echo "- Missed opportunity: $(echo "$RECENT" | grep -c '"missed_tool_opportunity": true')"
            echo "- Ineffective tool: $(echo "$RECENT" | grep -c '"ineffective_tool_use": true')"
        fi
    else
        echo "- No contradictions logged yet..."
    fi
    
    echo -e "\n📈 Latest Console Metrics:"
    echo "----------------------------------------"
    echo "Look for these patterns in your console output:"
    echo "[score] - Total reward"
    echo "[r_task] - Task reward (IoU)"
    echo "[r_tool] - Tool effectiveness reward"
    echo "[r_gate] - Tool penalty"
    echo "[iou] - Intersection over Union"
    
    # Show file sizes to track progress
    echo -e "\n💾 Log File Sizes:"
    echo "----------------------------------------"
    if [ -d "$LOGS_DIR" ]; then
        for file in reasoning_traces confidence_calibration tool_patterns contradictions; do
            if [ -f "$LOGS_DIR/${file}.jsonl" ]; then
                SIZE=$(du -h "$LOGS_DIR/${file}.jsonl" 2>/dev/null | cut -f1)
                LINES=$(wc -l < "$LOGS_DIR/${file}.jsonl" 2>/dev/null)
                echo "- ${file}: $SIZE ($LINES entries)"
            fi
        done
    fi
    
    echo -e "\n=========================================="
    echo "Training for 5 epochs | Saves every 25 steps"
    echo "Press Ctrl+C to exit | Refreshing in 10s..."
    
    sleep 10
done