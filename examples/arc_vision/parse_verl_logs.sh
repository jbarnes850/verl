#!/bin/bash
# Parse VERL console logs to extract key metrics and training progress

OUTPUT_DIR="${OUTPUT_DIR:-/root/verl/outputs/arc_vision}"
LOG_FILE="${1:-}"

echo "==========================================="
echo "VERL Log Parser for Arc Vision Training"
echo "==========================================="

# Function to extract metrics from console logs
extract_metrics() {
    local log_file=$1
    
    echo -e "\n📊 Extracting Training Metrics..."
    echo "----------------------------------------"
    
    # Count total samples
    TOTAL_SAMPLES=$(grep -c "len reward_extra_infos_dict\['reward'\]:" "$log_file" 2>/dev/null || echo 0)
    echo "Total batches processed: $TOTAL_SAMPLES"
    
    # Extract batch sizes and calculate total samples
    SAMPLE_COUNT=$(grep "len reward_extra_infos_dict\['reward'\]:" "$log_file" 2>/dev/null | \
                   awk -F': ' '{sum+=$2} END {print sum}' || echo 0)
    echo "Total samples processed: $SAMPLE_COUNT"
    
    # Calculate average IoU
    AVG_IOU=$(grep "\[iou\]" "$log_file" 2>/dev/null | \
              awk '{print $2}' | \
              awk '{sum+=$1; count++} END {if(count>0) printf "%.4f", sum/count; else print "0"}')
    echo "Average IoU: $AVG_IOU"
    
    # Calculate average rewards
    AVG_SCORE=$(grep "\[score\]" "$log_file" 2>/dev/null | \
                awk '{print $2}' | \
                awk '{sum+=$1; count++} END {if(count>0) printf "%.4f", sum/count; else print "0"}')
    echo "Average total reward: $AVG_SCORE"
    
    # Tool usage statistics
    TOOL_USAGE=$(grep "\[tool_invocations\]" "$log_file" 2>/dev/null | \
                 awk '{print $2}' | \
                 awk '{if($1>0) tools++; total++} END {if(total>0) printf "%d/%d (%.1f%%)", tools, total, tools/total*100; else print "0/0"}')
    echo "Samples with tool usage: $TOOL_USAGE"
    
    # Confidence analysis
    echo -e "\n🎯 Confidence Analysis:"
    AVG_CONF_BEFORE=$(grep "\[confidence_before\]" "$log_file" 2>/dev/null | \
                      awk '{print $2}' | \
                      awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "0"}')
    AVG_CONF_AFTER=$(grep "\[confidence_after\]" "$log_file" 2>/dev/null | \
                     awk '{print $2}' | \
                     awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "0"}')
    echo "- Average confidence before: $AVG_CONF_BEFORE"
    echo "- Average confidence after: $AVG_CONF_AFTER"
    
    # Show IoU progression over time
    echo -e "\n📈 IoU Progression (every 10 batches):"
    grep "\[iou\]" "$log_file" 2>/dev/null | \
        awk 'NR%10==0 {print "Batch", NR": IoU =", $2}' | tail -10
}

# Function to estimate epoch progress
estimate_epoch() {
    local log_file=$1
    
    # Count samples
    SAMPLE_COUNT=$(grep "len reward_extra_infos_dict\['reward'\]:" "$log_file" 2>/dev/null | \
                   awk -F': ' '{sum+=$2} END {print sum}' || echo 0)
    
    # Estimate based on ~50k samples per epoch
    if [ "$SAMPLE_COUNT" -gt 0 ]; then
        EPOCH=$(echo "scale=2; $SAMPLE_COUNT / 50000" | bc)
        CURRENT_EPOCH=$(echo "scale=0; $SAMPLE_COUNT / 50000 + 1" | bc)
        EPOCH_PROGRESS=$(echo "scale=1; ($SAMPLE_COUNT % 50000) / 50000 * 100" | bc)
        
        echo -e "\n📖 Epoch Progress:"
        echo "- Current Epoch: $CURRENT_EPOCH / 5"
        echo "- Progress in current epoch: ${EPOCH_PROGRESS}%"
        echo "- Total progress: $(echo "scale=1; $EPOCH * 20" | bc)% of training"
    fi
}

# Function to show recent samples
show_recent() {
    local log_file=$1
    
    echo -e "\n🔍 Recent Training Examples (last 3):"
    echo "----------------------------------------"
    
    # Extract last 3 prompts and responses
    tail -500 "$log_file" | grep -A1 -E "\[prompt\]|\[response\]|\[ground_truth\]|\[iou\]" | tail -20
}

# Main execution
if [ -z "$LOG_FILE" ]; then
    echo "Usage: $0 <log_file>"
    echo ""
    echo "Examples:"
    echo "  # Parse saved log file"
    echo "  $0 /path/to/training.log"
    echo ""
    echo "  # Parse live output"
    echo "  python train.py 2>&1 | tee training.log"
    echo "  $0 training.log"
    echo ""
    echo "  # Or use the Python monitor for real-time display:"
    echo "  python train.py 2>&1 | python monitor_verl_console.py"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    exit 1
fi

# Run analysis
extract_metrics "$LOG_FILE"
estimate_epoch "$LOG_FILE"
show_recent "$LOG_FILE"

echo -e "\n==========================================="
echo "Analysis complete!"
echo "For real-time monitoring, use:"
echo "  python train.py 2>&1 | python monitor_verl_console.py"