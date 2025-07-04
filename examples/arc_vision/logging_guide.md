# Arc Vision Training Logging Guide

This guide explains how to monitor and analyze Arc Vision training logs with full timestamps and epoch tracking.

## Quick Start

### Option 1: Run Training with Enhanced Logging
```bash
./run_arc_vision_grpo_with_logging.sh
```

This script:
- Creates timestamped log directories
- Captures all output with timestamps
- Separates metrics into a dedicated log file
- Creates a summary file with training configuration
- Maintains a symlink to the latest logs at `outputs/arc_vision/latest_logs`

### Option 2: Monitor Existing Training

#### Real-time monitoring of running training:
```bash
# If training is running in another terminal
./run_arc_vision_grpo.sh 2>&1 | python monitor_training.py --live
```

#### Monitor existing log files:
```bash
# Monitor the latest log file
python monitor_training.py

# Monitor a specific log file
python monitor_training.py --log-file /path/to/training.log

# Show all content without following
python monitor_training.py --no-follow --show-all
```

## Log File Locations

When using `run_arc_vision_grpo_with_logging.sh`, logs are organized as:
```
outputs/arc_vision/
├── latest_logs -> logs_20240104_123456/  # Symlink to latest run
├── logs_20240104_123456/
│   ├── training.log      # Complete training output with timestamps
│   ├── metrics.log       # Extracted metrics only
│   ├── errors.log        # Error messages only
│   ├── summary.txt       # Training configuration summary
│   └── hydra/           # Hydra configuration logs
```

## Understanding the Logs

### Epoch Information
The training runs for the number of epochs specified in `trainer.total_epochs` (default: 5).
Look for lines containing:
- `training/epoch: X` - Current epoch number
- `training/global_step: X` - Current global training step

### Key Metrics to Monitor
- **Loss Values**: Lower is generally better
  - `actor/loss` - Actor model loss
  - `critic/loss` - Critic model loss (if enabled)
  
- **Reward Metrics**: Higher is better
  - `reward/mean` - Average reward across batch
  - `reward/std` - Standard deviation of rewards
  
- **KL Divergence**: Should be controlled
  - `actor/kl` - KL divergence from reference policy
  
- **Entropy**: Indicates exploration
  - `actor/entropy` - Policy entropy (higher = more exploration)

### Color Coding in monitor_training.py
- **Cyan**: Timestamps
- **Green**: Epoch information and rewards
- **Blue**: Step numbers and entropy
- **Yellow**: Warnings and losses
- **Red**: Errors and exceptions
- **Magenta**: Checkpointing events

## Analyzing Training Progress

### View Metrics Only
```bash
# From latest run
tail -f outputs/arc_vision/latest_logs/metrics.log

# With timestamps
cat outputs/arc_vision/latest_logs/metrics.log | grep "epoch"
```

### Check Training Summary
```bash
cat outputs/arc_vision/latest_logs/summary.txt
```

### Find Specific Events
```bash
# Find all epoch transitions
grep -n "training/epoch" outputs/arc_vision/latest_logs/training.log

# Find checkpoints
grep -n "checkpoint" outputs/arc_vision/latest_logs/training.log

# Find validation results
grep -n "test_freq\|validation" outputs/arc_vision/latest_logs/training.log
```

## Troubleshooting

### No Logs Found
1. Check if training has started: `ps aux | grep main_ppo`
2. Verify output directory exists: `ls -la outputs/arc_vision/`
3. Check permissions: `ls -la outputs/`

### Missing Epoch Information
Epochs are logged at the end of each training step. If you don't see epoch info:
1. Training might still be in the first epoch
2. Check `trainer.logger` is set to include 'console'
3. Verify `trainer.total_epochs` is set correctly

### Real-time Monitoring Not Working
1. Ensure Python unbuffered mode: `export PYTHONUNBUFFERED=1`
2. Use `tee` to duplicate output: `./run_arc_vision_grpo.sh 2>&1 | tee training.log`
3. Try the `--live` option with monitor_training.py

## Advanced Usage

### Custom Log Filtering
```bash
# Monitor only specific metrics
python monitor_training.py | grep -E "epoch|reward|loss"

# Save filtered logs
python monitor_training.py --no-follow | grep "epoch" > epoch_transitions.log
```

### Automated Alerting
```bash
# Alert when training completes
while ! grep -q "Training completed" outputs/arc_vision/latest_logs/training.log 2>/dev/null; do
    sleep 60
done
echo "Training finished!" | mail -s "Arc Vision Training Complete" your@email.com
```

## Tips

1. **Storage Management**: Logs can grow large. Consider rotating old logs:
   ```bash
   find outputs/arc_vision/logs_* -mtime +7 -type d -exec rm -rf {} +
   ```

2. **Performance**: For very long training runs, consider using:
   ```bash
   # Compress old logs
   gzip outputs/arc_vision/logs_*/training.log
   ```

3. **Remote Monitoring**: Use SSH with tmux/screen for persistent monitoring:
   ```bash
   tmux new -s arc_vision_monitor
   python monitor_training.py
   # Detach: Ctrl+B, D
   # Reattach: tmux attach -t arc_vision_monitor
   ```