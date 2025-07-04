#!/usr/bin/env python3
"""
VERL Console Monitor - Parse and display Arc Vision training logs from console output
Designed to work with VERL's Ray TaskRunner output format
"""

import sys
import re
import time
from datetime import datetime
from collections import defaultdict

class VERLConsoleMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_batch_size = 0
        self.samples_processed = 0
        self.start_time = time.time()
        self.last_update = time.time()
        
        # Colors for terminal
        self.colors = {
            'RESET': '\033[0m',
            'BOLD': '\033[1m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'RED': '\033[91m',
            'CYAN': '\033[96m',
            'MAGENTA': '\033[95m'
        }
        
        # Patterns to extract from logs
        self.patterns = {
            'score': r'\[score\]\s*([\d.-]+)',
            'r_task': r'\[r_task\]\s*([\d.-]+)',
            'r_tool': r'\[r_tool\]\s*([\d.-]+)',
            'r_gate': r'\[r_gate\]\s*([\d.-]+)',
            'iou': r'\[iou\]\s*([\d.-]+)',
            'confidence_before': r'\[confidence_before\]\s*([\d.-]+)',
            'confidence_after': r'\[confidence_after\]\s*([\d.-]+)',
            'tool_invocations': r'\[tool_invocations\]\s*(\d+)',
            'predicted_bbox': r'\[predicted_bbox\]\s*\[([\d., -]+)\]',
            'ground_truth': r'\[ground_truth\]\s*\[([\d., -]+)\]',
            'batch_size': r'len reward_extra_infos_dict\[\'reward\'\]:\s*(\d+)',
            'prompt': r'\[prompt\]\s*(.*)',
            'response': r'\[response\]\s*(.*)',
        }
        
    def extract_metrics(self, line):
        """Extract metrics from a log line"""
        for metric, pattern in self.patterns.items():
            match = re.search(pattern, line)
            if match:
                if metric == 'batch_size':
                    self.current_batch_size = int(match.group(1))
                    self.samples_processed += self.current_batch_size
                elif metric in ['score', 'r_task', 'r_tool', 'r_gate', 'iou', 
                               'confidence_before', 'confidence_after']:
                    value = float(match.group(1))
                    self.metrics[metric].append(value)
                elif metric == 'tool_invocations':
                    value = int(match.group(1))
                    self.metrics[metric].append(value)
                    
    def calculate_averages(self, window=100):
        """Calculate averages for the last N samples"""
        averages = {}
        for metric, values in self.metrics.items():
            if values:
                recent = values[-window:] if len(values) > window else values
                averages[metric] = sum(recent) / len(recent)
            else:
                averages[metric] = 0.0
        return averages
    
    def format_time(self, seconds):
        """Format seconds into readable time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def print_status(self):
        """Print current training status"""
        # Clear screen for clean output
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        
        elapsed = time.time() - self.start_time
        averages = self.calculate_averages()
        
        print(f"{self.colors['CYAN']}{'='*60}{self.colors['RESET']}")
        print(f"{self.colors['BOLD']}VERL Arc Vision Training Monitor{self.colors['RESET']}")
        print(f"{self.colors['CYAN']}{'='*60}{self.colors['RESET']}")
        
        print(f"\n{self.colors['YELLOW']}⏱️  Training Duration:{self.colors['RESET']} {self.format_time(elapsed)}")
        print(f"{self.colors['YELLOW']}📊 Samples Processed:{self.colors['RESET']} {self.samples_processed}")
        print(f"{self.colors['YELLOW']}📦 Current Batch Size:{self.colors['RESET']} {self.current_batch_size}")
        
        # Estimate epoch progress (assuming ~50k samples per epoch)
        epoch_progress = (self.samples_processed / 50000) % 1
        current_epoch = int(self.samples_processed / 50000) + 1
        print(f"{self.colors['YELLOW']}📖 Estimated Epoch:{self.colors['RESET']} {current_epoch}/5 ({epoch_progress*100:.1f}%)")
        
        print(f"\n{self.colors['GREEN']}📈 Performance Metrics (last 100 samples):{self.colors['RESET']}")
        print("-" * 40)
        
        # Main metrics
        iou = averages.get('iou', 0)
        color = self.colors['GREEN'] if iou > 0.05 else self.colors['YELLOW']
        print(f"• Average IoU: {color}{iou:.4f}{self.colors['RESET']}")
        
        print(f"• Total Reward: {averages.get('score', 0):.4f}")
        print(f"  - Task Reward: {averages.get('r_task', 0):.4f}")
        print(f"  - Tool Reward: {averages.get('r_tool', 0):.4f}")
        print(f"  - Gate Penalty: {averages.get('r_gate', 0):.4f}")
        
        # Confidence analysis
        conf_before = averages.get('confidence_before', 0)
        conf_after = averages.get('confidence_after', 0)
        print(f"\n{self.colors['BLUE']}🎯 Confidence Analysis:{self.colors['RESET']}")
        print(f"• Before Tools: {conf_before:.3f}")
        print(f"• After Tools: {conf_after:.3f}")
        print(f"• Confidence Gain: {conf_after - conf_before:.3f}")
        
        # Tool usage
        tool_count = sum(1 for x in self.metrics.get('tool_invocations', []) if x > 0)
        print(f"\n{self.colors['MAGENTA']}🔧 Tool Usage:{self.colors['RESET']}")
        print(f"• Samples with tools: {tool_count}/{len(self.metrics.get('tool_invocations', []))}")
        print(f"• Avg invocations: {averages.get('tool_invocations', 0):.2f}")
        
        # Recent trend
        if len(self.metrics.get('iou', [])) > 20:
            recent_10 = sum(self.metrics['iou'][-10:]) / 10
            recent_20 = sum(self.metrics['iou'][-20:-10]) / 10
            trend = "📈" if recent_10 > recent_20 else "📉"
            print(f"\n{self.colors['CYAN']}📊 Recent Trend:{self.colors['RESET']} {trend}")
            print(f"• Last 10 samples: {recent_10:.4f}")
            print(f"• Previous 10: {recent_20:.4f}")
        
        print(f"\n{self.colors['CYAN']}{'='*60}{self.colors['RESET']}")
        print("Press Ctrl+C to exit | Updates every new batch")
        
    def run(self):
        """Main monitoring loop"""
        print("Monitoring VERL console output...")
        print("Pipe your training output to this script:")
        print("  python train.py 2>&1 | python monitor_verl_console.py")
        print("")
        
        buffer = []
        try:
            for line in sys.stdin:
                # Strip color codes and extra whitespace
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line.strip())
                
                # Look for TaskRunner lines
                if "(TaskRunner" in clean_line:
                    # Extract the actual log content
                    match = re.search(r'\(TaskRunner.*?\)\s*(.*)', clean_line)
                    if match:
                        content = match.group(1)
                        buffer.append(content)
                        self.extract_metrics(content)
                        
                        # Update display when we see batch size (end of batch)
                        if 'len reward_extra_infos_dict' in content:
                            if time.time() - self.last_update > 1:  # Update at most once per second
                                self.print_status()
                                self.last_update = time.time()
                            buffer = []  # Clear buffer
                
                # Also handle direct metric lines
                elif any(f"[{metric}]" in clean_line for metric in self.patterns.keys()):
                    self.extract_metrics(clean_line)
                    
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            self.print_final_summary()
            
    def print_final_summary(self):
        """Print final training summary"""
        elapsed = time.time() - self.start_time
        averages = self.calculate_averages(window=len(self.metrics.get('iou', [])))
        
        print(f"\n{self.colors['BOLD']}Final Training Summary{self.colors['RESET']}")
        print("=" * 40)
        print(f"Total Duration: {self.format_time(elapsed)}")
        print(f"Total Samples: {self.samples_processed}")
        print(f"Final Average IoU: {averages.get('iou', 0):.4f}")
        print(f"Final Average Reward: {averages.get('score', 0):.4f}")
        
        # Show improvement
        if len(self.metrics.get('iou', [])) > 100:
            first_100_iou = sum(self.metrics['iou'][:100]) / 100
            last_100_iou = sum(self.metrics['iou'][-100:]) / 100
            improvement = (last_100_iou - first_100_iou) / first_100_iou * 100
            print(f"IoU Improvement: {improvement:+.1f}%")

if __name__ == '__main__':
    monitor = VERLConsoleMonitor()
    monitor.run()