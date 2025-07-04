#!/usr/bin/env python3
"""
Arc Vision Training Monitor - Real-time training log viewer with comprehensive information
Displays timestamps, epochs, steps, and all training metrics
"""

import argparse
import subprocess
import re
import time
from datetime import datetime
import sys
import os


class TrainingMonitor:
    def __init__(self, log_file=None, follow=True, show_all=False, highlight_epochs=True):
        self.log_file = log_file
        self.follow = follow
        self.show_all = show_all
        self.highlight_epochs = highlight_epochs
        self.current_epoch = None
        self.current_step = None
        self.start_time = None
        
        # Color codes for terminal output
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
    
    def format_timestamp(self):
        """Add timestamp to log lines"""
        return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
    
    def parse_metrics(self, line):
        """Extract and format metrics from log lines"""
        # Look for patterns like "step:X - metric:value"
        step_match = re.search(r'step:(\d+)', line)
        if step_match:
            self.current_step = int(step_match.group(1))
        
        # Look for epoch information
        epoch_match = re.search(r'epoch[:\s]+(\d+)', line, re.IGNORECASE)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
        
        # Extract metrics
        metrics = {}
        # Pattern for "key:value" or "key: value"
        metric_pattern = r'(\w+[/\w]*)\s*:\s*([\d.]+(?:e[+-]?\d+)?)'
        for match in re.finditer(metric_pattern, line):
            key, value = match.groups()
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
        
        return metrics
    
    def format_line(self, line):
        """Format a log line with colors and highlighting"""
        timestamp = self.format_timestamp()
        metrics = self.parse_metrics(line)
        
        # Skip empty lines unless showing all
        if not line.strip() and not self.show_all:
            return None
        
        # Highlight important lines
        formatted_line = f"{self.colors['CYAN']}{timestamp}{self.colors['RESET']} "
        
        # Add epoch/step info if available
        if self.current_epoch is not None or self.current_step is not None:
            info_parts = []
            if self.current_epoch is not None:
                info_parts.append(f"{self.colors['GREEN']}Epoch:{self.current_epoch}{self.colors['RESET']}")
            if self.current_step is not None:
                info_parts.append(f"{self.colors['BLUE']}Step:{self.current_step}{self.colors['RESET']}")
            if info_parts:
                formatted_line += f"[{' | '.join(info_parts)}] "
        
        # Highlight specific patterns
        if re.search(r'epoch', line, re.IGNORECASE) and self.highlight_epochs:
            line = f"{self.colors['BOLD']}{self.colors['GREEN']}{line}{self.colors['RESET']}"
        elif 'error' in line.lower() or 'exception' in line.lower():
            line = f"{self.colors['RED']}{line}{self.colors['RESET']}"
        elif 'warning' in line.lower() or 'warn' in line.lower():
            line = f"{self.colors['YELLOW']}{line}{self.colors['RESET']}"
        elif 'saving' in line.lower() or 'checkpoint' in line.lower():
            line = f"{self.colors['MAGENTA']}{line}{self.colors['RESET']}"
        elif any(key in line.lower() for key in ['loss', 'reward', 'entropy', 'kl']):
            # Highlight training metrics
            for metric, value in metrics.items():
                if isinstance(value, float):
                    # Color code based on metric type
                    if 'loss' in metric.lower():
                        color = self.colors['YELLOW']
                    elif 'reward' in metric.lower():
                        color = self.colors['GREEN']
                    elif 'entropy' in metric.lower():
                        color = self.colors['BLUE']
                    else:
                        color = self.colors['CYAN']
                    
                    # Replace the metric in the line with colored version
                    pattern = f"{metric}\\s*:\\s*{value}"
                    replacement = f"{color}{metric}: {value:.4f}{self.colors['RESET']}"
                    line = re.sub(pattern, replacement, line)
        
        return formatted_line + line
    
    def monitor_file(self, file_path):
        """Monitor a specific log file"""
        print(f"{self.colors['BOLD']}Monitoring log file: {file_path}{self.colors['RESET']}")
        print(f"{self.colors['CYAN']}Press Ctrl+C to stop{self.colors['RESET']}\n")
        
        try:
            if self.follow:
                # Use tail -f for real-time following
                process = subprocess.Popen(
                    ['tail', '-f', '-n', '100', file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                for line in process.stdout:
                    formatted_line = self.format_line(line.rstrip())
                    if formatted_line:
                        print(formatted_line)
            else:
                # Read entire file
                with open(file_path, 'r') as f:
                    for line in f:
                        formatted_line = self.format_line(line.rstrip())
                        if formatted_line:
                            print(formatted_line)
        except KeyboardInterrupt:
            print(f"\n{self.colors['YELLOW']}Monitoring stopped by user{self.colors['RESET']}")
        except Exception as e:
            print(f"{self.colors['RED']}Error: {e}{self.colors['RESET']}")
    
    def find_latest_log(self, output_dir):
        """Find the most recent log file in the output directory"""
        log_patterns = [
            os.path.join(output_dir, '**', '*.log'),
            os.path.join(output_dir, '**', 'log.txt'),
            os.path.join(output_dir, '**', 'train.log'),
            os.path.join(output_dir, '**', 'output.log')
        ]
        
        import glob
        all_logs = []
        for pattern in log_patterns:
            all_logs.extend(glob.glob(pattern, recursive=True))
        
        if not all_logs:
            # Try to find any text file that might contain logs
            all_logs = glob.glob(os.path.join(output_dir, '**', '*.txt'), recursive=True)
        
        if all_logs:
            # Return the most recently modified log file
            return max(all_logs, key=os.path.getmtime)
        return None
    
    def monitor_live_training(self):
        """Monitor training output in real-time from stdout/stderr"""
        print(f"{self.colors['BOLD']}Monitoring live training output{self.colors['RESET']}")
        print(f"{self.colors['CYAN']}Waiting for training logs...{self.colors['RESET']}\n")
        
        self.start_time = datetime.now()
        
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                formatted_line = self.format_line(line.rstrip())
                if formatted_line:
                    print(formatted_line)
                    
                # Show elapsed time every 100 steps
                if self.current_step and self.current_step % 100 == 0:
                    elapsed = datetime.now() - self.start_time
                    print(f"{self.colors['CYAN']}[Elapsed time: {elapsed}]{self.colors['RESET']}")
                    
        except KeyboardInterrupt:
            print(f"\n{self.colors['YELLOW']}Monitoring stopped by user{self.colors['RESET']}")


def main():
    parser = argparse.ArgumentParser(description='Monitor Arc Vision training logs with enhanced formatting')
    parser.add_argument('--log-file', type=str, help='Path to specific log file to monitor')
    parser.add_argument('--output-dir', type=str, default='outputs/arc_vision', 
                       help='Output directory to search for logs (default: outputs/arc_vision)')
    parser.add_argument('--no-follow', action='store_true', 
                       help='Do not follow log file (show current content only)')
    parser.add_argument('--show-all', action='store_true', 
                       help='Show all lines including empty ones')
    parser.add_argument('--no-highlight-epochs', action='store_true', 
                       help='Disable epoch highlighting')
    parser.add_argument('--live', action='store_true',
                       help='Monitor live training output from stdin')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(
        log_file=args.log_file,
        follow=not args.no_follow,
        show_all=args.show_all,
        highlight_epochs=not args.no_highlight_epochs
    )
    
    if args.live:
        # Monitor live output from stdin
        monitor.monitor_live_training()
    elif args.log_file:
        # Monitor specific log file
        if os.path.exists(args.log_file):
            monitor.monitor_file(args.log_file)
        else:
            print(f"Error: Log file '{args.log_file}' not found")
            sys.exit(1)
    else:
        # Find and monitor latest log in output directory
        latest_log = monitor.find_latest_log(args.output_dir)
        if latest_log:
            print(f"Found log file: {latest_log}")
            monitor.monitor_file(latest_log)
        else:
            print(f"No log files found in '{args.output_dir}'")
            print("You can:")
            print("1. Specify a log file with --log-file")
            print("2. Use --live to monitor training output in real-time")
            print("3. Check if the output directory is correct")
            sys.exit(1)


if __name__ == '__main__':
    main()