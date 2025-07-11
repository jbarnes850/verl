#!/usr/bin/env python3
"""Debug script to check data format issues."""

import pandas as pd
import json
from pathlib import Path
import sys

def debug_parquet_file(file_path):
    """Debug the format of data in parquet file."""
    print(f"\n=== Debugging {file_path} ===")
    
    df = pd.read_parquet(file_path)
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check first row
    if len(df) > 0:
        row = df.iloc[0]
        print("\nFirst row data types:")
        for col in df.columns:
            value = row[col]
            print(f"  {col}: {type(value).__name__}")
            if col == 'images':
                print(f"    Value: {value}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"    First element type: {type(value[0])}")
                    print(f"    First element: {value[0]}")
            elif col == 'prompt':
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        print(f"    Parsed type: {type(parsed).__name__}")
                    except:
                        print(f"    Failed to parse as JSON")
                else:
                    print(f"    Direct type: {type(value).__name__}")

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default paths
        for path in ["desktop_task_data_verl/verl_train.parquet", 
                     "desktop_task_data_verl/verl_val.parquet",
                     "desktop_task_data_balanced/verl_train.parquet",
                     "desktop_task_data_balanced/verl_val.parquet"]:
            if Path(path).exists():
                debug_parquet_file(path)
            else:
                print(f"\nFile not found: {path}")

if __name__ == "__main__":
    main()