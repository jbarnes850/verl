#!/usr/bin/env python3
"""
Regenerate data using verl_data_utils.py and verify the conversion is correct.
"""

import sys
import pandas as pd
import json
from datasets import load_dataset

# Add the crm-arena src directory to path
sys.path.append('crm-arena/src')

from verl_data_utils import VERLDataFormatter

def main():
    print("="*80)
    print("REGENERATING DATA WITH verl_data_utils.py")
    print("="*80)
    
    # Create formatter instance
    formatter = VERLDataFormatter(output_dir="./data/verl_format")
    
    # Use the balanced split from the config
    train_tasks = [
        # WORKFLOW (1/2 tasks)
        "case_routing",
        # POLICY (3/4 tasks) 
        "invalid_config",
        "policy_violation_identification", 
        "lead_qualification",
        # TEXT (4/5 tasks)
        "knowledge_qa",
        "sales_insight_mining",
        "wrong_stage_rectification",
        "activity_priority",
        # DATABASE (6/8 tasks)
        "handle_time",
        "transfer_count", 
        "top_issue_identification",
        "monthly_trend_analysis",
        "best_region_identification",
        "sales_amount_understanding"
    ]
    
    test_tasks = [
        # WORKFLOW (1/2 tasks)
        "lead_routing",
        # POLICY (1/4 tasks)
        "quote_approval",
        # TEXT (1/5 tasks)
        "named_entity_disambiguation",
        # DATABASE (2/8 tasks)
        "sales_cycle_understanding",
        "conversion_rate_comprehension"
    ]
    
    print(f"\nTrain tasks: {len(train_tasks)} tasks")
    print(f"Test tasks: {len(test_tasks)} tasks")
    
    # Mix all tasks and split with fixed sizes: 100 val, 100 test, rest train
    print("\nRegenerating data files with fixed splits...")
    print("  Mixing all 19 tasks together")
    print("  Val: 100 examples")
    print("  Test: 100 examples")  
    print("  Train: remaining examples")
    print("  Using max 100 examples per task")
    
    # Generate all data in one pass
    train_path, val_path, test_path = formatter.format_crmarena_for_verl(
        dataset_name="Salesforce/CRMArenaPro",
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        max_examples_per_task=100
    )
    
    print(f"\n✓ Data files generated successfully!")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")
    
    # Verify the generated data
    print("\n" + "="*80)
    print("VERIFYING GENERATED DATA")
    print("="*80)
    
    # Load original HuggingFace data for comparison
    print("\n1. Loading original HuggingFace data for comparison...")
    try:
        dataset = load_dataset('Salesforce/CRMArenaPro', 'CRMArenaPro')
        
        # Get a few examples to compare
        hf_examples = {}
        for i in range(min(10, len(dataset['b2b']))):
            ex = dataset['b2b'][i]
            task = ex['task']
            if task in train_tasks or task in test_tasks:
                hf_examples[task] = ex
                
        print(f"✓ Loaded {len(hf_examples)} relevant examples from HuggingFace")
        
    except Exception as e:
        print(f"✗ Error loading HuggingFace data: {e}")
        return False
    
    # Verify each generated file
    all_good = True
    for split_name, file_path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        print(f"\n2. Checking {split_name} split...")
        
        df = pd.read_parquet(file_path)
        print(f"   Total examples: {len(df)}")
        
        # Check if responses are populated
        non_empty = (df['response'] != '').sum()
        empty = (df['response'] == '').sum()
        
        print(f"   Ground truth present: {non_empty}/{len(df)} ({non_empty/len(df)*100:.1f}%)")
        print(f"   Empty responses: {empty}")
        
        if non_empty == 0:
            print("   ✗ ERROR: No ground truth found!")
            all_good = False
            continue
        else:
            print("   ✓ Ground truth is present!")
        
        # Show distribution by task
        print(f"\n   Task distribution:")
        task_counts = df['task_name'].value_counts()
        for task, count in task_counts.items():
            print(f"     {task}: {count} examples")
            
        # Verify a few examples match HuggingFace data
        print(f"\n   Verifying data integrity (checking first 3 examples)...")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            original_task = row['original_task']
            
            if original_task in hf_examples:
                hf_ex = hf_examples[original_task]
                
                # Check answer conversion
                hf_answer = hf_ex['answer']
                parquet_response = row['response']
                expected_response = ", ".join(hf_answer) if isinstance(hf_answer, list) else str(hf_answer)
                
                if parquet_response == expected_response:
                    print(f"     ✓ Example {i}: '{original_task}' - answer correctly converted")
                else:
                    print(f"     ✗ Example {i}: '{original_task}' - answer mismatch!")
                    print(f"       HuggingFace: {hf_answer}")
                    print(f"       Expected: {expected_response}")
                    print(f"       Got: {parquet_response}")
                    all_good = False
                    
                # Check reward_model ground truth
                reward_gt = row['reward_model'].get('ground_truth', '')
                if reward_gt == expected_response:
                    print(f"       ✓ reward_model ground_truth matches")
                else:
                    print(f"       ✗ reward_model ground_truth mismatch!")
                    all_good = False
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if all_good:
        print("✓ All checks passed!")
        print("✓ Data regeneration successful with proper ground truth")
        print("✓ Ready for training and evaluation")
    else:
        print("✗ Some issues found during verification")
        print("✗ Please check the errors above")
        
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)