"""
CRM TEXT skill reward function using CRMArena's evaluation logic.
Adapted from CRMArena's Evaluator class for use in VERL training.
"""

import json
import re
from typing import Dict, Any, Union, List
import string
from collections import Counter
import math


def normalize_answer(s):
    """Normalize text for fuzzy matching - from CRMArena's get_all_metrics"""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    
    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def compute_f1(prediction, ground_truth):
    """Compute F1 score - from CRMArena's get_all_metrics"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # If either is empty
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def parse_answer_for_exact_match(response: str, task_name: str) -> List[str]:
    """
    Simple parsing logic for exact match tasks.
    In production CRMArena uses an LLM for this, but for training
    we'll use rule-based parsing.
    """
    response = response.strip()
    
    # Try to parse as JSON list first
    if response.startswith('[') and response.endswith(']'):
        try:
            # Handle both single and double quotes
            cleaned = response.replace("'", '"')
            items = json.loads(cleaned)
            return [str(item).strip() for item in items]
        except:
            # Fallback: manual parsing for simple lists
            inner = response[1:-1]  # Remove brackets
            if inner:
                # Split by comma and clean each item
                items = [item.strip().strip("'\"") for item in inner.split(',')]
                return [item for item in items if item]
    
    # For single IDs (named_entity_disambiguation, activity_priority)
    if task_name in ['named_entity_disambiguation', 'activity_priority']:
        # First check if the whole response is a valid ID
        if re.match(r'^[0-9a-zA-Z]{15,18}$', response):
            return [response]
        # Look for ID patterns (alphanumeric strings)
        id_pattern = r'\b[0-9a-zA-Z]{15,18}\b'
        matches = re.findall(id_pattern, response)
        if matches:
            return matches
    
    # For stage names (wrong_stage_rectification)
    elif task_name == 'wrong_stage_rectification':
        stages = ['Qualification', 'Discovery', 'Quote', 'Negotiation', 'Closed']
        # First check exact match
        if response in stages:
            return [response]
        # Then check case-insensitive
        for stage in stages:
            if response.lower() == stage.lower():
                return [stage]
        # Finally check if contained
        for stage in stages:
            if stage.lower() in response.lower():
                return [stage]
    
    # Default: return the whole response as a single item
    if response and response.lower() != 'none':
        return [response]
    
    return ["None"]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs
) -> float:
    """
    Compute reward score for CRM TEXT skill tasks using CRMArena logic.
    
    Args:
        data_source: Task type (e.g., 'knowledge_qa', 'sales_insight_mining')
        solution_str: Model's response
        ground_truth: Expected correct answer (string representation of list)
        extra_info: Additional information (optional)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        float: Reward score
        - For exact_match: 1.0 (correct) or 0.0 (incorrect)
        - For fuzzy_match: F1 score between 0.0 and 1.0
        - Fallback: 0.1 for well-formed but incorrect responses
    """
    
    # Define task types
    exact_match_tasks = ['activity_priority', 'named_entity_disambiguation', 'wrong_stage_rectification']
    fuzzy_match_tasks = ['knowledge_qa', 'sales_insight_mining']
    
    # Clean inputs
    response = solution_str.strip()
    ground_truth = ground_truth.strip()
    
    # Parse ground truth
    gt_list = None
    if ground_truth.startswith('[') and ground_truth.endswith(']'):
        try:
            # Handle both single and double quotes in ground truth
            cleaned_gt = ground_truth.replace("'", '"')
            gt_list = json.loads(cleaned_gt)
            gt_list = [str(item) for item in gt_list]
        except:
            # Fallback: manual parsing
            inner = ground_truth[1:-1]  # Remove brackets
            if inner:
                # Split by comma and clean each item
                items = [item.strip().strip("'\"") for item in inner.split(',')]
                gt_list = [item for item in items if item]
            else:
                gt_list = [ground_truth]
    else:
        gt_list = [ground_truth]
    
    # Handle None values
    if gt_list == [None] or gt_list == ['None']:
        gt_list = ["None"]
    
    # Empty response gets 0
    if not response or response.isspace():
        return 0.0
    
    # Exact match evaluation
    if data_source in exact_match_tasks:
        # Parse response
        parsed_response = parse_answer_for_exact_match(response, data_source)
        
        # Sort and compare
        if sorted(parsed_response) == sorted(gt_list):
            return 1.0
        
        # For exact match tasks, CRMArena gives partial credit (0.1) only for:
        # 1. Well-formed IDs that are wrong
        # 2. Valid stage names that are wrong
        # But NOT for partial lists or single items from a list
        
        # Check if response is well-formed but wrong
        if data_source in ['named_entity_disambiguation', 'activity_priority']:
            # For ID tasks: only give 0.1 if it's a valid ID format but wrong
            # AND the ground truth is also a single ID (not a list)
            if len(gt_list) == 1 and len(parsed_response) == 1:
                # Single ID expected, single ID given
                if len(parsed_response[0]) >= 15 and any(c.isdigit() for c in parsed_response[0]):
                    return 0.1
            # For lists, only give 0.1 if response is also a proper list with valid IDs
            elif len(gt_list) > 1 and len(parsed_response) > 1:
                # All items must be valid IDs
                if all(len(item) >= 15 and any(c.isdigit() for c in item) for item in parsed_response):
                    return 0.1
        elif data_source == 'wrong_stage_rectification':
            valid_stages = ['qualification', 'discovery', 'quote', 'negotiation', 'closed']
            if len(parsed_response) == 1 and parsed_response[0].lower() in valid_stages:
                return 0.1
        
        return 0.0
    
    # Fuzzy match evaluation
    elif data_source in fuzzy_match_tasks:
        # Join list items for ground truth
        gt_text = ' '.join(gt_list) if isinstance(gt_list, list) else str(gt_list)
        
        # Compute F1 score (main metric for fuzzy match in CRMArena)
        f1_score = compute_f1(response, gt_text)
        
        if f1_score >= 0.8:
            return 1.0
        elif len(response.split()) >= 3:
            return 0.1
        else:
            return 0.0
    
    # Unknown task type - fallback scoring
    else:
        # Simple exact match check
        if response in gt_list or response == ground_truth:
            return 1.0
        elif len(response) > 5:  # Well-formed attempt
            return 0.1
        else:
            return 0.0