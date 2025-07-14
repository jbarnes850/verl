#!/usr/bin/env python3
"""Test script for advisor model."""

import requests
import json

def test_advisor(endpoint="http://localhost:8000/v1", model="qwen3_4b_sec_advisor_step80"):
    """Test the advisor model with a sample CRM query."""
    
    # Test completions endpoint
    print("Testing advisor model...")
    
    messages = [
        {"role": "system", "content": "You are a CRM expert advisor. Suggest the best SOQL query approach."},
        {"role": "user", "content": "I need to find all opportunities that closed last quarter with amount > $50,000"}
    ]
    
    response = requests.post(
        f"{endpoint}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 200
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nAdvisor suggestion:")
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_advisor()
