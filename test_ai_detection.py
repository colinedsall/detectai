#!/usr/bin/env python3
"""
Test script for AI-generated text detection.
Tests the detector against a sample AI-generated text file.
"""

import requests
import json
from typing import Dict, Any

def test_ai_text_detection(file_path: str = "sample_ai_text.txt") -> Dict[str, Any]:
    """Test the text detection API with an AI-generated text file."""
    
    print(f" Testing AI text detection on file: {file_path}")
    print("=" * 60)
    
    # Make API request with file path
    response = requests.post(
        "http://127.0.0.1:8000/v1/detect/text",
        json={"file_path": file_path},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f" Error: {response.status_code}")
        print(response.text)
        return {}

def print_detailed_results(results: Dict[str, Any]):
    """Print detailed detection results with analysis."""
    
    print(f"\n DETECTION RESULTS")
    print("=" * 60)
    
    # Main scores
    ai_probability = results.get('probability_ai', 0)
    confidence = results.get('confidence', 0)
    
    print(f" AI Probability: {ai_probability:.1%}")
    print(f" Confidence: {confidence:.1%}")
    
    # Overall assessment
    if ai_probability > 0.7:
        print(" HIGH likelihood of AI-generated content")
    elif ai_probability > 0.5:
        print(" MODERATE likelihood of AI-generated content")
    else:
        print(" LOW likelihood of AI-generated content")
    
    # Text statistics
    metadata = results.get('metadata', {})
    print(f"\n Text Statistics:")
    print(f"  Words: {metadata.get('word_count', 0)}")
    print(f"  Characters: {metadata.get('character_count', 0)}")
    print(f"  Unique Words: {metadata.get('unique_words', 0)}")
    print(f"  Vocabulary Diversity: {metadata.get('unique_words', 0) / max(1, metadata.get('word_count', 1)):.1%}")
    print(f"  Avg Word Length: {metadata.get('avg_word_length', 0):.2f}")
    print(f"  Processing Time: {metadata.get('processing_time_ms', 0):.2f}ms")
    print(f"  Source Type: {metadata.get('source_type', 'unknown')}")
    
    # Method breakdown
    print(f"\n Detection Methods:")
    methods = results.get('methods', [])
    for method in methods:
        method_name = method['name'].replace('_', ' ').title()
        method_conf = method['confidence']
        method_weight = method['weight']
        method_explanation = method['explanation']
        
        # Color coding for method confidence
        if method_conf > 0.7:
            confidence_icon = ""
        elif method_conf > 0.5:
            confidence_icon = ""
        else:
            confidence_icon = ""
        
        print(f"  {confidence_icon} {method_name}:")
        print(f"    Confidence: {method_conf:.1%}")
        print(f"    Weight: {method_weight:.1%}")
        print(f"    Explanation: {method_explanation}")
    
    # Highlighted suspicious sections
    highlight_spans = results.get('highlight_spans', [])
    print(f"\n⚠️  Suspicious Sections: {len(highlight_spans)}")
    
    if highlight_spans:
        # Group by method
        by_method = {}
        for span in highlight_spans:
            method = span['method']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(span)
        
        for method, spans in by_method.items():
            method_name = method.replace('_', ' ').title()
            print(f"\n  {method_name} ({len(spans)} sections):")
            for span in spans[:3]:  # Show first 3 per method
                reason = span['reason']
                conf = span['confidence']
                print(f"    • {reason} (confidence: {conf:.1%})")
            if len(spans) > 3:
                print(f"    ... and {len(spans) - 3} more")
    else:
        print("  No suspicious sections detected")
    
    # Disclaimer
    print(f"\n Disclaimer: {results.get('disclaimer', '')}")

def compare_with_human_text():
    """Compare AI-generated text with human-written text."""
    
    print("\n" + "="*60)
    print(" COMPARISON: AI vs Human Text")
    print("="*60)
    
    # Test AI-generated text
    print("\n AI-Generated Text (sample_ai_text.txt):")
    ai_results = test_ai_text_detection("sample_ai_text.txt")
    if ai_results:
        print_detailed_results(ai_results)
    
    # Test human-like text
    print("\n Human-Like Text (sample text):")
    human_text = """
    I went to the store yesterday to buy some groceries. The weather was nice, 
    so I decided to walk instead of driving. On my way there, I saw my neighbor 
    walking their dog. We chatted for a few minutes about the weather and how 
    their kids were doing in school. The store was busy, but I managed to find 
    everything I needed. The cashier was friendly and we talked about the local 
    sports team while she rang up my items.
    """
    
    human_response = requests.post(
        "http://127.0.0.1:8000/v1/detect/text",
        json={"text": human_text},
        headers={"Content-Type": "application/json"}
    )
    
    if human_response.status_code == 200:
        human_results = human_response.json()
        print_detailed_results(human_results)
        
        # Show comparison
        ai_prob = ai_results.get('probability_ai', 0)
        human_prob = human_results.get('probability_ai', 0)
        
        print(f"\n COMPARISON SUMMARY:")
        print(f"  AI Text: {ai_prob:.1%} AI probability")
        print(f"  Human Text: {human_prob:.1%} AI probability")
        print(f"  Difference: {abs(ai_prob - human_prob):.1%}")
        
        if ai_prob > human_prob:
            print("   Detector correctly identified AI text as more likely AI-generated")
        else:
            print("   Detector may need tuning")

def main():
    """Main test function."""
    
    print(" AI-Generated Text Detection Test")
    print("=" * 60)
    
    # Test the AI-generated text file
    results = test_ai_text_detection()
    if results:
        print_detailed_results(results)
    
    # Compare with human text
    compare_with_human_text()
    
    print(f"\n Test completed! Check the results above.")

if __name__ == "__main__":
    main()
