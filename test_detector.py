#!/usr/bin/env python3
"""
Test script for the enhanced text detector.
Demonstrates different input methods and shows the detailed analysis results.
"""

import requests
import json
from typing import Dict, Any

def test_text_detection(text: str = None, url: str = None, html: str = None) -> Dict[str, Any]:
    """Test the text detection API with different input types."""
    
    # Prepare request data
    request_data = {}
    if text:
        request_data['text'] = text
    if url:
        request_data['url'] = url
    if html:
        request_data['html'] = html
    
    # Make API request
    response = requests.post(
        "http://127.0.0.1:8000/v1/detect/text",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}

def print_detection_results(results: Dict[str, Any], input_type: str):
    """Pretty print the detection results."""
    print(f"\n{'='*60}")
    print(f"TEXT DETECTION RESULTS - {input_type.upper()}")
    print(f"{'='*60}")
    
    print(f"AI Probability: {results.get('probability_ai', 0):.3f}")
    print(f"Confidence: {results.get('confidence', 0):.3f}")
    print(f"Processing Time: {results.get('metadata', {}).get('processing_time_ms', 0):.2f}ms")
    print(f"Source Type: {results.get('metadata', {}).get('source_type', 'unknown')}")
    
    print(f"\nText Statistics:")
    metadata = results.get('metadata', {})
    print(f"  Words: {metadata.get('word_count', 0)}")
    print(f"  Characters: {metadata.get('character_count', 0)}")
    print(f"  Unique Words: {metadata.get('unique_words', 0)}")
    print(f"  Avg Word Length: {metadata.get('avg_word_length', 0):.2f}")
    
    print(f"\nDetection Methods:")
    for method in results.get('methods', []):
        print(f"  {method['name'].replace('_', ' ').title()}:")
        print(f"    Confidence: {method['confidence']:.3f}")
        print(f"    Weight: {method['weight']:.2f}")
        print(f"    Explanation: {method['explanation']}")
    
    print(f"\nHighlighted Spans: {len(results.get('highlight_spans', []))}")
    for span in results.get('highlight_spans', [])[:5]:  # Show first 5
        print(f"  [{span['start']}:{span['end']}] {span['reason']} (confidence: {span['confidence']:.2f})")
    
    if len(results.get('highlight_spans', [])) > 5:
        print(f"  ... and {len(results.get('highlight_spans', [])) - 5} more")
    
    print(f"\nDisclaimer: {results.get('disclaimer', '')}")

def main():
    """Run various test cases."""
    
    print("Testing Enhanced Text Detector")
    print("=" * 60)
    
    # Test 1: Simple text input
    sample_text = """
    Artificial intelligence has revolutionized many industries in recent years. 
    Machine learning algorithms can now process vast amounts of data and identify 
    patterns that were previously impossible to detect. This technology continues 
    to evolve rapidly, offering new possibilities for automation and decision-making.
    """
    
    results = test_text_detection(text=sample_text)
    if results:
        print_detection_results(results, "Sample Text")
    
    # Test 2: URL input (using a simple test endpoint)
    print("\n" + "="*60)
    print("Testing URL input (this may take a moment)...")
    
    results = test_text_detection(url="https://httpbin.org/html")
    if results:
        print_detection_results(results, "URL Content")
    
    # Test 3: HTML input
    sample_html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Welcome to the Test Page</h1>
            <p>This is a paragraph with some <strong>bold text</strong> and <em>italic text</em>.</p>
            <p>Another paragraph to provide more content for analysis.</p>
            <script>console.log('This should be ignored');</script>
        </body>
    </html>
    """
    
    results = test_text_detection(html=sample_html)
    if results:
        print_detection_results(results, "HTML Content")
    
    # Test 4: AI-like text (repetitive, formulaic)
    ai_like_text = """
    The implementation of machine learning algorithms requires careful consideration 
    of several key factors. First, data preprocessing is essential for optimal performance. 
    Second, feature engineering plays a crucial role in model accuracy. Third, 
    hyperparameter tuning is necessary for achieving the best results. Fourth, 
    model evaluation metrics must be carefully selected. Fifth, deployment strategies 
    should be planned in advance.
    """
    
    results = test_text_detection(text=ai_like_text)
    if results:
        print_detection_results(results, "AI-like Text")

if __name__ == "__main__":
    main()
