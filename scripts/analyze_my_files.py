#!/usr/bin/env python3
"""
Customizable script to analyze your own text files for AI detection.
Uses the trained ML model directly and reads file list from config.yaml.
"""

import yaml
import os
import pathlib
from typing import Dict, Any, List
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from app.services.ml_text_detector import MLTextDetector

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(config_path):
        print(f" Config file not found: {config_path}")
        print("Please create config.yaml with your file paths.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f" Error loading config: {e}")
        return {}

def analyze_file(file_path: str, detector: MLTextDetector) -> Dict[str, Any]:
    """Analyze a single file for AI detection."""
    
    print(f"\n Analyzing: {file_path}")
    print("-" * 50)
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            return {}
        
        # Read and analyze file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = detector.predict(text)
        
        # Print results
        ai_probability = result['probability_ai']
        confidence = result['confidence']
        
        print(f" AI Probability: {ai_probability:.1%}")
        print(f" Confidence: {confidence:.1%}")
        
        # Assessment
        if ai_probability > 0.7:
            assessment = " HIGH likelihood of AI-generated content"
        elif ai_probability > 0.5:
            assessment = " MODERATE likelihood of AI-generated content"
        else:
            assessment = " LOW likelihood of AI-generated content"
        
        print(f" Assessment: {assessment}")
        
        # Text stats
        features = result['features']
        print(f" Stats: {features['word_count']} words, "
              f"{int(features['word_count'] * features['type_token_ratio'])} unique "
              f"({features['type_token_ratio']:.1%} diversity)")
        
        # Explanations
        if result['explanations']:
            print(f" Signals: {', '.join(result['explanations'])}")
        
        print()
        return result
        
    except Exception as e:
        print(f" Failed to analyze {file_path}: {str(e)}")
        return {}

def batch_analyze_files():
    """Analyze all files specified in config.yaml."""
    
    print(" AI Text Detection - Batch Analysis")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    if not config:
        return
    
    files_to_analyze = config.get('files_to_analyze', [])
    if not files_to_analyze:
        print(" No files specified in config.yaml!")
        print("Please add file paths to the 'files_to_analyze' section.")
        return
    
    print(f" Analyzing {len(files_to_analyze)} files...")
    
    # Load trained model
    try:
        detector = MLTextDetector()
        if not detector.is_trained:
            print(" No trained model found!")
            print("Please run the training script first:")
            print("  python3 train_with_collected_data.py")
            return
    except Exception as e:
        print(f" Failed to load model: {e}")
        return
    
    all_results = []
    
    # Check which files exist
    existing_files = []
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"   {file_path}")
        else:
            print(f"   {file_path} (not found)")
    
    if not existing_files:
        print(" No files found to analyze!")
        return
    
    # Analyze files
    for file_path in existing_files:
        results = analyze_file(file_path, detector)
        if results:
            all_results.append((file_path, results))
    
    # Summary comparison
    if all_results:
        print("\n" + "="*60)
        print(" BATCH ANALYSIS SUMMARY")
        print("="*60)
        
        # Sort by AI probability
        sorted_results = sorted(all_results, key=lambda x: x[1]['probability_ai'], reverse=True)
        
        print("Ranked by AI Probability (highest to lowest):")
        for i, (file_path, results) in enumerate(sorted_results, 1):
            ai_prob = results['probability_ai']
            confidence = results['confidence']
            word_count = results['features']['word_count']
            
            print(f"{i:2d}. {os.path.basename(file_path):<30} "
                  f"AI: {ai_prob:>5.1%} | Confidence: {confidence:>5.1%} | Words: {word_count:>4}")
        
        # Statistics
        ai_probs = [r[1]['probability_ai'] for r in all_results]
        avg_ai_prob = sum(ai_probs) / len(ai_probs)
        max_ai_prob = max(ai_probs)
        min_ai_prob = min(ai_probs)
        
        print(f"\n Statistics:")
        print(f"  Average AI Probability: {avg_ai_prob:.1%}")
        print(f"  Highest AI Probability: {max_ai_prob:.1%}")
        print(f"  Lowest AI Probability: {min_ai_prob:.1%}")
        print(f"  Range: {max_ai_prob - min_ai_prob:.1%}")

def interactive_mode():
    """Interactive mode to analyze files one by one."""
    
    print(" Interactive AI Text Detection")
    print("=" * 60)
    
    # Load trained model
    try:
        detector = MLTextDetector()
        if not detector.is_trained:
            print(" No trained model found!")
            print("Please run the training script first:")
            print("  python3 train_with_collected_data.py")
            return
    except Exception as e:
        print(f" Failed to load model: {e}")
        return
    
    while True:
        print("\nOptions:")
        print("1. Analyze a specific file")
        print("2. Run batch analysis on all files")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            file_path = input("Enter file path: ").strip()
            if file_path:
                analyze_file(file_path, detector)
        
        elif choice == "2":
            batch_analyze_files()
        
        elif choice == "3":
            print(" Goodbye!")
            break
        
        else:
            print(" Invalid choice. Please enter 1, 2, or 3.")

def main():
    """Main function with options."""
    
    print(" AI Text Detection Tool")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    if not config:
        return
    
    files_to_analyze = config.get('files_to_analyze', [])
    
    if not files_to_analyze:
        print(" No files specified in config.yaml!")
        print("Please add file paths to the 'files_to_analyze' section.")
        return
    
    print(" Files to analyze:")
    for i, file_path in enumerate(files_to_analyze, 1):
        exists = "" if os.path.exists(file_path) else ""
        print(f"  {i}. {exists} {file_path}")
    
    print(f"\nOptions:")
    print("1. Run batch analysis on all files")
    print("2. Interactive mode")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "1":
        batch_analyze_files()
    elif choice == "2":
        interactive_mode()
    else:
        print(" Invalid choice. Running batch analysis...")
        batch_analyze_files()

if __name__ == "__main__":
    main()
