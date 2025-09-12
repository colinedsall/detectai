#!/usr/bin/env python3
"""
File labeling script to help you categorize your text files as AI or human.
This creates the training data needed for the ML detector.
"""

import os
import json
from typing import Dict, List

def preview_file(file_path: str, max_chars: int = 500) -> str:
    """Show a preview of a file's content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            preview = content[:max_chars]
            if len(content) > max_chars:
                preview += "..."
            return preview
    except Exception as e:
        return f"Error reading file: {e}"

def label_files():
    """Interactive file labeling."""
    
    print("️  File Labeling Tool")
    print("=" * 50)
    print("Label your text files as AI-generated or human-written for training.")
    print()
    
    # Find all text files in current directory
    text_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    if not text_files:
        print(" No .txt files found in current directory!")
        return
    
    print(f" Found {len(text_files)} text files:")
    for i, file_path in enumerate(text_files, 1):
        print(f"  {i}. {file_path}")
    
    print()
    print(" Let's examine each file to determine if it's AI or human...")
    print()
    
    labels = {}
    
    for file_path in text_files:
        print(f" File: {file_path}")
        print("-" * 40)
        
        # Show preview
        preview = preview_file(file_path)
        print("Preview:")
        print(preview)
        print()
        
        # Get user input
        while True:
            label = input("Is this AI-generated or human-written? (ai/human): ").strip().lower()
            if label in ['ai', 'human']:
                labels[file_path] = label
                break
            else:
                print(" Please enter 'ai' or 'human'")
        
        print(f" Labeled as: {label.upper()}")
        print()
    
    # Save labels
    with open('file_labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(" Labels saved to file_labels.json")
    
    # Show summary
    ai_count = sum(1 for label in labels.values() if label == 'ai')
    human_count = sum(1 for label in labels.values() if label == 'human')
    
    print(f"\n Labeling Summary:")
    print(f"  AI files: {ai_count}")
    print(f"  Human files: {human_count}")
    print(f"  Total: {len(labels)}")
    
    return labels

def create_training_config():
    """Create a training configuration file."""
    
    if not os.path.exists('file_labels.json'):
        print(" No labels file found. Please run labeling first.")
        return
    
    with open('file_labels.json', 'r') as f:
        labels = json.load(f)
    
    # Separate files by label
    ai_files = [f for f, label in labels.items() if label == 'ai']
    human_files = [f for f, label in labels.items() if label == 'human']
    
    config = {
        'ai_files': ai_files,
        'human_files': human_files,
        'model_type': 'random_forest',  # or 'logistic_regression'
        'test_size': 0.2,
        'cross_validation_folds': 5
    }
    
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("  Training configuration saved to training_config.json")
    print(f"  AI files: {ai_files}")
    print(f"  Human files: {human_files}")

def main():
    """Main function."""
    
    print("️  AI Text Detector - File Labeling")
    print("=" * 60)
    
    # Check if labels already exist
    if os.path.exists('file_labels.json'):
        print(" Found existing labels file!")
        with open('file_labels.json', 'r') as f:
            labels = json.load(f)
        
        print("Current labels:")
        for file_path, label in labels.items():
            print(f"  {file_path}: {label.upper()}")
        
        relabel = input("\nDo you want to relabel files? (y/n): ").strip().lower()
        if relabel == 'y':
            labels = label_files()
        else:
            print(" Using existing labels.")
    else:
        print(" No labels found. Starting labeling process...")
        labels = label_files()
    
    if labels:
        print("\n" + "="*60)
        create_training = input("Create training configuration? (y/n): ").strip().lower()
        if create_training == 'y':
            create_training_config()
            print("\n Next steps:")
            print("1. Review the training configuration")
            print("2. Run: python train_ml_detector.py")
            print("3. Test the trained model on your files")

if __name__ == "__main__":
    main()
