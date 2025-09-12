#!/usr/bin/env python3
"""
Train the ML text detector using collected web scraping and AI generation data.
"""

import os
import json
from app.services.ml_text_detector import MLTextDetector

def load_collected_data():
    """Load the collected training data."""
    
    print(" Loading Collected Training Data")
    print("=" * 50)
    
    # Check if training data exists
    if not os.path.exists('training_data'):
        print(" No training data found!")
        print("Please run: python collect_training_data.py")
        return None, None
    
    # Load labels
    labels_file = 'training_data/auto_labels.json'
    if not os.path.exists(labels_file):
        print(" No labels file found!")
        print("Please run: python collect_training_data.py")
        return None, None
    
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    # Load human texts
    human_texts = []
    human_dir = 'training_data/human'
    if os.path.exists(human_dir):
        human_files = [f for f in os.listdir(human_dir) if f.endswith('.txt')]
        for filename in human_files:
            filepath = os.path.join(human_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract just the text content (skip metadata)
                    text_start = content.find('-' * 80)
                    if text_start != -1:
                        text = content[text_start + 80:].strip()
                        if text:
                            human_texts.append(text)
                            print(f"   Loaded human: {filename} ({len(text.split())} words)")
            except Exception as e:
                print(f"   Error loading {filename}: {e}")
    
    # Load AI texts
    ai_texts = []
    ai_dir = 'training_data/ai'
    if os.path.exists(ai_dir):
        ai_files = [f for f in os.listdir(ai_dir) if f.endswith('.txt')]
        for filename in ai_files:
            filepath = os.path.join(ai_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract just the text content (skip metadata)
                    text_start = content.find('-' * 80)
                    if text_start != -1:
                        text = content[text_start + 80:].strip()
                        if text:
                            ai_texts.append(text)
                            print(f"   Loaded AI: {filename} ({len(text.split())} words)")
            except Exception as e:
                print(f"   Error loading {filename}: {e}")
    
    print(f"\n Data Summary:")
    print(f"  Human texts: {len(human_texts)}")
    print(f"  AI texts: {len(ai_texts)}")
    print(f"  Total: {len(human_texts) + len(ai_texts)}")
    
    return human_texts, ai_texts

def train_detector(human_texts, ai_texts):
    """Train the ML detector with collected data."""
    
    print(f"\n Training ML Text Detector")
    print("=" * 50)
    
    if not human_texts or not ai_texts:
        print(" Need both human and AI texts to train!")
        return None
    
    # Create detector
    detector = MLTextDetector()
    
    # Train the model
    try:
        print(f" Training with {len(human_texts)} human and {len(ai_texts)} AI samples...")
        
        accuracy = detector.train_model(
            ai_texts=ai_texts,
            human_texts=human_texts,
            model_type="ensemble",
            test_size=0.2
        )
        
        print(f"\n Training completed with {accuracy:.3f} accuracy!")
        return detector
        
    except Exception as e:
        print(f" Training failed: {e}")
        return None

def test_on_collected_data(detector):
    """Test the trained detector on the collected data."""
    
    print("\n--->>> RUNNING THE LATEST VERSION OF THE TEST FUNCTION <<<---\n")

    print(f"\n Testing on Collected Data")
    print("=" * 50)
    
    if not detector or not detector.is_trained:
        print(" No trained model available!")
        return
    
    # Test on a few samples from each category
    test_samples = []
    
    # Add some human samples
    human_dir = 'training_data/human'
    if os.path.exists(human_dir):
        human_files = [f for f in os.listdir(human_dir) if f.endswith('.txt')][:5]
        for filename in human_files:
            filepath = os.path.join(human_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_start = content.find('-' * 80)
                    if text_start != -1:
                        text = content[text_start + 80:].strip()
                        if text:
                            test_samples.append(('human', filename, text))
            except Exception as e:
                print(f"   Error loading {filename}: {e}")
    
    # Add some AI samples
    ai_dir = 'training_data/ai'
    if os.path.exists(ai_dir):
        ai_files = [f for f in os.listdir(ai_dir) if f.endswith('.txt')][:5]
        for filename in ai_files:
            filepath = os.path.join(ai_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_start = content.find('-' * 80)
                    if text_start != -1:
                        text = content[text_start + 80:].strip()
                        if text:
                            test_samples.append(('ai', filename, text))
            except Exception as e:
                print(f"   Error loading {filename}: {e}")
    
    # Test each sample
    print(f"Testing {len(test_samples)} samples...")
    
    correct_predictions = 0
    total_predictions = len(test_samples)
    
    for true_label, filename, text in test_samples:
        try:
            result = detector.predict(text)
            predicted_label = result['prediction'].lower()
            
            # Check if prediction is correct
            is_correct = (true_label == 'human' and predicted_label == 'human') or \
                        (true_label == 'ai' and predicted_label == 'ai')
            
            if is_correct:
                correct_predictions += 1
                status = ""
            else:
                status = ""
            
            print(f"{status} {filename}:")
            print(f"    True: {true_label.upper()}")
            print(f"    Predicted: {predicted_label.upper()}")
            print(f"    AI Probability: {result['probability_ai']:.1%}")
            print(f"    Confidence: {result['confidence']:.1%}")
            
        except Exception as e:
            print(f" Error testing {filename}: {e}")
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n Test Results:")
    print(f"  Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"  Accuracy: {accuracy:.1%}")

def main():
    """Main training function."""
    
    print(" ML Text Detector Training with Collected Data")
    print("=" * 70)
    
    # Check if we have collected data
    if not os.path.exists('training_data'):
        print(" No training data found!")
        print("\n To collect training data, run:")
        print("   python collect_training_data.py")
        print("\nThis will:")
        print("  - Scrape human-written content from websites")
        print("  - Generate AI-written content samples")
        print("  - Create a labeled training dataset")
        return
    
    # Load collected data
    human_texts, ai_texts = load_collected_data()
    
    if not human_texts or not ai_texts:
        print(" Insufficient data for training!")
        return
    
    # Train the detector
    detector = train_detector(human_texts, ai_texts)
    
    if detector:
        # Test on collected data
        test_on_collected_data(detector)
        
        print(f"\n Training completed successfully!")
        print(f" Model saved as: ai_detector_model.pkl")
        print(f"\n Next steps:")
        print(f"  1. Test on your own files: python analyze_my_files.py")
        print(f"  2. Use the API endpoint: curl -X POST http://localhost:8000/v1/detect/text")
        print(f"  3. Integrate into your applications")
    else:
        print(" Training failed!")

if __name__ == "__main__":
    main()
