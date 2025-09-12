#!/usr/bin/env python3
"""
Training script for the ML-based text detector.
Uses your example files to train a model that can distinguish between AI and human text.
"""

import os
from app.services.ml_text_detector import MLTextDetector

def load_text_from_file(file_path: str) -> str:
    """Load text content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f" Error reading {file_path}: {e}")
        return ""

def create_training_data():
    """Create training data from your example files."""
    
    print(" Creating Training Dataset...")
    print("=" * 50)
    
    # Define which files are AI vs Human
    # You can modify these based on your knowledge of the content
    ai_files = [
        "sample_ai_text.txt",  # We know this is AI-generated
        # Add other AI-generated files here
    ]
    
    human_files = [
        "example.txt",         # Assuming this is human-written
        "example1.txt",        # Assuming this is human-written
        "example2.txt",        # Assuming this is human-written
        # Add other human-written files here
    ]
    
    # Load AI texts
    ai_texts = []
    for file_path in ai_files:
        if os.path.exists(file_path):
            text = load_text_from_file(file_path)
            if text:
                ai_texts.append(text)
                print(f" Loaded AI text: {file_path} ({len(text)} characters)")
        else:
            print(f" AI file not found: {file_path}")
    
    # Load human texts
    human_texts = []
    for file_path in human_files:
        if os.path.exists(file_path):
            text = load_text_from_file(file_path)
            if text:
                human_texts.append(text)
                print(f" Loaded human text: {file_path} ({len(text)} characters)")
        else:
            print(f" Human file not found: {file_path}")
    
    print(f"\n Dataset Summary:")
    print(f"  AI texts: {len(ai_texts)}")
    print(f"  Human texts: {len(human_texts)}")
    print(f"  Total samples: {len(ai_texts) + len(human_texts)}")
    
    return ai_texts, human_texts

def train_detector():
    """Train the ML text detector."""
    
    print("\n Training ML Text Detector")
    print("=" * 50)
    
    # Create detector
    detector = MLTextDetector()
    
    # Create training data
    ai_texts, human_texts = create_training_data()
    
    if not ai_texts or not human_texts:
        print(" Need both AI and human texts to train!")
        return None
    
    # Train the model
    try:
        accuracy = detector.train_model(
            ai_texts=ai_texts,
            human_texts=human_texts,
            model_type="random_forest",  # or "logistic_regression"
            test_size=0.2
        )
        
        print(f"\n Training completed with {accuracy:.3f} accuracy!")
        return detector
        
    except Exception as e:
        print(f" Training failed: {e}")
        return None

def test_detector(detector: MLTextDetector):
    """Test the trained detector on your files."""
    
    print("\n Testing Trained Detector")
    print("=" * 50)
    
    # Test files to evaluate
    test_files = [
        "sample_ai_text.txt",
        "example.txt",
        "example1.txt", 
        "example2.txt"
    ]
    
    print(" Testing on your files:")
    for file_path in test_files:
        if os.path.exists(file_path):
            text = load_text_from_file(file_path)
            if text:
                try:
                    result = detector.predict(text)
                    print(f"\n {file_path}:")
                    print(f"   AI Probability: {result['probability_ai']:.1%}")
                    print(f"   Prediction: {result['prediction']}")
                    print(f"   Confidence: {result['confidence']:.1%}")
                    if result['explanations']:
                        print(f"   Explanations: {', '.join(result['explanations'])}")
                except Exception as e:
                    print(f"   Prediction failed: {e}")
        else:
            print(f" Test file not found: {file_path}")

def interactive_training():
    """Interactive training mode."""
    
    print(" Interactive ML Text Detector Training")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Train new model")
        print("2. Test existing model")
        print("3. Evaluate model performance")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            detector = train_detector()
            if detector:
                print(" Model trained successfully!")
        
        elif choice == "2":
            detector = MLTextDetector()
            if detector.is_trained:
                test_detector(detector)
            else:
                print(" No trained model found. Please train first.")
        
        elif choice == "3":
            detector = MLTextDetector()
            if detector.is_trained:
                # Evaluate on your files
                ai_files = ["sample_ai_text.txt"]
                human_files = ["example.txt", "example1.txt", "example2.txt"]
                detector.evaluate_on_files(ai_files, human_files)
            else:
                print(" No trained model found. Please train first.")
        
        elif choice == "4":
            print(" Goodbye!")
            break
        
        else:
            print(" Invalid choice. Please enter 1-4.")

def main():
    """Main training function."""
    
    print(" ML Text Detector Training")
    print("=" * 60)
    
    # Check if model already exists
    if os.path.exists("ai_detector_model.pkl"):
        print(" Found existing trained model!")
        detector = MLTextDetector()
        
        if detector.is_trained:
            print(" Model loaded successfully!")
            
            # Ask if user wants to retrain
            retrain = input("Do you want to retrain the model? (y/n): ").strip().lower()
            if retrain == 'y':
                detector = train_detector()
                if detector:
                    test_detector(detector)
            else:
                test_detector(detector)
        else:
            print(" Failed to load existing model. Training new one...")
            detector = train_detector()
            if detector:
                test_detector(detector)
    else:
        print(" No existing model found. Training new one...")
        detector = train_detector()
        if detector:
            test_detector(detector)
    
    # Offer interactive mode
    print("\n" + "="*60)
    interactive = input("Enter interactive training mode? (y/n): ").strip().lower()
    if interactive == 'y':
        interactive_training()

if __name__ == "__main__":
    main()
