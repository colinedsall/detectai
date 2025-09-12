#!/usr/bin/env python3
"""
Training script that uses config.yaml for data collection and model training.
This script reads configuration from config.yaml and orchestrates the entire training process.
"""

import os
import yaml
import time
from datetime import datetime
from collect_training_data import TrainingDataCollector
from train_with_collected_data import train_detector
from app.services.ml_text_detector import MLTextDetector

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f" Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f" Configuration file {config_path} not found!")
        return None
    except yaml.YAMLError as e:
        print(f" Error parsing {config_path}: {e}")
        return None

def collect_training_data_with_config(config):
    """Collect training data using configuration parameters."""
    
    print(" Collecting Training Data with Configuration")
    print("=" * 60)
    
    # Extract training parameters
    training_config = config.get('training', {})
    human_articles_per_site = training_config.get('human_articles_per_site', 20)
    ai_samples = training_config.get('ai_samples', 2000)
    target_words_per_sample = training_config.get('target_words_per_sample', 1000)
    min_human_words = training_config.get('min_human_words', 500)
    max_sites = training_config.get('max_sites', 50)
    
    print(f" Training Configuration:")
    print(f"  Human articles per site: {human_articles_per_site}")
    print(f"  AI samples to generate: {ai_samples}")
    print(f"  Target words per AI sample: {target_words_per_sample}")
    print(f"  Minimum human words: {min_human_words}")
    print(f"  Maximum sites: {max_sites}")
    print()
    
    # Create collector
    collector = TrainingDataCollector()
    
    # Collect human content
    print(" Collecting Human Content...")
    from web_scraper import get_target_sites
    
    sites = get_target_sites()
    # Prioritize RSS-enabled sites
    rss_sites = [s for s in sites if s.get('rss')]
    other_sites = [s for s in sites if not s.get('rss')]
    
    # Use RSS sites first, then others
    selected_sites = rss_sites[:max_sites//2] + other_sites[:max_sites//2]
    selected_sites = selected_sites[:max_sites]
    
    # Update max_pages for each site
    for site in selected_sites:
        site['max_pages'] = human_articles_per_site
    
    print(f" Targeting {len(selected_sites)} websites...")
    human_results = collector.scraper.scrape_multiple_sites(selected_sites)
    
    # Collect AI content
    print(f"\n Generating AI Content...")
    ai_samples_list = collector.collect_ai_content(ai_samples, target_words_per_sample)
    
    # Create training dataset
    print(f"\n Creating Training Dataset...")
    dataset = collector.create_training_dataset(human_results, ai_samples_list)
    
    # Create auto-labels
    print(f"\n Creating Auto-Labels...")
    collector.create_auto_labels(dataset)
    
    return dataset

def train_model_with_config(config, dataset):
    """Train the model using configuration parameters."""
    
    print("\n Training Model with Configuration")
    print("=" * 60)
    
    # Extract model parameters
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'ensemble')
    test_size = model_config.get('test_size', 0.2)
    cv_folds = model_config.get('cv_folds', 5)
    
    print(f" Model Configuration:")
    print(f"  Model type: {model_type}")
    print(f"  Test size: {test_size}")
    print(f"  CV folds: {cv_folds}")
    print()
    
    # Create detector
    detector = MLTextDetector()
    
    # Load training data
    print(" Loading training data...")
    from train_with_collected_data import load_collected_data
    human_texts, ai_texts = load_collected_data()
    
    if not ai_texts or not human_texts:
        print(" No training data found! Please run data collection first.")
        return None
    
    print(f" Training Data Summary:")
    print(f"  AI texts: {len(ai_texts)}")
    print(f"  Human texts: {len(human_texts)}")
    print(f"  Total: {len(ai_texts) + len(human_texts)}")
    print()
    
    # Train the model
    try:
        print(" Training model...")
        accuracy = detector.train_model(
            ai_texts=ai_texts,
            human_texts=human_texts,
            model_type=model_type,
            test_size=test_size
        )
        
        print(f"\n Training completed successfully!")
        print(f" Model accuracy: {accuracy:.3f}")
        print(f" Model saved to: ai_detector_model.pkl")
        
        return detector
        
    except Exception as e:
        print(f" Training failed: {e}")
        return None

def test_model_with_config(config, detector):
    """Test the model on configured files."""
    
    print("\n Testing Model with Configuration")
    print("=" * 60)
    
    # Get files to analyze from config
    files_to_analyze = config.get('files_to_analyze', [])
    
    if not files_to_analyze:
        print("  No files specified in config.yaml for testing")
        return
    
    print(f" Testing on {len(files_to_analyze)} files...")
    
    results = []
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                result = detector.predict(text)
                results.append((file_path, result))
                
                print(f"\n {file_path}:")
                print(f"   AI Probability: {result['probability_ai']:.1%}")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.1%}")
                if result['explanations']:
                    print(f"   Explanations: {', '.join(result['explanations'])}")
                    
            except Exception as e:
                print(f" Failed to analyze {file_path}: {e}")
        else:
            print(f" File not found: {file_path}")
    
    # Summary
    if results:
        print(f"\n Test Summary:")
        print(f"  Files tested: {len(results)}")
        ai_predictions = sum(1 for _, r in results if r['prediction'] == 'AI')
        print(f"  AI predictions: {ai_predictions}")
        print(f"  Human predictions: {len(results) - ai_predictions}")

def main():
    """Main function to run the complete training pipeline."""
    
    print(" AI Text Detection - Configuration-Based Training")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    if not config:
        return
    
    start_time = time.time()
    
    try:
        # Step 1: Collect training data
        print("\n" + "="*70)
        dataset = collect_training_data_with_config(config)
        
        # Step 2: Train model
        print("\n" + "="*70)
        detector = train_model_with_config(config, dataset)
        
        if detector:
            # Step 3: Test model
            print("\n" + "="*70)
            test_model_with_config(config, detector)
        
        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\n Training Pipeline Completed Successfully!")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        
        print(f"\n Next steps:")
        print(f"  1. Review the trained model performance")
        print(f"  2. Test on your own files using: python analyze_my_files.py")
        print(f"  3. Start API server: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
        
    except KeyboardInterrupt:
        print(f"\n Training interrupted by user")
    except Exception as e:
        print(f"\n Error during training: {e}")
        print("Check the logs above for details.")

if __name__ == "__main__":
    main()
