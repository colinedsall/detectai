#!/usr/bin/env python3
"""
Data collection script that uses config.yaml parameters.
This script only collects training data without training the model.
"""

import os
import yaml
import time
from datetime import datetime
from collect_training_data import TrainingDataCollector

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file {config_path} not found!")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Error parsing {config_path}: {e}")
        return None

def collect_data_with_config(config):
    """Collect training data using configuration parameters."""
    
    print("🌐 Collecting Training Data with Configuration")
    print("=" * 60)
    
    # Extract training parameters
    training_config = config.get('training', {})
    human_articles_per_site = training_config.get('human_articles_per_site', 50)
    ai_samples = training_config.get('ai_samples', 500)
    target_words_per_sample = training_config.get('target_words_per_sample', 700)
    min_human_words = training_config.get('min_human_words', 400)
    max_sites = training_config.get('max_sites', 10)
    
    print(f"📊 Data Collection Configuration:")
    print(f"  Human articles per site: {human_articles_per_site}")
    print(f"  AI samples to generate: {ai_samples}")
    print(f"  Target words per AI sample: {target_words_per_sample}")
    print(f"  Minimum human words: {min_human_words}")
    print(f"  Maximum sites: {max_sites}")
    print()
    
    # Create collector
    collector = TrainingDataCollector()
    
    # Collect human content
    print("🔍 Collecting Human Content...")
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
    
    print(f"🎯 Targeting {len(selected_sites)} websites...")
    human_results = collector.scraper.scrape_multiple_sites(selected_sites)
    
    # Collect AI content
    print(f"\n🤖 Generating AI Content...")
    ai_samples_list = collector.collect_ai_content(ai_samples, target_words_per_sample)
    
    # Create training dataset
    print(f"\n📚 Creating Training Dataset...")
    dataset = collector.create_training_dataset(human_results, ai_samples_list)
    
    # Create auto-labels
    print(f"\n🏷️  Creating Auto-Labels...")
    collector.create_auto_labels(dataset)
    
    return dataset

def main():
    """Main function to run data collection."""
    
    print("🤖 AI Text Detection - Data Collection with Configuration")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    if not config:
        return
    
    start_time = time.time()
    
    try:
        # Collect training data
        dataset = collect_data_with_config(config)
        
        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Data Collection Completed Successfully!")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"📊 Dataset summary:")
        print(f"  Total samples: {dataset['total_samples']}")
        print(f"  Human samples: {dataset['human_samples']}")
        print(f"  AI samples: {dataset['ai_samples']}")
        print(f"  Total words: {dataset['total_words']:,}")
        
        print(f"\n💡 Next steps:")
        print(f"  1. Train the model: python train_with_collected_data.py")
        print(f"  2. Or use config-based training: python train_with_config.py")
        print(f"  3. Test on your files: python analyze_my_files.py")
        
    except KeyboardInterrupt:
        print(f"\n❌ Data collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        print("Check the logs above for details.")

if __name__ == "__main__":
    main()
