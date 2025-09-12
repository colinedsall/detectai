#!/usr/bin/env python3
"""
Master script to collect training data for AI text detection.
Orchestrates web scraping and AI content generation.
"""

import os
import json
import time
from datetime import datetime
from web_scraper import WebScraper, get_target_sites
from ai_content_generator import AIContentGenerator

class TrainingDataCollector:
    def __init__(self):
        """Initialize the training data collector."""
        self.scraper = WebScraper(delay_range=(2, 5))
        self.ai_generator = AIContentGenerator()
        
        # Create main output directory
        os.makedirs('training_data', exist_ok=True)
    
    def collect_human_content(self, max_sites: int = None) -> dict:
        """Collect human-written content from the web."""
        
        print(" Collecting Human-Written Content")
        print("=" * 50)
        
        # Get target sites
        sites = get_target_sites()
        if max_sites:
            sites = sites[:max_sites]
        
        print(f" Targeting {len(sites)} websites...")
        
        # Start scraping
        results = self.scraper.scrape_multiple_sites(sites)
        
        # Generate report
        self.scraper.generate_summary_report(results)
        
        return results
    
    def collect_ai_content(self, num_samples: int = 50, target_words: int = 500) -> list:
        """Generate AI-written content samples."""
        
        print(f"\n Generating AI Content Samples")
        print("=" * 50)
        
        # Generate samples
        samples = self.ai_generator.generate_content_samples(num_samples, target_words)
        
        # Save samples
        self.ai_generator.save_samples(samples)
        
        # Generate report
        self.ai_generator.generate_summary_report(samples)
        
        return samples
    
    def create_training_dataset(self, human_results: dict, ai_samples: list) -> dict:
        """Create a comprehensive training dataset."""
        
        print(f"\n Creating Training Dataset")
        print("=" * 50)
        
        # Count human articles
        human_articles = []
        for site_name, articles in human_results.items():
            for article in articles:
                human_articles.append({
                    'source': 'web_scraped',
                    'site': site_name,
                    'url': article['url'],
                    'title': article['title'],
                    'text': article['text'],
                    'word_count': article['word_count'],
                    'label': 'human',
                    'timestamp': article['timestamp']
                })
        
        # Count AI articles
        ai_articles = []
        for sample in ai_samples:
            ai_articles.append({
                'source': 'ai_generated',
                'method': sample['generation_method'],
                'topic': sample['topic'],
                'text': sample['content'],
                'word_count': sample['word_count'],
                'label': 'ai',
                'timestamp': sample['timestamp']
            })
        
        # Create dataset summary
        dataset = {
            'creation_date': datetime.now().isoformat(),
            'total_samples': len(human_articles) + len(ai_articles),
            'human_samples': len(human_articles),
            'ai_samples': len(ai_samples),
            'total_words': sum(article['word_count'] for article in human_articles + ai_articles),
            'human_words': sum(article['word_count'] for article in human_articles),
            'ai_words': sum(article['word_count'] for article in ai_articles),
            'sources': {
                'human': list(set(article['site'] for article in human_articles)),
                'ai': list(set(article['method'] for article in ai_articles))
            }
        }
        
        # Save dataset info
        with open('training_data/dataset_summary.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Save individual samples for easy access
        os.makedirs('training_data/human', exist_ok=True)
        os.makedirs('training_data/ai', exist_ok=True)
        
        # Save human samples
        for i, article in enumerate(human_articles):
            filename = f"training_data/human/human_{i+1:03d}_{article['site']}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Source: {article['url']}\n")
                f.write(f"Site: {article['site']}\n")
                f.write(f"Title: {article['title']}\n")
                f.write(f"Word Count: {article['word_count']}\n")
                f.write(f"Label: {article['label']}\n")
                f.write(f"Timestamp: {article['timestamp']}\n")
                f.write("-" * 80 + "\n\n")
                f.write(article['text'])
        
        # Save AI samples
        for i, article in enumerate(ai_articles):
            filename = f"training_data/ai/ai_{i+1:03d}_{article['topic'].replace(' ', '_')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Source: {article['source']}\n")
                f.write(f"Method: {article['method']}\n")
                f.write(f"Topic: {article['topic']}\n")
                f.write(f"Word Count: {article['word_count']}\n")
                f.write(f"Label: {article['label']}\n")
                f.write(f"Timestamp: {article['timestamp']}\n")
                f.write("-" * 80 + "\n\n")
                f.write(article['text'])
        
        print(f" Dataset created successfully!")
        print(f"  Human samples: {len(human_articles)}")
        print(f"  AI samples: {len(ai_samples)}")
        print(f"  Total samples: {dataset['total_samples']}")
        print(f"  Total words: {dataset['total_words']:,}")
        
        return dataset
    
    def create_auto_labels(self, dataset: dict):
        """Create automatic labels for the dataset."""
        
        print(f"\n️  Creating Automatic Labels")
        print("=" * 50)
        
        # Create labels file
        labels = {}
        
        # Add human samples
        human_files = [f for f in os.listdir('training_data/human') if f.endswith('.txt')]
        for filename in human_files:
            labels[f"training_data/human/{filename}"] = "human"
        
        # Add AI samples
        ai_files = [f for f in os.listdir('training_data/ai') if f.endswith('.txt')]
        for filename in ai_files:
            labels[f"training_data/ai/{filename}"] = "ai"
        
        # Save labels
        with open('training_data/auto_labels.json', 'w') as f:
            json.dump(labels, f, indent=2)
        
        print(f" Auto-labels created for {len(labels)} files")
        print(f"  Human: {len(human_files)}")
        print(f"  AI: {len(ai_files)}")
        print(f"  Labels saved: training_data/auto_labels.json")
    
    def run_full_collection(self, max_sites: int = 10, ai_samples: int = 50, target_words: int = 500):
        """Run the complete data collection process."""
        
        print(" AI Text Detection - Training Data Collection")
        print("=" * 70)
        print(f"This will collect data from ~{max_sites} websites and generate {ai_samples} AI samples.")
        print()
        
        # Confirm before starting
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print(" Collection cancelled.")
            return
        
        start_time = time.time()
        
        try:
            # Step 1: Collect human content
            print("\n" + "="*70)
            human_results = self.collect_human_content(max_sites)
            
            # Step 2: Generate AI content
            print("\n" + "="*70)
            ai_samples_list = self.collect_ai_content(ai_samples, target_words)
            
            # Step 3: Create training dataset
            print("\n" + "="*70)
            dataset = self.create_training_dataset(human_results, ai_samples_list)
            
            # Step 4: Create auto-labels
            print("\n" + "="*70)
            self.create_auto_labels(dataset)
            
            # Final summary
            elapsed_time = time.time() - start_time
            print(f"\n Data Collection Completed Successfully!")
            print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
            print(f" Final dataset: {dataset['total_samples']} samples")
            print(f" Output directory: training_data/")
            
            print(f"\n Next steps:")
            print(f"  1. Review the collected data")
            print(f"  2. Train the ML model: python train_ml_detector.py")
            print(f"  3. Test the detector on your own files")
            
        except Exception as e:
            print(f" Error during collection: {e}")
            print("Check the logs above for details.")

def main():
    """Main function."""
    
    collector = TrainingDataCollector()
    
    print(" Training Data Collection for AI Text Detection")
    print("=" * 70)
    
    # Configuration options
    print("Configuration Options:")
    print("1. Quick collection (5 sites, 25 AI samples)")
    print("2. Standard collection (10 sites, 50 AI samples)")
    print("3. Comprehensive collection (15 sites, 100 AI samples)")
    print("4. Custom configuration")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        collector.run_full_collection(max_sites=5, ai_samples=25, target_words=500)
    elif choice == "2":
        collector.run_full_collection(max_sites=10, ai_samples=50, target_words=500)
    elif choice == "3":
        collector.run_full_collection(max_sites=15, ai_samples=100, target_words=500)
    elif choice == "4":
        max_sites = int(input("Max websites to scrape (default 10): ") or "10")
        ai_samples = int(input("AI samples to generate (default 50): ") or "50")
        target_words = int(input("Target words per sample (default 500): ") or "500")
        collector.run_full_collection(max_sites, ai_samples, target_words)
    else:
        print(" Invalid choice. Using standard configuration.")
        collector.run_full_collection()

if __name__ == "__main__":
    main()
