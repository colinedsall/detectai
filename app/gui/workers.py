"""
QThread workers for long-running operations.
"""
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import os

# Add project root and scripts directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


class DataCollectionWorker(QThread):
    """Worker thread for collecting training data."""
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(bool, str)  # success, message
    log = pyqtSignal(str)  # log message
    
    def __init__(self, collect_human=True, collect_ai=True, ai_model="gpt-oss:20b", ai_samples=100):
        super().__init__()
        self.collect_human = collect_human
        self.collect_ai = collect_ai
        self.ai_model = ai_model
        self.ai_samples = ai_samples
        self._is_cancelled = False
    
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        try:
            from collect_training_data import TrainingDataCollector
            from ai_content_generator import AIContentGenerator
            
            collector = TrainingDataCollector()
            
            if self.collect_human:
                self.log.emit("Starting human content collection...")
                self.progress.emit(10, "Collecting human content from web...")
                
                from web_scraper import WebScraper, get_target_sites, load_config, get_sites_from_config
                scraper = WebScraper()
                
                # Load sites from config.yaml first, fallback to hardcoded list
                cfg = load_config()
                sites = get_sites_from_config(cfg)
                if not sites:
                    self.log.emit("No sites in config, using defaults...")
                    sites = get_target_sites()
                else:
                    self.log.emit(f"Loaded {len(sites)} sites from config.yaml")
                
                total_articles = 0
                for i, site in enumerate(sites):
                    if self._is_cancelled:
                        self.finished.emit(False, "Cancelled by user")
                        return
                    
                    site_name = site.get('name', site.get('url', 'Unknown'))
                    self.log.emit(f"Scraping: {site_name}")
                    percent = 10 + int((i / len(sites)) * 40)
                    self.progress.emit(percent, f"Scraping site {i+1}/{len(sites)}: {site_name}")
                    
                    # Actually scrape the website
                    try:
                        articles = scraper.scrape_website(
                            site['url'], 
                            max_pages=site.get('max_pages', 5),
                            min_words=300
                        )
                        if articles:
                            scraper.save_articles(articles, site_name)
                            total_articles += len(articles)
                            self.log.emit(f"  → Collected {len(articles)} articles from {site_name}")
                    except Exception as e:
                        self.log.emit(f"  → Error scraping {site_name}: {e}")
                
                self.log.emit(f"Human collection complete: {total_articles} total articles")
                self.progress.emit(50, "Human content collection complete")
            
            if self.collect_ai:
                self.log.emit("Starting AI content generation with Ollama...")
                self.progress.emit(55, "Generating AI content with Ollama...")
                
                generator = AIContentGenerator()
                # Use stored number of samples
                num_samples = self.ai_samples
                
                # Ensure output directory exists
                import os
                output_dir = os.path.join(generator.project_root, 'training_data', 'ai')
                os.makedirs(output_dir, exist_ok=True)
                self.log.emit(f"AI Output Dir: {output_dir}")
                
                # Use topics from generator (loaded from config)
                topics = generator.topics
                if not topics:
                     topics = ["machine learning", "technology", "science"] # Ultimate fallback
                     self.log.emit("Warning: No topics found in config, using defaults")
                
                self.log.emit(f"Generating content for {len(topics)} topics")
                
                for i in range(num_samples):
                    if self._is_cancelled:
                        self.finished.emit(False, "Cancelled by user")
                        return
                    
                    try:
                        topic = topics[i % len(topics)]
                        prompt = f"Write a detailed 500-word article about {topic}. Include specific examples and actionable advice."
                        
                        # Get target length from config
                        target_words = 700
                        if hasattr(generator, 'config'):
                            target_words = generator.config.get('ai_generation', {}).get('target_words_per_sample') or \
                                           generator.config.get('training', {}).get('target_words_per_sample') or 700
                                           
                        # Generate with Ollama
                        content = generator.generate_with_ollama(
                            prompt=prompt,
                            max_tokens=int(target_words),
                            model=self.ai_model
                        )
                        
                        if content and len(content.split()) > 100:
                            # Save to training_data/ai/
                            from datetime import datetime
                            import re
                            
                            # Create safe filename with timestamp to prevent overwriting
                            safe_topic = re.sub(r'[^a-zA-Z0-9]', '_', topic)[:50]
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(output_dir, f"ai_{timestamp}_{i+1}_{safe_topic}.txt")
                            
                            with open(filename, 'w', encoding='utf-8') as f:
                                # Prevent model from learning from header to distinguish between AI and human writing
                                f.write(content)
                            
                            self.log.emit(f"Generated AI sample {i+1}/{num_samples}: {topic}")
                            self.log.emit(f"  -> Saved {os.path.basename(filename)}")
                        else:
                            self.log.emit(f"AI sample {i+1} too short, skipping")
                            
                    except Exception as e:
                        self.log.emit(f"AI generation error: {e}")
                    
                    percent = 55 + int((i / num_samples) * 40)
                    self.progress.emit(percent, f"Generating AI sample {i+1}/{num_samples}")
                
                self.progress.emit(95, "AI content generation complete")
            
            self.progress.emit(100, "Data collection complete!")
            self.finished.emit(True, "Data collection completed successfully")
            
        except Exception as e:
            self.log.emit(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.finished.emit(False, f"Error: {str(e)}")


class TrainingWorker(QThread):
    """Worker thread for model training."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, dict)  # success, message, results
    log = pyqtSignal(str)
    
    def __init__(self, model_type="ensemble", epochs=50, learning_rate=0.001, batch_size=32):
        super().__init__()
        self.model_type = model_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self._is_cancelled = False
    
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        try:
            self.log.emit("Loading training data...")
            self.progress.emit(10, "Loading training data...")
            
            from train_with_collected_data import load_collected_data
            human_texts, ai_texts = load_collected_data()
            
            if not human_texts or not ai_texts:
                self.finished.emit(False, "No training data found", {})
                return
            
            self.log.emit(f"Found {len(human_texts)} human and {len(ai_texts)} AI samples")
            self.progress.emit(30, "Initializing model...")
            
            # Choose detector based on model type
            if self.model_type == "neural_network":
                self.log.emit("Using Neural Network (PyTorch MLP)...")
                from app.services.nn_text_detector import NeuralNetworkDetector
                detector = NeuralNetworkDetector()
                
                self.log.emit("Training neural network (this may take a few minutes)...")
                self.progress.emit(40, "Training neural network...")
                
                # Progress callback for epoch updates
                def epoch_callback(epoch, train_loss, val_loss, train_acc, val_acc):
                    percent = 40 + int((epoch / 50) * 55)  # Scale to 40-95%
                    self.log.emit(f"Epoch {epoch}: loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                    self.progress.emit(min(percent, 95), f"Epoch {epoch}")
                
                metrics = detector.train_model(
                    ai_texts=ai_texts,
                    human_texts=human_texts,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    progress_callback=epoch_callback
                )
            else:
                from app.services.ml_text_detector import MLTextDetector
                detector = MLTextDetector()
                
                self.log.emit(f"Training {self.model_type} model...")
                self.progress.emit(50, "Training model...")
                
                metrics = detector.train_model(
                    ai_texts=ai_texts,
                    human_texts=human_texts,
                    model_type=self.model_type,
                    test_size=0.2
                )
            
            self.progress.emit(100, "Training complete!")
            
            # Add sample counts to metrics
            metrics['human_samples'] = len(human_texts)
            metrics['ai_samples'] = len(ai_texts)
            metrics['model_type'] = self.model_type
            
            accuracy = metrics.get('accuracy', 0)
            auc = metrics.get('auc_score', 0)
            
            self.finished.emit(True, f"Training complete! Accuracy: {accuracy:.1%}, AUC: {auc:.3f}", metrics)
            
        except Exception as e:
            self.log.emit(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.finished.emit(False, f"Error: {str(e)}", {})

