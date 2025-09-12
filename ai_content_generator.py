#!/usr/bin/env python3
"""
AI Content Generator for training data.
Uses free APIs to generate AI-written content samples.
"""

import requests
import time
import random
import json
import os
from typing import List, Dict
from datetime import datetime

class AIContentGenerator:
    def __init__(self):
        """Initialize the AI content generator."""
        
        # Create output directory
        os.makedirs('scraped_data/ai', exist_ok=True)
        
        # Free AI text generation APIs
        self.apis = [
            {
                'name': 'huggingface',
                'url': 'https://api-inference.huggingface.co/models/gpt2',
                'headers': {'Authorization': 'Bearer hf_xxx'},  # You'll need to get a free token
                'method': 'post',
                'data_key': 'inputs',
                'response_key': 'generated_text'
            },
            {
                'name': 'deepai',
                'url': 'https://api.deepai.org/api/text-generator',
                'headers': {'api-key': 'your_api_key'},  # Free tier available
                'method': 'post',
                'data_key': 'text',
                'response_key': 'output'
            }
        ]
        
        # Fallback: Use predefined AI-like text patterns for diverse topics
        self.fallback_patterns = [
            # Technical/Professional patterns
            "The implementation of machine learning algorithms requires careful consideration of several key factors. First, data preprocessing is essential for optimal performance. Second, feature engineering plays a crucial role in model accuracy. Third, hyperparameter tuning is necessary for achieving the best results.",
            
            "Artificial intelligence has revolutionized many industries in recent years. Machine learning algorithms can now process vast amounts of data and identify patterns that were previously impossible to detect. This technology continues to evolve rapidly, offering new possibilities for automation and decision-making.",
            
            "The development of neural networks has led to significant breakthroughs in computer vision and natural language processing. These models can learn complex representations from large datasets and generalize well to unseen examples. The architecture typically consists of multiple layers with non-linear activation functions.",
            
            # Lifestyle/Personal development patterns
            "When it comes to achieving personal goals, there are several important principles to consider. First, setting clear and specific objectives is essential for success. Second, developing consistent habits and routines plays a crucial role in progress. Third, maintaining motivation and accountability is necessary for long-term achievement.",
            
            "The journey of self-improvement involves multiple interconnected aspects of life. Personal development encompasses physical health, mental well-being, emotional intelligence, and professional growth. Each area requires dedicated attention and consistent effort to see meaningful results.",
            
            # Food/Cooking patterns
            "Creating delicious and nutritious meals involves understanding several fundamental cooking principles. First, selecting high-quality ingredients is essential for optimal flavor. Second, proper cooking techniques play a crucial role in texture and taste. Third, seasoning and timing are necessary for achieving the perfect balance.",
            
            "The art of cooking combines creativity with scientific precision. Understanding ingredient interactions, temperature control, and flavor profiles allows chefs to create memorable dining experiences. Each recipe represents a balance of tradition and innovation.",
            
            # Philosophy/Life patterns
            "Exploring the deeper questions of existence requires thoughtful consideration of multiple perspectives. First, understanding different philosophical traditions is essential for comprehensive analysis. Second, examining personal experiences plays a crucial role in forming beliefs. Third, engaging with diverse viewpoints is necessary for intellectual growth.",
            
            "The search for meaning in life involves examining various aspects of human experience. Personal fulfillment comes from understanding one's values, building meaningful relationships, and contributing to something larger than oneself. This journey requires both introspection and action.",
            
            # Business/Professional patterns
            "Successful business strategies require careful analysis of several key factors. First, understanding market dynamics is essential for competitive positioning. Second, developing strong customer relationships plays a crucial role in long-term success. Third, maintaining operational efficiency is necessary for sustainable growth.",
            
            "Leadership in modern organizations involves balancing multiple competing priorities. Effective managers must develop clear communication skills, foster team collaboration, and maintain strategic focus. The ability to adapt to changing circumstances is crucial for continued success."
        ]
    
    def generate_with_huggingface(self, prompt: str, max_length: int = 200) -> str:
        """Generate text using Hugging Face's free API."""
        
        try:
            # Note: You need to get a free API token from huggingface.co
            # For now, we'll use fallback patterns
            return self.generate_fallback_text(prompt, max_length)
            
        except Exception as e:
            print(f"❌ Hugging Face API error: {e}")
            return self.generate_fallback_text(prompt, max_length)
    
    def generate_fallback_text(self, prompt: str, target_words: int = 500) -> str:
        """Generate AI-like text using predefined patterns and variations."""
        
        # Start with the prompt
        text = prompt + " "
        
        # Add variations of our patterns
        patterns = self.fallback_patterns.copy()
        random.shuffle(patterns)
        
        while len(text.split()) < target_words:
            # Pick a random pattern
            pattern = random.choice(patterns)
            
            # Create variations for diverse topics
            variations = [
                # Technical variations
                pattern.replace("machine learning", "deep learning"),
                pattern.replace("algorithms", "models"),
                pattern.replace("data", "information"),
                pattern.replace("performance", "efficiency"),
                pattern.replace("accuracy", "precision"),
                pattern.replace("neural networks", "artificial neural networks"),
                pattern.replace("computer vision", "image recognition"),
                pattern.replace("natural language processing", "text analysis"),
                pattern.replace("cloud computing", "distributed computing"),
                pattern.replace("IT infrastructure", "technology infrastructure"),
                
                # Lifestyle variations
                pattern.replace("personal goals", "life objectives"),
                pattern.replace("habits and routines", "daily practices"),
                pattern.replace("motivation and accountability", "inspiration and responsibility"),
                pattern.replace("self-improvement", "personal growth"),
                pattern.replace("physical health", "wellness"),
                pattern.replace("mental well-being", "psychological health"),
                
                # Food variations
                pattern.replace("cooking principles", "culinary techniques"),
                pattern.replace("high-quality ingredients", "fresh ingredients"),
                pattern.replace("cooking techniques", "culinary methods"),
                pattern.replace("flavor profiles", "taste combinations"),
                pattern.replace("ingredient interactions", "food chemistry"),
                
                # Philosophy variations
                pattern.replace("philosophical traditions", "intellectual frameworks"),
                pattern.replace("personal experiences", "individual journeys"),
                pattern.replace("intellectual growth", "mental development"),
                pattern.replace("deeper questions", "fundamental inquiries"),
                pattern.replace("human experience", "life experience"),
                
                # Business variations
                pattern.replace("business strategies", "organizational approaches"),
                pattern.replace("market dynamics", "industry trends"),
                pattern.replace("customer relationships", "client partnerships"),
                pattern.replace("operational efficiency", "business performance"),
                pattern.replace("leadership", "management")
            ]
            
            # Add a variation
            text += random.choice(variations) + " "
            
            # Add diverse filler sentences for different topics
            fillers = [
                # Technical/Research fillers
                "Furthermore, this approach demonstrates significant improvements over traditional methods.",
                "Additionally, the results indicate a clear advantage in terms of computational efficiency.",
                "Moreover, the analysis reveals important insights into the underlying mechanisms.",
                "In conclusion, this methodology provides a robust foundation for future research.",
                "It is important to note that these findings have broader implications for the field.",
                
                # Lifestyle/Personal development fillers
                "Furthermore, this approach demonstrates significant improvements in personal well-being.",
                "Additionally, the results indicate a clear advantage in terms of life satisfaction.",
                "Moreover, the analysis reveals important insights into human behavior patterns.",
                "In conclusion, this methodology provides a robust foundation for personal growth.",
                "It is important to note that these findings have broader implications for daily life.",
                
                # Food/Cooking fillers
                "Furthermore, this approach demonstrates significant improvements in culinary outcomes.",
                "Additionally, the results indicate a clear advantage in terms of flavor development.",
                "Moreover, the analysis reveals important insights into cooking techniques.",
                "In conclusion, this methodology provides a robust foundation for culinary excellence.",
                "It is important to note that these findings have broader implications for food preparation.",
                
                # Philosophy/Life fillers
                "Furthermore, this approach demonstrates significant improvements in understanding.",
                "Additionally, the results indicate a clear advantage in terms of perspective.",
                "Moreover, the analysis reveals important insights into human nature.",
                "In conclusion, this methodology provides a robust foundation for wisdom.",
                "It is important to note that these findings have broader implications for existence.",
                
                # Business/Professional fillers
                "Furthermore, this approach demonstrates significant improvements in organizational performance.",
                "Additionally, the results indicate a clear advantage in terms of business outcomes.",
                "Moreover, the analysis reveals important insights into market dynamics.",
                "In conclusion, this methodology provides a robust foundation for business success.",
                "It is important to note that these findings have broader implications for industry practices."
            ]
            
            text += random.choice(fillers) + " "
        
        # Clean up and return
        text = text.strip()
        words = text.split()
        if len(words) > target_words:
            text = ' '.join(words[:target_words])
        
        return text
    
    def generate_content_samples(self, num_samples: int = 50, target_words: int = 500) -> List[Dict]:
        """Generate multiple AI content samples."""
        
        print(f"🤖 Generating {num_samples} AI content samples...")
        
        samples = []
        
        # Topics for AI generation - diverse real-world content
        topics = [
            # Technology & Science
            "machine learning algorithms",
            "artificial intelligence applications",
            "data science methodologies",
            "neural network architectures",
            "cloud computing solutions",
            "cybersecurity best practices",
            "blockchain technology",
            "internet of things",
            "quantum computing",
            "robotics and automation",
            "natural language processing",
            "computer vision systems",
            "big data analytics",
            "software development practices",
            "database management systems",
            
            # Philosophy & Life
            "the meaning of life and purpose",
            "happiness and personal fulfillment",
            "ethics and moral philosophy",
            "consciousness and the mind",
            "free will versus determinism",
            "the nature of reality",
            "spirituality and religion",
            "death and mortality",
            "love and relationships",
            "success and achievement",
            
            # Food & Recipes
            "quick and easy dinner recipes",
            "healthy breakfast ideas",
            "vegetarian cooking techniques",
            "baking bread from scratch",
            "meal prep strategies",
            "international cuisine recipes",
            "dessert and pastry making",
            "cooking for beginners",
            "seasonal ingredient cooking",
            "dietary restriction recipes",
            
            # Lifestyle & Personal Development
            "productivity and time management",
            "work-life balance strategies",
            "stress management techniques",
            "mindfulness and meditation",
            "personal finance tips",
            "career development advice",
            "health and wellness tips",
            "travel planning and tips",
            "home organization methods",
            "self-improvement strategies",
            
            # Common Blog Topics
            "daily life experiences",
            "parenting advice and tips",
            "book reviews and recommendations",
            "movie and entertainment reviews",
            "fashion and style advice",
            "beauty and skincare tips",
            "fitness and exercise routines",
            "mental health awareness",
            "environmental sustainability",
            "social media and digital life",
            
            # Business & Professional
            "entrepreneurship and startups",
            "marketing strategies",
            "leadership and management",
            "remote work best practices",
            "networking and relationship building",
            "public speaking skills",
            "negotiation techniques",
            "project management tips",
            "customer service excellence",
            "innovation and creativity"
        ]
        
        for i in range(num_samples):
            print(f"  🔄 Generating sample {i+1}/{num_samples}...")
            
            # Pick a random topic
            topic = random.choice(topics)
            
            # Generate content
            content = self.generate_fallback_text(topic, target_words)
            
            # Create sample
            sample = {
                'id': i + 1,
                'topic': topic,
                'content': content,
                'word_count': len(content.split()),
                'generation_method': 'fallback_patterns',
                'timestamp': datetime.now().isoformat()
            }
            
            samples.append(sample)
            
            # Small delay
            time.sleep(0.1)
        
        return samples
    
    def save_samples(self, samples: List[Dict]):
        """Save generated samples to files."""
        
        print(f"💾 Saving {len(samples)} AI content samples...")
        
        for sample in samples:
            # Create filename
            safe_topic = sample['topic'].replace(' ', '_').replace('-', '_')
            filename = f"scraped_data/ai/ai_generated_{sample['id']:03d}_{safe_topic}.txt"
            
            # Save sample
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"AI Generated Content Sample #{sample['id']}\n")
                f.write(f"Topic: {sample['topic']}\n")
                f.write(f"Word Count: {sample['word_count']}\n")
                f.write(f"Generation Method: {sample['generation_method']}\n")
                f.write(f"Timestamp: {sample['timestamp']}\n")
                f.write("-" * 80 + "\n\n")
                f.write(sample['content'])
            
            print(f"    💾 Saved: {filename}")
    
    def generate_summary_report(self, samples: List[Dict]):
        """Generate a summary report of generated content."""
        
        total_words = sum(sample['word_count'] for sample in samples)
        avg_words = total_words / len(samples) if samples else 0
        
        report = {
            'generation_date': datetime.now().isoformat(),
            'total_samples': len(samples),
            'total_words': total_words,
            'average_words_per_sample': avg_words,
            'topics_covered': list(set(sample['topic'] for sample in samples)),
            'generation_method': 'fallback_patterns'
        }
        
        # Save report
        with open('scraped_data/ai_generation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 AI Generation Summary:")
        print(f"  Samples generated: {len(samples)}")
        print(f"  Total words: {total_words:,}")
        print(f"  Average words per sample: {avg_words:.1f}")
        print(f"  Report saved: scraped_data/ai_generation_report.json")

def main():
    """Main AI content generation function."""
    
    print("🤖 AI Content Generator for Training Data")
    print("=" * 60)
    
    # Create generator
    generator = AIContentGenerator()
    
    # Generate samples
    num_samples = int(input("How many AI samples to generate? (default 50): ") or "50")
    target_words = int(input("Target words per sample? (default 500): ") or "500")
    
    print(f"\n🎯 Generating {num_samples} samples with ~{target_words} words each...")
    
    # Generate content
    samples = generator.generate_content_samples(num_samples, target_words)
    
    # Save samples
    generator.save_samples(samples)
    
    # Generate report
    generator.generate_summary_report(samples)
    
    print(f"\n✅ AI content generation completed!")
    print(f"📁 Check the 'scraped_data/ai/' directory for generated samples.")
    print(f"\n💡 Next steps:")
    print(f"  1. Run web scraper: python web_scraper.py")
    print(f"  2. Label your files: python label_files.py")
    print(f"  3. Train the model: python train_ml_detector.py")

if __name__ == "__main__":
    main()
