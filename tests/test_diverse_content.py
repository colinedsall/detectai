#!/usr/bin/env python3
"""
Test script to demonstrate the improved diversity of AI content generation.
Shows examples from different topic categories.
"""

from ai_content_generator import AIContentGenerator

def test_diverse_content():
    """Test the diverse content generation capabilities."""
    
    print(" Testing Diverse AI Content Generation")
    print("=" * 60)
    
    # Create generator
    generator = AIContentGenerator()
    
    # Test different topic categories
    test_topics = [
        "the meaning of life and purpose",
        "quick and easy dinner recipes", 
        "productivity and time management",
        "daily life experiences",
        "entrepreneurship and startups"
    ]
    
    print(" Testing 5 diverse topics with 200-word samples:")
    print()
    
    for i, topic in enumerate(test_topics, 1):
        print(f" Topic {i}: {topic}")
        print("-" * 50)
        
        # Generate content
        content = generator.generate_fallback_text(topic, target_words=200)
        
        # Show first 150 characters
        preview = content[:150] + "..." if len(content) > 150 else content
        print(f"Content: {preview}")
        print(f"Word count: {len(content.split())}")
        print()
    
    print(" Now let's generate a full dataset:")
    print()
    
    # Generate multiple samples
    samples = generator.generate_content_samples(num_samples=10, target_words=300)
    
    print(f" Generated {len(samples)} diverse samples")
    print(f" Topics covered: {len(set(sample['topic'] for sample in samples))}")
    print(f" Total words: {sum(sample['word_count'] for sample in samples):,}")
    
    # Show topic distribution
    topic_counts = {}
    for sample in samples:
        topic = sample['topic']
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print(f"\n Topic Distribution:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} samples")

def main():
    """Main test function."""
    
    print(" AI Content Generator - Diversity Test")
    print("=" * 60)
    
    try:
        test_diverse_content()
        
        print(f"\n Diversity test completed!")
        print(f"\n The generator now covers:")
        print(f"   Philosophy & Life (meaning of life, happiness, ethics)")
        print(f"   Food & Recipes (cooking, meal prep, international cuisine)")
        print(f"   Lifestyle & Personal Development (productivity, wellness)")
        print(f"   Common Blog Topics (parenting, reviews, fashion)")
        print(f"   Business & Professional (entrepreneurship, leadership)")
        print(f"   Technology & Science (AI, data science, computing)")
        
        print(f"\n Ready to collect training data? Run:")
        print(f"  python collect_training_data.py")
        
    except Exception as e:
        print(f" Test failed: {e}")

if __name__ == "__main__":
    main()
