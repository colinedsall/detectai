from __future__ import annotations

import time
import re
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextDetector:
    def __init__(self) -> None:
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self._min_text_length_for_confidence = 40
        
        # Method weights (can be tuned based on performance)
        self.method_weights = {
            'perplexity': 0.35,
            'watermark': 0.25,
            'semantic_consistency': 0.20,
            'repetition_patterns': 0.20
        }

    def _extract_text_from_input(self, request_data: Dict) -> Tuple[str, str]:
        """Extract text from various input types and return (text, source_type)."""
        start_time = time.time()
        
        # Validate that at least one input method is provided
        if not any([request_data.get('text'), request_data.get('url'), request_data.get('html'), request_data.get('file_path')]):
            raise ValueError("At least one of text, url, html, or file_path must be provided")
        
        if request_data.get('text'):
            return request_data['text'], 'text'
        
        elif request_data.get('file_path'):
            try:
                # Handle local file paths
                file_path = request_data['file_path']
                if not file_path.startswith('/'):
                    # Assume relative to current working directory
                    import os
                    file_path = os.path.join(os.getcwd(), file_path)
                
                # Check if file exists
                if not os.path.exists(file_path):
                    raise ValueError(f"File not found: {file_path}")
                
                # Read text file
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                
                return text, 'file'
                
            except Exception as e:
                raise ValueError(f"Failed to read file: {str(e)}")
        
        elif request_data.get('url'):
            try:
                response = requests.get(str(request_data['url']), timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text from body
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text, 'url'
            except Exception as e:
                raise ValueError(f"Failed to fetch URL: {str(e)}")
        
        elif request_data.get('html'):
            soup = BeautifulSoup(request_data['html'], 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text, 'html'
        
        else:
            raise ValueError("No valid input provided")

    def _calculate_text_metadata(self, text: str, processing_time: float) -> Dict:
        """Calculate basic text statistics and metadata."""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]
        
        return {
            'word_count': len(words),
            'character_count': len(text),
            'unique_words': len(set(words)),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0.0,
            'processing_time_ms': processing_time * 1000,
            'source_type': 'text'  # Will be updated by caller
        }

    def _enhanced_perplexity_analysis(self, text: str) -> Tuple[float, str, List[HighlightSpan]]:
        """Enhanced perplexity-based analysis with better tokenization."""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.5, "No sentences found", []
        
        # Calculate sentence-level perplexity proxies
        sentence_scores = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum()]
            
            if len(words) < 3:
                continue
            
            # Type-token ratio (vocabulary diversity)
            type_token_ratio = len(set(words)) / len(words)
            
            # Average word length
            avg_word_length = np.mean([len(word) for word in words])
            
            # Stop word ratio
            stop_word_ratio = sum(1 for word in words if word in self.stop_words) / len(words)
            
            # Combine metrics into a perplexity proxy
            # Lower values suggest more AI-like text
            perplexity_proxy = (1 - type_token_ratio) * 0.4 + \
                             (1 - stop_word_ratio) * 0.3 + \
                             (avg_word_length / 10) * 0.3
            
            sentence_scores.append(perplexity_proxy)
        
        if not sentence_scores:
            return 0.5, "Insufficient text for analysis", []
        
        avg_perplexity = np.mean(sentence_scores)
        
        # Convert to AI probability (lower perplexity = higher AI probability)
        ai_probability = max(0.0, min(1.0, 1.0 - avg_perplexity))
        
        explanation = f"Perplexity analysis: avg={avg_perplexity:.3f}, sentences={len(sentences)}"
        
        # Highlight sentences with very low perplexity
        highlight_spans = []
        for i, sentence in enumerate(sentences):
            if i < len(sentence_scores) and sentence_scores[i] < 0.3:
                start = text.find(sentence)
                if start != -1:
                    highlight_spans.append({
                        'start': start,
                        'end': start + len(sentence),
                        'reason': f"Low perplexity ({sentence_scores[i]:.3f})",
                        'confidence': 0.8,
                        'method': 'perplexity'
                    })
        
        return ai_probability, explanation, highlight_spans

    def _watermark_detection(self, text: str) -> Tuple[float, str, List[HighlightSpan]]:
        """Statistical watermark detection using pattern analysis."""
        if len(text) < 100:
            return 0.5, "Text too short for watermark analysis", []
        
        # Look for statistical patterns that might indicate watermarks
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]
        
        if len(words) < 20:
            return 0.5, "Insufficient words for watermark analysis", []
        
        # Character frequency analysis
        char_freq = {}
        for char in text.lower():
            if char.isalpha():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Normalize frequencies
        total_chars = sum(char_freq.values())
        if total_chars == 0:
            return 0.5, "No alphabetic characters found", []
        
        char_freq = {k: v/total_chars for k, v in char_freq.items()}
        
        # Check for unusually uniform character distribution
        # AI-generated text often has more uniform character frequencies
        expected_uniform = 1.0 / len(char_freq)
        uniformity_score = 1.0 - np.std(list(char_freq.values()))
        
        # Word length distribution analysis
        word_lengths = [len(word) for word in words]
        word_length_std = np.std(word_lengths)
        word_length_uniformity = 1.0 / (1.0 + word_length_std)
        
        # Combine watermark signals
        watermark_score = (uniformity_score * 0.6 + word_length_uniformity * 0.4)
        
        # Convert to AI probability
        ai_probability = max(0.0, min(1.0, watermark_score))
        
        explanation = f"Watermark analysis: char_uniformity={uniformity_score:.3f}, word_uniformity={word_length_uniformity:.3f}"
        
        # Highlight suspicious patterns
        highlight_spans = []
        if watermark_score > 0.7:
            # Highlight sections with very uniform patterns
            sentences = sent_tokenize(text)
            for sentence in sentences:
                if len(sentence) > 50:  # Only highlight longer sentences
                    start = text.find(sentence)
                    if start != -1:
                        highlight_spans.append({
                            'start': start,
                            'end': start + len(sentence),
                            'reason': f"High watermark score ({watermark_score:.3f})",
                            'confidence': 0.7,
                            'method': 'watermark'
                        })
                        break  # Only highlight first suspicious sentence
        
        return ai_probability, explanation, highlight_spans

    def _semantic_consistency_check(self, text: str) -> Tuple[float, str, List[HighlightSpan]]:
        """Check semantic consistency across text segments."""
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return 0.5, "Insufficient sentences for semantic analysis", []
        
        # Use TF-IDF to measure semantic similarity between sentences
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Clean sentences for vectorization
            clean_sentences = []
            for sentence in sentences:
                clean = re.sub(r'[^\w\s]', '', sentence.lower())
                if len(clean.strip()) > 10:
                    clean_sentences.append(clean)
            
            if len(clean_sentences) < 2:
                return 0.5, "Insufficient clean sentences for semantic analysis", []
            
            tfidf_matrix = vectorizer.fit_transform(clean_sentences)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # Calculate average similarity (excluding self-similarity)
            total_similarity = 0
            count = 0
            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    total_similarity += similarities[i][j]
                    count += 1
            
            if count == 0:
                return 0.5, "Could not calculate semantic similarities", []
            
            avg_similarity = total_similarity / count
            
            # High similarity might indicate repetitive or formulaic text (AI-like)
            # But very low similarity might indicate incoherent text (also AI-like)
            # We'll use a U-shaped function
            if avg_similarity < 0.1 or avg_similarity > 0.8:
                ai_probability = 0.8  # High AI probability for extreme values
            else:
                ai_probability = 0.3  # Lower AI probability for moderate similarity
            
            explanation = f"Semantic consistency: avg_similarity={avg_similarity:.3f}"
            
            # Highlight sentences with extreme similarities
            highlight_spans = []
            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    if similarities[i][j] > 0.9:  # Very high similarity
                        # Highlight both sentences
                        for idx in [i, j]:
                            if idx < len(sentences):
                                sentence = sentences[idx]
                                start = text.find(sentence)
                                if start != -1:
                                    highlight_spans.append({
                                        'start': start,
                                        'end': start + len(sentence),
                                        'reason': f"High semantic similarity ({similarities[i][j]:.3f})",
                                        'confidence': 0.8,
                                        'method': 'semantic_consistency'
                                    })
            
            return ai_probability, explanation, highlight_spans
            
        except Exception as e:
            return 0.5, f"Semantic analysis failed: {str(e)}", []

    def _repetition_pattern_analysis(self, text: str) -> Tuple[float, str, List[HighlightSpan]]:
        """Enhanced repetition pattern analysis."""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]
        
        if len(words) < 10:
            return 0.5, "Insufficient words for repetition analysis", []
        
        # Word repetition analysis
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate repetition metrics
        total_words = len(words)
        unique_words = len(word_counts)
        repetition_ratio = 1.0 - (unique_words / total_words)
        
        # Phrase repetition (bigrams)
        bigrams = []
        for i in range(len(words) - 1):
            bigrams.append((words[i], words[i + 1]))
        
        bigram_counts = {}
        for bigram in bigrams:
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
        
        bigram_repetition = sum(1 for count in bigram_counts.values() if count > 1) / len(bigram_counts) if bigram_counts else 0
        
        # Combine repetition signals
        repetition_score = (repetition_ratio * 0.7 + bigram_repetition * 0.3)
        
        # Convert to AI probability (higher repetition = higher AI probability)
        ai_probability = max(0.0, min(1.0, repetition_score))
        
        explanation = f"Repetition analysis: word_repetition={repetition_ratio:.3f}, bigram_repetition={bigram_repetition:.3f}"
        
        # Highlight repeated words and phrases
        highlight_spans = []
        for word, count in word_counts.items():
            if count >= 3:  # Highlight words that appear 3+ times
                # Find all occurrences
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                for match in pattern.finditer(text):
                    highlight_spans.append({
                        'start': match.start(),
                        'end': match.end(),
                        'reason': f"Repeated {count} times",
                        'confidence': min(0.9, count / 5),  # Higher confidence for more repetitions
                        'method': 'repetition_patterns'
                    })
        
        return ai_probability, explanation, highlight_spans

    def detect(self, request_data: Dict) -> Dict:
        """Main detection method that orchestrates all detection techniques."""
        start_time = time.time()
        
        # Extract text from input
        text, source_type = self._extract_text_from_input(request_data)
        
        # Calculate metadata
        processing_time = time.time() - start_time
        metadata = self._calculate_text_metadata(text, processing_time)
        metadata['source_type'] = source_type
        
        # Run all detection methods
        methods = []
        all_highlight_spans = []
        
        # Perplexity analysis
        ppl_score, ppl_explanation, ppl_highlights = self._enhanced_perplexity_analysis(text)
        methods.append({
            'name': 'perplexity',
            'confidence': ppl_score,
            'explanation': ppl_explanation,
            'weight': self.method_weights['perplexity']
        })
        all_highlight_spans.extend(ppl_highlights)
        
        # Watermark detection
        wm_score, wm_explanation, wm_highlights = self._watermark_detection(text)
        methods.append({
            'name': 'watermark',
            'confidence': wm_score,
            'explanation': wm_explanation,
            'weight': self.method_weights['watermark']
        })
        all_highlight_spans.extend(wm_highlights)
        
        # Semantic consistency
        sem_score, sem_explanation, sem_highlights = self._semantic_consistency_check(text)
        methods.append({
            'name': 'semantic_consistency',
            'confidence': sem_score,
            'explanation': sem_explanation,
            'weight': self.method_weights['semantic_consistency']
        })
        all_highlight_spans.extend(sem_highlights)
        
        # Repetition patterns
        rep_score, rep_explanation, rep_highlights = self._repetition_pattern_analysis(text)
        methods.append({
            'name': 'repetition_patterns',
            'confidence': rep_score,
            'explanation': rep_explanation,
            'weight': self.method_weights['repetition_patterns']
        })
        all_highlight_spans.extend(rep_highlights)
        
        # Calculate weighted final probability
        total_weight = sum(method['weight'] for method in methods)
        weighted_probability = sum(
            method['confidence'] * method['weight'] for method in methods
        ) / total_weight
        
        # Calculate overall confidence based on text length and method agreement
        method_scores = [method['confidence'] for method in methods]
        score_variance = np.var(method_scores)
        length_confidence = min(1.0, len(text) / 1000)  # Higher confidence for longer texts
        
        overall_confidence = (1.0 - score_variance) * 0.7 + length_confidence * 0.3
        
        return {
            'probability_ai': weighted_probability,
            'confidence': overall_confidence,
            'methods': methods,
            'highlight_spans': all_highlight_spans,
            'metadata': metadata
        }

import requests
import json

def analyze_websites(urls):
    """Analyze multiple websites and compare results."""
    
    results = {}
    for url in urls:
        print(f"\n Analyzing: {url}")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/v1/detect/text",
                json={"url": url},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_prob = data.get('probability_ai', 0)
                confidence = data.get('confidence', 0)
                
                print(f"  AI Probability: {ai_prob:.1%}")
                print(f"  Confidence: {confidence:.1%}")
                
                # Show top detection method
                methods = data.get('methods', [])
                if methods:
                    top_method = max(methods, key=lambda x: x['confidence'])
                    print(f"  Top Signal: {top_method['name']} ({top_method['confidence']:.1%})")
                
                results[url] = data
            else:
                print(f"   Error: {response.status_code}")
                
        except Exception as e:
            print(f"   Failed: {str(e)}")
    
    return results


if __name__ == "__main__":
    # Test some websites
    websites = [
        "https://httpbin.org/html",
        "https://example.com",
        "https://news.ycombinator.com",
        "https://blog.hyperwriteai.com/written-by-ai/"
    ]

    analyze_websites(websites)