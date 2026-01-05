from __future__ import annotations

import time
import re
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

class MLTextDetector:
    def __init__(self, model_path: str = "ai_detector_model.pkl"):
        """Initialize the ML-based text detector."""
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.feature_scaler = None
        self.is_trained = False
        self.calibration_model = None
        self.confidence_thresholds = None
        
        # Load existing model if available
        self.load_model()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for ML model."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize and lemmatize
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]
        
        features = {}
        
        # Basic statistics
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = len(words) / max(1, len(sentences))
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Vocabulary diversity
        unique_words = set(words)
        features['type_token_ratio'] = len(unique_words) / max(1, len(words))
        features['unique_word_ratio'] = len(unique_words) / max(1, len(words))
        
        # Stop word ratio
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        features['stop_word_ratio'] = stop_word_count / max(1, len(words))
        
        # Repetition patterns
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        features['repetition_ratio'] = repeated_words / max(1, len(word_counts))
        
        # Character frequency uniformity
        char_freq = {}
        for char in text.lower():
            if char.isalpha():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        if char_freq:
            total_chars = sum(char_freq.values())
            char_freq = {k: v/total_chars for k, v in char_freq.items()}
            features['char_uniformity'] = 1.0 - np.std(list(char_freq.values()))
        else:
            features['char_uniformity'] = 0.0
        
        # Additional features for better confidence estimation
        features['sentence_length_std'] = np.std([len(sent.split()) for sent in sentences]) if len(sentences) > 1 else 0
        features['word_length_std'] = np.std([len(word) for word in words]) if words else 0
        features['punctuation_ratio'] = len(re.findall(r'[^\w\s]', text)) / max(1, len(text))
        
        return features
    
    def create_training_data(self, ai_texts: List[str], human_texts: List[str]) -> Tuple[List[str], List[int], List[Dict]]:
        """Create training data from labeled examples."""
        texts = []
        labels = []  # 1 for AI, 0 for human
        feature_vectors = []
        
        # Add AI texts
        for text in ai_texts:
            texts.append(self.preprocess_text(text))
            labels.append(1)
            feature_vectors.append(self.extract_features(text))
        
        # Add human texts
        for text in human_texts:
            texts.append(self.preprocess_text(text))
            labels.append(0)
            feature_vectors.append(self.extract_features(text))
        
        return texts, labels, feature_vectors
    
    def estimate_confidence(self, prob_ai: float, features: Dict[str, float], text_length: int) -> float:
        """Estimate confidence based on probability, features, and text characteristics."""
        
        # Base confidence from probability distance from decision boundary
        prob_distance = abs(prob_ai - 0.5)
        base_confidence = min(0.95, 0.5 + prob_distance)
        
        # Length-based confidence (longer texts = more confident)
        length_confidence = min(0.2, text_length / 1000)
        
        # Feature-based confidence adjustments
        feature_confidence = 0.0
        
        # High vocabulary diversity suggests more confident prediction
        if features['type_token_ratio'] > 0.7:
            feature_confidence += 0.1
        elif features['type_token_ratio'] < 0.3:
            feature_confidence -= 0.05
        
        # Very uniform character distribution suggests AI
        if features['char_uniformity'] > 0.9:
            feature_confidence += 0.05
        
        # High repetition suggests AI
        if features['repetition_ratio'] > 0.5:
            feature_confidence += 0.05
        
        # Sentence length variation suggests human
        if features['sentence_length_std'] > 5:
            feature_confidence += 0.05
        
        # Combine all confidence factors
        total_confidence = base_confidence + length_confidence + feature_confidence
        
        # Clamp to reasonable range
        return max(0.3, min(0.95, total_confidence))
    
    def train_model(self, ai_texts: List[str], human_texts: List[str], 
                   model_type: str = "ensemble", test_size: float = 0.2):
        """Train the ML model on labeled data with improved architecture."""
        
        print(" Training ML Text Detector...")
        print(f" AI texts: {len(ai_texts)}")
        print(f" Human texts: {len(human_texts)}")
        
        # Create training data
        texts, labels, feature_vectors = self.create_training_data(ai_texts, human_texts)
        
        # Split into train/test
        labels_array = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_array, test_size=test_size, random_state=42, stratify=labels_array
        )
        
        # Create feature scaler for linguistic features
        feature_names = list(feature_vectors[0].keys())
        feature_matrix = np.array([[fv[f] for f in feature_names] for fv in feature_vectors])
        
        # Split feature matrix the same way as texts
        f_train, f_test, _, _ = train_test_split(
            feature_matrix, labels_array, test_size=test_size, random_state=42, stratify=labels_array
        )
        
        self.feature_scaler = StandardScaler()
        f_train_scaled = self.feature_scaler.fit_transform(f_train)
        f_test_scaled = self.feature_scaler.transform(f_test)
        
        # Create improved model architecture
        if model_type == "ensemble":
            # Use Random Forest as the ensemble model (simpler approach)
            self.model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            )
            
            # TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=3,
                max_df=0.9,
                sublinear_tf=True
            )

            # 1. Create TF-IDF features
            print(" Creating TF-IDF features...")
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)

            # 2. Combine TF-IDF features with your linguistic features
            print(" Combining TF-IDF and linguistic features...")
            X_train_combined = hstack([X_train_tfidf, f_train_scaled])
            X_test_combined = hstack([X_test_tfidf, f_test_scaled])

            # Use Random Forest as the primary model
            self.model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            )

            # 3. Train the model on the COMBINED features
            print(" Training Random Forest model on combined features...")
            self.model.fit(X_train_combined, y_train)

            # 4. Evaluate on the COMBINED test features
            y_pred = self.model.predict(X_test_combined)
            y_pred_proba = self.model.predict_proba(X_test_combined)
            
        elif model_type == "random_forest":
            classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=3,
                max_df=0.9
            )
            
            self.model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', classifier)
            ])
            
            print(" Training Random Forest model...")
            self.model.fit(X_train, y_train)
            
        elif model_type == "logistic_regression":
            classifier = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=3,
                max_df=0.9
            )
            
            self.model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', classifier)
            ])
            
            print(" Training Logistic Regression model...")
            self.model.fit(X_train, y_train)
        
        # Evaluate
        if model_type == "ensemble":
            y_pred = self.model.predict(X_test_combined)
            y_pred_proba = self.model.predict_proba(X_test_combined)
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        print(f" Training completed!")
        print(f" Test Accuracy: {accuracy:.3f}")
        print(f" AUC Score: {auc_score:.3f}")
        print(f" Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
        
        # Cross-validation on training data (commented out for now due to data shape issues)
        # if model_type == "ensemble":
        #     # Use the base ensemble for cross-validation (before calibration)
        #     base_ensemble = VotingClassifier(
        #         estimators=[('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        #                    ('lr', LogisticRegression(random_state=42, max_iter=1000, C=1.0))],
        #         voting='soft'
        #     )
        #     cv_scores = cross_val_score(base_ensemble, X_train_tfidf, y_train, cv=5, scoring='accuracy')
        # else:
        #     cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        # print(f" Cross-validation scores: {cv_scores}")
        # print(f" Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f" Cross-validation skipped due to data shape compatibility")
        
        # Calculate confidence thresholds based on validation performance
        self.confidence_thresholds = self._calculate_confidence_thresholds(y_test, y_pred_proba[:, 1])
        
        self.is_trained = True
        
        # Save the model
        self.save_model()
        
        # Build detailed metrics for visualization
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
        
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba[:, 1])
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
        
        # Feature importance (for ensemble/random_forest)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_.tolist()
        elif hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps.get('classifier', None), 'feature_importances_'):
            feature_importance = self.model.named_steps['classifier'].feature_importances_.tolist()
        
        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'precision_recall': {'precision': precision.tolist(), 'recall': recall.tolist()},
            'feature_importance': feature_importance,
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba[:, 1].tolist()
        }
        
        return metrics
    
    def _calculate_confidence_thresholds(self, y_true, y_pred_proba):
        """Calculate confidence thresholds based on validation performance."""
        # Find probability thresholds that give different confidence levels
        thresholds = {}
        
        # High confidence threshold (90% precision)
        for threshold in np.arange(0.5, 0.95, 0.01):
            high_conf_pred = (y_pred_proba > threshold).astype(int)
            if np.sum(high_conf_pred) > 0:
                precision = np.sum((high_conf_pred == 1) & (y_true == 1)) / np.sum(high_conf_pred)
                if precision >= 0.9:
                    thresholds['high'] = threshold
                    break
        
        # Medium confidence threshold (80% precision)
        for threshold in np.arange(0.5, 0.95, 0.01):
            med_conf_pred = (y_pred_proba > threshold).astype(int)
            if np.sum(med_conf_pred) > 0:
                precision = np.sum((med_conf_pred == 1) & (y_true == 1)) / np.sum(med_conf_pred)
                if precision >= 0.8:
                    thresholds['medium'] = threshold
                    break
        
        return thresholds
    
    def predict(self, text: str) -> Dict[str, any]:
        """Predict whether text is AI-generated with improved confidence estimation."""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
    
        # 1. Preprocess the text for TF-IDF
        processed_text = self.preprocess_text(text)
        
        # 2. Extract the 12 linguistic features from the original text
        linguistic_features = self.extract_features(text)
        
        # Create a numpy array in the correct order
        feature_names = list(linguistic_features.keys())
        feature_vector = np.array([linguistic_features[f] for f in feature_names]).reshape(1, -1)
        
        # 3. Scale the linguistic features using the FITTED scaler
        scaled_feature_vector = self.feature_scaler.transform(feature_vector)
        
        # 4. Create TF-IDF features using the FITTED vectorizer
        tfidf_vector = self.vectorizer.transform([processed_text])
        
        # 5. Combine the features exactly like in training
        combined_features = hstack([tfidf_vector, scaled_feature_vector])
        
        # 6. Get the prediction probability from the COMBINED features
        prob_ai = self.model.predict_proba(combined_features)[0][1]
        
        # Estimate confidence using the original (unscaled) features
        confidence = self.estimate_confidence(prob_ai, linguistic_features, len(text))
        
        # Create explanation
        explanations = []
        if linguistic_features['type_token_ratio'] < 0.6:
            explanations.append("Low vocabulary diversity")
        if linguistic_features['repetition_ratio'] > 0.3:
            explanations.append("High word repetition")
        if linguistic_features['char_uniformity'] > 0.8:
            explanations.append("Very uniform character distribution")
        if linguistic_features['avg_sentence_length'] > 25:
            explanations.append("Long, complex sentences")
        
        # Add confidence-based explanations
        if confidence < 0.6:
            explanations.append("Low confidence prediction")
        elif confidence > 0.9:
            explanations.append("High confidence prediction")
        
        return {
            'probability_ai': float(prob_ai),
            'confidence': float(confidence),
            'prediction': 'AI' if prob_ai > 0.5 else 'Human',
            'explanations': explanations,
            'features': linguistic_features,
            'method': 'ml_classifier'
        }
    
    def save_model(self):
        """Save the trained model to disk."""
        if self.model is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'feature_scaler': self.feature_scaler,
                    'calibration_model': self.calibration_model,
                    'confidence_thresholds': self.confidence_thresholds,
                    'is_trained': self.is_trained
                }, f)
            print(f" Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.vectorizer = data.get('vectorizer')
                    self.feature_scaler = data.get('feature_scaler')
                    self.calibration_model = data.get('calibration_model')
                    self.confidence_thresholds = data.get('confidence_thresholds')
                    self.is_trained = data['is_trained']
                print(f" Model loaded from {self.model_path}")
            except Exception as e:
                print(f" Failed to load model: {e}")
                self.model = None
                self.is_trained = False
    
    def evaluate_on_files(self, ai_files: List[str], human_files: List[str]):
        """Evaluate model performance on test files."""
        
        if not self.is_trained:
            print(" Model not trained. Please train first.")
            return
        
        print(" Evaluating model on test files...")
        
        # Load and predict on AI files
        ai_predictions = []
        for file_path in ai_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                pred = self.predict(text)
                ai_predictions.append(pred['probability_ai'])
                print(f" {file_path}: {pred['probability_ai']:.3f}")
            except Exception as e:
                print(f" Error reading {file_path}: {e}")
        
        # Load and predict on human files
        human_predictions = []
        for file_path in human_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                pred = self.predict(text)
                human_predictions.append(pred['probability_ai'])
                print(f" {file_path}: {pred['probability_ai']:.3f}")
            except Exception as e:
                print(f" Error reading {file_path}: {e}")
        
        # Calculate metrics
        if ai_predictions and human_predictions:
            ai_avg = np.mean(ai_predictions)
            human_avg = np.mean(human_predictions)
            separation = ai_avg - human_avg
            
            print(f"\n Evaluation Results:")
            print(f"  AI texts average: {ai_avg:.3f}")
            print(f"  Human texts average: {human_avg:.3f}")
            print(f"  Separation: {separation:.3f}")
            
            if separation > 0.3:
                print("   Good separation between AI and human text")
            elif separation > 0.1:
                print("   Moderate separation between AI and human text")
            else:
                print("   Poor separation between AI and human text")
