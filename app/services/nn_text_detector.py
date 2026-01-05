"""
Neural Network Text Detector for AI content detection.
Uses PyTorch MLP with epoch-based training and loss tracking.
"""
import os
import sys
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Neural network training unavailable.")


class MLPClassifier(nn.Module):
    """Simple MLP for text classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkDetector:
    """Neural network-based AI text detector with epoch training."""
    
    def __init__(self, model_path: str = "nn_detector_model.pkl"):
        self.model_path = os.path.join(PROJECT_ROOT, model_path)
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        # Device selection
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")  # Apple Silicon
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            print(f" Using device: {self.device}")
        
        # Try to load existing model
        self.load_model()
    
    def train_model(self, ai_texts: List[str], human_texts: List[str],
                   epochs: int = 50, batch_size: int = 32, 
                   learning_rate: float = 0.001, test_size: float = 0.2,
                   progress_callback=None) -> Dict:
        """
        Train the neural network with epoch-based optimization.
        
        Args:
            ai_texts: List of AI-generated text samples
            human_texts: List of human-written text samples
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            test_size: Fraction for validation set
            progress_callback: Optional callback(epoch, train_loss, val_loss, train_acc, val_acc)
            
        Returns:
            Dictionary with training metrics and history
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Please run: pip install torch")
        
        print(f" Training Neural Network Detector...")
        print(f" AI texts: {len(ai_texts)}")
        print(f" Human texts: {len(human_texts)}")
        print(f" Epochs: {epochs}, Batch size: {batch_size}")
        
        # Prepare data
        texts = human_texts + ai_texts
        labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0=human, 1=AI
        
        # Create TF-IDF features
        print(" Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        X = self.vectorizer.fit_transform(texts).toarray()
        y = np.array(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = MLPClassifier(input_dim, hidden_dims=[512, 256, 128]).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training history
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        # Move validation data to device
        X_val_t = X_val_t.to(self.device)
        y_val_t = y_val_t.to(self.device)
        
        print(" Starting training...")
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_predicted = (val_outputs > 0.5).float()
                val_acc = (val_predicted == y_val_t).sum().item() / len(y_val_t)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, train_loss, val_loss, train_acc, val_acc)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
            
            # Early stopping
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val_t)
            y_pred_proba = val_outputs.cpu().numpy().flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_val, y_pred)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        cm = confusion_matrix(y_val, y_pred)
        
        print(f"\n Training completed!")
        print(f" Final Accuracy: {accuracy:.3f}")
        print(f" AUC Score: {auc_score:.3f}")
        print(classification_report(y_val, y_pred, target_names=['Human', 'AI']))
        
        self.is_trained = True
        self.save_model()
        
        # Return detailed metrics
        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'confusion_matrix': cm.tolist(),
            'training_history': self.training_history,
            'epochs_trained': len(self.training_history['train_loss']),
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
        }
        
        return metrics
    
    def predict(self, text: str) -> Dict:
        """Predict whether text is AI-generated."""
        if not self.is_trained or self.model is None:
            return {'prediction': 'unknown', 'probability': 0.5, 'confidence': 'low'}
        
        # Transform text
        X = self.vectorizer.transform([text]).toarray()
        X_t = torch.FloatTensor(X).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_t)
            prob = output.cpu().numpy().flatten()[0]
        
        prediction = 'ai' if prob > 0.5 else 'human'
        confidence = 'high' if abs(prob - 0.5) > 0.3 else 'medium' if abs(prob - 0.5) > 0.15 else 'low'
        
        return {
            'prediction': prediction,
            'probability': float(prob),
            'confidence': confidence,
            'label': 1 if prediction == 'ai' else 0
        }
    
    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            return
        
        data = {
            'model_state': self.model.state_dict(),
            'vectorizer': self.vectorizer,
            'input_dim': self.vectorizer.max_features if self.vectorizer else 5000,
            'training_history': self.training_history
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f" Neural network model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model."""
        if not os.path.exists(self.model_path):
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.vectorizer = data['vectorizer']
            input_dim = data.get('input_dim', 5000)
            
            self.model = MLPClassifier(input_dim, hidden_dims=[512, 256, 128]).to(self.device)
            self.model.load_state_dict(data['model_state'])
            self.model.eval()
            
            self.training_history = data.get('training_history', {})
            self.is_trained = True
            
            print(f" Neural network model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            print(f" Failed to load model: {e}")
            return False
