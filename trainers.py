"""Trainers for issue label classification models."""

import logging
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb

logger = logging.getLogger(__name__)


class LabelingModel:
    """Multi-label issue classification models."""
    
    def __init__(self, model_type: str = "tfidf_logreg", random_state: int = 42):
        """
        Initialize labeling model.
        
        Args:
            model_type: Type of model ('tfidf_logreg', 'embedding_logreg', 'mlp')
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.label_binarizer = MultiLabelBinarizer()
        self.is_fitted = False
        
    def _create_model(self, embedding_dim: Optional[int] = None):
        """Create the appropriate model based on model_type."""
        if self.model_type == "tfidf_logreg":
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    verbose=0,
                    n_jobs=-1
                ))
            ])
        elif self.model_type == "embedding_logreg":
            if embedding_dim is None:
                raise ValueError("embedding_dim must be provided for embedding models")
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                verbose=0
            )
        elif self.model_type == "mlp":
            if embedding_dim is None:
                raise ValueError("embedding_dim must be provided for MLP models")
            self.model = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=self.random_state,
                verbose=False
            )
        elif self.model_type == "xgboost":
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    n_jobs=-1
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: List[List[str]]):
        """
        Train the labeling model.
        
        Args:
            X: Feature matrix (text or embeddings)
            y: List of label lists for each sample
        """
        if self.model is None:
            self._create_model(embedding_dim=X.shape[1] if len(X.shape) > 1 else None)
        
        # Binarize labels
        logger.info("Binarizing labels...")
        y_binarized = self.label_binarizer.fit_transform(y)
        logger.info(f"Created {len(self.label_binarizer.classes_)} label classes")
        
        logger.info(f"Training {self.model_type} model on {len(X)} samples")
        
        if self.model_type in ["embedding_logreg", "mlp"]:
            # For embedding-based models, fit directly
            self.model.fit(X, y_binarized)
        else:
            # For pipeline models, fit with text
            self.model.fit(X, y)
        
        self.is_fitted = True
        logger.info(f"{self.model_type} model trained successfully")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict label probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        if self.model_type in ["embedding_logreg", "mlp"]:
            return self.model.predict_proba(X)
        else:
            # For pipeline models
            return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> List[List[str]]:
        """
        Predict labels for samples.
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for label assignment
            
        Returns:
            List of predicted label lists
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        probas = self.predict_proba(X)
        
        # Binary predictions based on threshold
        predictions = (probas >= threshold).astype(int)
        
        # Convert back to label lists
        label_lists = []
        for pred in predictions:
            labels = [
                self.label_binarizer.classes_[i] 
                for i, val in enumerate(pred) if val == 1
            ]
            label_lists.append(labels)
        
        return label_lists
    
    def get_label_counts(self, y: List[List[str]]) -> Dict[str, int]:
        """
        Get label frequency counts.
        
        Args:
            y: List of label lists
            
        Returns:
            Dictionary mapping labels to counts
        """
        all_labels = []
        for labels in y:
            all_labels.extend(labels)
        
        return Counter(all_labels)
    
    def save(self, file_path: Path):
        """
        Save the trained model.
        
        Args:
            file_path: Path to save the model
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'model': self.model,
                'label_binarizer': self.label_binarizer,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Saved labeling model to {file_path}")
    
    def load(self, file_path: Path):
        """
        Load a trained model.
        
        Args:
            file_path: Path to the saved model
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model_type = data['model_type']
        self.model = data['model']
        self.label_binarizer = data['label_binarizer']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Loaded labeling model from {file_path}")
