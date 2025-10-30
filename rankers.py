"""Ranker models for PR-issue linking."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

logger = logging.getLogger(__name__)


class RankerModel:
    """Learned ranker models for re-ranking retrieved results."""
    
    def __init__(self, ranker_type: str = "logistic", random_state: int = 42):
        """
        Initialize ranker model.
        
        Args:
            ranker_type: Type of ranker ('logistic', 'xgboost', 'mlp')
            random_state: Random state for reproducibility
        """
        self.ranker_type = ranker_type
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    def _create_model(self):
        """Create the appropriate model based on ranker_type."""
        if self.ranker_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                verbose=0
            )
        elif self.ranker_type == "xgboost":
            self.model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif self.ranker_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                random_state=self.random_state,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown ranker type: {self.ranker_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the ranker model.
        
        Args:
            X: Feature matrix
            y: Binary labels (0 or 1)
        """
        if self.model is None:
            self._create_model()
        
        logger.info(f"Training {self.ranker_type} ranker on {len(X)} samples")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"{self.ranker_type} ranker trained successfully")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        return self.model.predict_proba(X)[:, 1]  # Return probability of class 1
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        return self.model.predict(X)
    
    def save(self, file_path: Path):
        """
        Save the trained model.
        
        Args:
            file_path: Path to save the model
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump({
                'ranker_type': self.ranker_type,
                'model': self.model,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Saved ranker model to {file_path}")
    
    def load(self, file_path: Path):
        """
        Load a trained model.
        
        Args:
            file_path: Path to the saved model
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.ranker_type = data['ranker_type']
        self.model = data['model']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Loaded ranker model from {file_path}")
    
    @staticmethod
    def extract_features(retrieved_results: List[Dict[str, Any]], 
                        query_embedding: np.ndarray) -> np.ndarray:
        """
        Extract features from retrieved results for ranking.
        
        Args:
            retrieved_results: List of retrieved records
            query_embedding: Query embedding vector
            
        Returns:
            Feature matrix
        """
        features = []
        
        for result in retrieved_results:
            score = result.get('score', 0.0)
            title_len = len(result.get('title', ''))
            body_len = len(result.get('text_preview', ''))
            
            # Basic features
            feat = [
                score,  # Cosine similarity
                title_len,
                body_len,
                title_len / max(body_len, 1),  # Title/body ratio
            ]
            
            features.append(feat)
        
        return np.array(features)
