"""Comprehensive metrics and evaluation for linking and labeling models."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive evaluator for linking and labeling tasks."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
    
    def compute_linking_metrics(self, 
                               y_true: List[int], 
                               y_pred: List[int],
                               y_scores: List[float],
                               k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Compute linking metrics (Precision@k, Recall@k, MAP, MRR, etc.).
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            k_values: List of k values for Precision@k and Recall@k
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['precision_neg'] = float(precision[0])
        metrics['precision_pos'] = float(precision[1])
        metrics['recall_neg'] = float(recall[0])
        metrics['recall_pos'] = float(recall[1])
        metrics['f1_neg'] = float(f1[0])
        metrics['f1_pos'] = float(f1[1])
        
        # Macro averages
        metrics['precision_macro'] = float(np.mean(precision))
        metrics['recall_macro'] = float(np.mean(recall))
        metrics['f1_macro'] = float(np.mean(f1))
        
        # Micro averages (overall)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        metrics['precision_micro'] = float(precision_micro)
        metrics['recall_micro'] = float(recall_micro)
        metrics['f1_micro'] = float(f1_micro)
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = float(precision_weighted)
        metrics['recall_weighted'] = float(recall_weighted)
        metrics['f1_weighted'] = float(f1_weighted)
        
        # Accuracy
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # AUC-ROC
        try:
            if len(np.unique(y_true)) > 1:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_scores))
        except Exception as e:
            logger.warning(f"Could not compute AUC-ROC: {e}")
            metrics['auc_roc'] = 0.0
        
        # MAP (Mean Average Precision)
        try:
            metrics['map'] = float(average_precision_score(y_true, y_scores))
        except Exception as e:
            logger.warning(f"Could not compute MAP: {e}")
            metrics['map'] = 0.0
        
        # MRR (Mean Reciprocal Rank)
        metrics['mrr'] = self._compute_mrr(y_true, y_pred, y_scores)
        
        # Precision@k and Recall@k
        for k in k_values:
            metrics[f'precision_at_{k}'] = self._compute_precision_at_k(
                y_true, y_pred, y_scores, k
            )
            metrics[f'recall_at_{k}'] = self._compute_recall_at_k(
                y_true, y_pred, y_scores, k
            )
        
        return metrics
    
    def compute_labeling_metrics(self,
                                y_true: List[List[str]],
                                y_pred: List[List[str]]) -> Dict[str, Any]:
        """
        Compute labeling metrics (multi-label classification).
        
        Args:
            y_true: True label lists
            y_pred: Predicted label lists
            
        Returns:
            Dictionary of computed metrics
        """
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import hamming_loss
        
        # Binarize labels
        mlb = MultiLabelBinarizer()
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)
        
        metrics = {}
        
        # Hamming loss
        metrics['hamming_loss'] = float(hamming_loss(y_true_bin, y_pred_bin))
        
        # Accuracy (subset accuracy / exact match ratio)
        metrics['accuracy'] = float(np.mean(y_true_bin == y_pred_bin).all(axis=1))
        
        # Precision, Recall, F1
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average='micro', zero_division=0
        )
        metrics['precision_micro'] = float(precision_micro)
        metrics['recall_micro'] = float(recall_micro)
        metrics['f1_micro'] = float(f1_micro)
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average='macro', zero_division=0
        )
        metrics['precision_macro'] = float(precision_macro)
        metrics['recall_macro'] = float(recall_macro)
        metrics['f1_macro'] = float(f1_macro)
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = float(precision_weighted)
        metrics['recall_weighted'] = float(recall_weighted)
        metrics['f1_weighted'] = float(f1_weighted)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average=None, zero_division=0
        )
        
        metrics['per_class_metrics'] = {
            label: {
                'precision': float(p),
                'recall': float(r),
                'f1': float(f),
                'support': int(s)
            }
            for label, p, r, f, s in zip(mlb.classes_, precision, recall, f1, support)
        }
        
        # Label coverage and imbalance
        all_labels = [label for labels in y_true for label in labels]
        label_counts = {label: all_labels.count(label) for label in set(all_labels)}
        
        metrics['label_coverage'] = {
            'unique_labels': len(label_counts),
            'total_samples': len(y_true),
            'avg_labels_per_sample': np.mean([len(labels) for labels in y_true]),
            'top_labels': dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        return metrics
    
    def _compute_precision_at_k(self, y_true, y_pred, y_scores, k: int) -> float:
        """Compute Precision@k."""
        # Sort by score descending
        sorted_indices = np.argsort(y_scores)[::-1]
        top_k = sorted_indices[:k]
        
        if len(top_k) == 0:
            return 0.0
        
        true_positives = sum(y_true[i] == 1 and y_pred[i] == 1 for i in top_k)
        return true_positives / len(top_k)
    
    def _compute_recall_at_k(self, y_true, y_pred, y_scores, k: int) -> float:
        """Compute Recall@k."""
        total_positive = sum(y_true == 1)
        if total_positive == 0:
            return 0.0
        
        sorted_indices = np.argsort(y_scores)[::-1]
        top_k = sorted_indices[:k]
        
        true_positives = sum(y_true[i] == 1 for i in top_k)
        return true_positives / total_positive
    
    def _compute_mrr(self, y_true, y_pred, y_scores) -> float:
        """Compute Mean Reciprocal Rank."""
        # For multi-query scenario
        reciprocals = []
        
        sorted_indices = np.argsort(y_scores)[::-1]
        
        # Find rank of first relevant item
        for rank, idx in enumerate(sorted_indices, 1):
            if y_true[idx] == 1:
                reciprocals.append(1.0 / rank)
                break
        
        return np.mean(reciprocals) if reciprocals else 0.0
