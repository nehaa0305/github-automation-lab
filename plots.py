"""Generate plots and visualizations for metrics."""

import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns

logger = logging.getLogger(__name__)


class MetricsPlotter:
    """Generate plots for metrics visualization."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize plotter.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8')  # Modern style
    
    def plot_precision_at_k(self, 
                           metrics_dict: Dict[str, Dict[str, float]],
                           k_values: List[int] = [1, 3, 5, 10, 20]):
        """
        Plot Precision@k curves for different models.
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            k_values: List of k values to plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, metrics in metrics_dict.items():
            precisions = [
                metrics.get(f'precision_at_{k}', 0)
                for k in k_values
            ]
            ax.plot(k_values, precisions, marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('k (top-k)', fontsize=12)
        ax.set_ylabel('Precision@k', fontsize=12)
        ax.set_title('Precision@k Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'precision_at_k.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Precision@k plot to {output_path}")
    
    def plot_map_bar_chart(self, metrics_dict: Dict[str, Dict[str, float]]):
        """
        Plot MAP (Mean Average Precision) bar chart.
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = list(metrics_dict.keys())
        map_values = [metrics.get('map', 0) for metrics in metrics_dict.values()]
        
        bars = ax.bar(model_names, map_values, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, val in zip(bars, map_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom')
        
        ax.set_ylabel('MAP Score', fontsize=12)
        ax.set_title('Mean Average Precision (MAP) Comparison', 
                    fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        output_path = self.output_dir / 'map_bar_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved MAP bar chart to {output_path}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str = "Model"):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        output_path = self.output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_path}")
    
    def plot_f1_comparison(self, 
                          micro_f1: List[float],
                          macro_f1: List[float],
                          model_names: List[str]):
        """
        Plot F1 micro vs macro comparison.
        
        Args:
            micro_f1: List of F1 micro scores
            macro_f1: List of F1 macro scores
            model_names: List of model names
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, micro_f1, width, label='F1-Micro', color='steelblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, macro_f1, width, label='F1-Macro', color='lightcoral', alpha=0.7)
        
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_title('F1 Micro vs Macro Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        output_path = self.output_dir / 'f1_micro_vs_macro.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved F1 comparison plot to {output_path}")
    
    def plot_per_class_metrics(self, per_class_data: Dict[str, Dict[str, float]]):
        """
        Plot per-class precision, recall, F1.
        
        Args:
            per_class_data: Dictionary mapping class names to metrics
        """
        if not per_class_data:
            return
        
        classes = list(per_class_data.keys())
        precision = [per_class_data[c]['precision'] for c in classes]
        recall = [per_class_data[c]['recall'] for c in classes]
        f1 = [per_class_data[c]['f1'] for c in classes]
        
        fig, ax = plt.subplots(figsize=(max(12, len(classes)), 6))
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.7)
        ax.bar(x, recall, width, label='Recall', alpha=0.7)
        ax.bar(x + width, f1, width, label='F1', alpha=0.7)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        output_path = self.output_dir / 'per_class_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved per-class metrics plot to {output_path}")
