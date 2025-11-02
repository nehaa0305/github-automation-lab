"""Retraining pipeline for different model components."""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from ..feedback.store import FeedbackStore
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Pipeline for retraining models with feedback data."""
    
    def __init__(
        self,
        feedback_store: FeedbackStore,
        model_registry: ModelRegistry,
        training_corpus_path: Path = Path("training_corpus")
    ):
        self.feedback_store = feedback_store
        self.model_registry = model_registry
        self.training_corpus_path = Path(training_corpus_path)
        self.training_corpus_path.mkdir(parents=True, exist_ok=True)
    
    def retrain_linking_model(
        self,
        feedback_data: List[Dict[str, Any]],
        model_name: str = "linking_model"
    ) -> Dict[str, Any]:
        """
        Retrain linking model with feedback data.
        
        Args:
            feedback_data: Feedback data for training
            model_name: Name of the model
            
        Returns:
            Training results
        """
        logger.info(f"Retraining {model_name}...")
        
        # Extract positive and negative examples
        positive_examples = []
        negative_examples = []
        
        for entry in feedback_data:
            if entry.get("entity_type") != "link":
                continue
            
            if entry.get("signal") in ["accept", "approve"]:
                positive_examples.append(entry)
            elif entry.get("signal") in ["reject", "request_changes"]:
                negative_examples.append(entry)
        
        logger.info(f"Found {len(positive_examples)} positive, {len(negative_examples)} negative examples")
        
        # Simulate training (would use actual ML training)
        training_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1": 0.80,
            "auc": 0.88
        }
        
        # Create training data info
        training_data_info = {
            "size": len(positive_examples) + len(negative_examples),
            "positive_examples": len(positive_examples),
            "negative_examples": len(negative_examples),
            "time_range": self._get_time_range(feedback_data),
            "repos": list(set(entry.get("repo", "") for entry in feedback_data))
        }
        
        # Register new model version
        version = self._get_next_version(model_name)
        model_path = self.training_corpus_path / f"{model_name}_{version}.pkl"
        
        # Save dummy model (would save actual trained model)
        with open(model_path, 'w') as f:
            json.dump({"model": "dummy_linking_model", "version": version}, f)
        
        self.model_registry.register_model(
            model_name=model_name,
            model_type="linking",
            version=version,
            model_path=model_path,
            metrics=training_metrics,
            training_data_info=training_data_info
        )
        
        logger.info(f"Retrained {model_name} v{version}")
        
        return {
            "model_name": model_name,
            "version": version,
            "metrics": training_metrics,
            "training_data_info": training_data_info
        }
    
    def retrain_labeling_model(
        self,
        feedback_data: List[Dict[str, Any]],
        model_name: str = "labeling_model"
    ) -> Dict[str, Any]:
        """Retrain labeling model with feedback data."""
        logger.info(f"Retraining {model_name}...")
        
        # Extract corrected labels
        corrected_labels = []
        for entry in feedback_data:
            if entry.get("entity_type") != "label":
                continue
            
            if entry.get("signal") in ["accept", "edit"]:
                corrected_labels.append(entry)
        
        logger.info(f"Found {len(corrected_labels)} corrected labels")
        
        # Simulate training
        training_metrics = {
            "accuracy": 0.78,
            "precision_macro": 0.75,
            "recall_macro": 0.73,
            "f1_macro": 0.74,
            "f1_weighted": 0.76
        }
        
        training_data_info = {
            "size": len(corrected_labels),
            "time_range": self._get_time_range(feedback_data),
            "repos": list(set(entry.get("repo", "") for entry in feedback_data))
        }
        
        version = self._get_next_version(model_name)
        model_path = self.training_corpus_path / f"{model_name}_{version}.pkl"
        
        with open(model_path, 'w') as f:
            json.dump({"model": "dummy_labeling_model", "version": version}, f)
        
        self.model_registry.register_model(
            model_name=model_name,
            model_type="labeling",
            version=version,
            model_path=model_path,
            metrics=training_metrics,
            training_data_info=training_data_info
        )
        
        return {
            "model_name": model_name,
            "version": version,
            "metrics": training_metrics,
            "training_data_info": training_data_info
        }
    
    def retrain_rag_model(
        self,
        feedback_data: List[Dict[str, Any]],
        model_name: str = "rag_model"
    ) -> Dict[str, Any]:
        """Retrain RAG model with feedback data."""
        logger.info(f"Retraining {model_name}...")
        
        # Extract accepted generations
        accepted_generations = []
        for entry in feedback_data:
            if entry.get("entity_type") in ["pr", "issue"]:
                if entry.get("signal") in ["accept", "edit"]:
                    accepted_generations.append(entry)
        
        logger.info(f"Found {len(accepted_generations)} accepted generations")
        
        # Simulate training
        training_metrics = {
            "bleu": 0.42,
            "rouge_l": 0.45,
            "meteor": 0.38,
            "semantic_similarity": 0.82
        }
        
        training_data_info = {
            "size": len(accepted_generations),
            "time_range": self._get_time_range(feedback_data),
            "repos": list(set(entry.get("repo", "") for entry in feedback_data))
        }
        
        version = self._get_next_version(model_name)
        model_path = self.training_corpus_path / f"{model_name}_{version}.pkl"
        
        with open(model_path, 'w') as f:
            json.dump({"model": "dummy_rag_model", "version": version}, f)
        
        self.model_registry.register_model(
            model_name=model_name,
            model_type="rag",
            version=version,
            model_path=model_path,
            metrics=training_metrics,
            training_data_info=training_data_info
        )
        
        return {
            "model_name": model_name,
            "version": version,
            "metrics": training_metrics,
            "training_data_info": training_data_info
        }
    
    def retrain_risk_model(
        self,
        feedback_data: List[Dict[str, Any]],
        model_name: str = "risk_model"
    ) -> Dict[str, Any]:
        """Retrain risk model with feedback data."""
        logger.info(f"Retraining {model_name}...")
        
        # Extract merge outcomes
        merge_outcomes = []
        for entry in feedback_data:
            if entry.get("entity_type") == "merge_policy":
                merge_outcomes.append(entry)
        
        logger.info(f"Found {len(merge_outcomes)} merge outcomes")
        
        # Simulate training
        training_metrics = {
            "roc_auc": 0.86,
            "precision": 0.82,
            "recall": 0.78,
            "f1": 0.80,
            "accuracy": 0.83
        }
        
        training_data_info = {
            "size": len(merge_outcomes),
            "time_range": self._get_time_range(feedback_data),
            "repos": list(set(entry.get("repo", "") for entry in feedback_data))
        }
        
        version = self._get_next_version(model_name)
        model_path = self.training_corpus_path / f"{model_name}_{version}.pkl"
        
        with open(model_path, 'w') as f:
            json.dump({"model": "dummy_risk_model", "version": version}, f)
        
        self.model_registry.register_model(
            model_name=model_name,
            model_type="risk",
            version=version,
            model_path=model_path,
            metrics=training_metrics,
            training_data_info=training_data_info
        )
        
        return {
            "model_name": model_name,
            "version": version,
            "metrics": training_metrics,
            "training_data_info": training_data_info
        }
    
    def retrain_all_models(
        self,
        components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrain all specified model components.
        
        Args:
            components: List of components to retrain
            
        Returns:
            Retraining results
        """
        if components is None:
            components = ["linking", "labeling", "rag", "risk"]
        
        # Get feedback data
        feedback_data = self.feedback_store.export_for_training()
        
        results = {}
        
        for component in components:
            if component == "linking":
                results["linking"] = self.retrain_linking_model(feedback_data)
            elif component == "labeling":
                results["labeling"] = self.retrain_labeling_model(feedback_data)
            elif component == "rag":
                results["rag"] = self.retrain_rag_model(feedback_data)
            elif component == "risk":
                results["risk"] = self.retrain_risk_model(feedback_data)
        
        logger.info(f"Retrained components: {list(results.keys())}")
        return results
    
    def _get_time_range(self, feedback_data: List[Dict[str, Any]]) -> str:
        """Get time range from feedback data."""
        if not feedback_data:
            return "Unknown"
        
        timestamps = [entry.get("ts", "") for entry in feedback_data if entry.get("ts")]
        if not timestamps:
            return "Unknown"
        
        timestamps.sort()
        return f"{timestamps[0]} to {timestamps[-1]}"
    
    def _get_next_version(self, model_name: str) -> str:
        """Get next version for model."""
        latest_version = self.model_registry.get_latest_version(model_name)
        if latest_version is None:
            return "1.0.0"
        
        # Increment patch version
        import semver
        current = semver.VersionInfo.parse(latest_version)
        next_version = current.bump_patch()
        return str(next_version)
