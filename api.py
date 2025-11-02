"""Main API for HITL learning system."""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .feedback.store import FeedbackStore
from .feedback.schema import FeedbackEntry, FeedbackSchema
from .active_learning.sampler import ActiveLearningSampler
from .retraining.pipeline import RetrainingPipeline
from .retraining.registry import ModelRegistry
from .ab_testing.router import ABRouter, ABEvaluator

logger = logging.getLogger(__name__)


class HITLAPI:
    """Main API for Human-in-the-Loop learning system."""
    
    def __init__(
        self,
        feedback_path: Path = Path("feedback"),
        model_registry_path: Path = Path("models/registry")
    ):
        self.feedback_store = FeedbackStore(feedback_path)
        self.model_registry = ModelRegistry(model_registry_path)
        self.sampler = ActiveLearningSampler(self.feedback_store)
        self.retraining_pipeline = RetrainingPipeline(
            self.feedback_store, 
            self.model_registry
        )
        self.ab_router = ABRouter()
        self.ab_evaluator = ABEvaluator()
    
    def log_feedback(
        self,
        repo: str,
        entity_type: str,
        entity_id: str,
        suggestion: Dict[str, Any],
        final_decision: Dict[str, Any],
        signal: str,
        edit_distance: float = 0.0,
        confidence: float = 1.0,
        model_versions: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Log feedback from maintainers.
        
        Args:
            repo: Repository name
            entity_type: Type of entity (pr, issue, link, label, merge_policy)
            entity_id: Unique identifier
            suggestion: Model suggestion
            final_decision: Human decision/action
            signal: Feedback signal
            edit_distance: Edit distance (0-1)
            confidence: Model confidence (0-1)
            model_versions: Model versions used
            
        Returns:
            True if successful
        """
        try:
            entry = FeedbackSchema.create_entry(
                repo=repo,
                entity_type=entity_type,
                entity_id=entity_id,
                suggestion=suggestion,
                final_decision=final_decision,
                signal=signal,
                edit_distance=edit_distance,
                confidence=confidence,
                model_versions=model_versions
            )
            
            success = self.feedback_store.append(entry)
            if success:
                logger.info(f"Logged feedback: {entity_type}:{entity_id} -> {signal}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
            return False
    
    def sample_active_learning(
        self,
        n: int,
        strategy: str = "uncertainty",
        entity_type: Optional[str] = None
    ) -> List[FeedbackEntry]:
        """
        Sample examples for active learning.
        
        Args:
            n: Number of samples
            strategy: Sampling strategy
            entity_type: Filter by entity type
            
        Returns:
            List of sampled entries
        """
        if strategy == "uncertainty":
            return self.sampler.sample_uncertainty(n, entity_type)
        elif strategy == "diversity":
            return self.sampler.sample_diversity(n, entity_type)
        elif strategy == "drift":
            return self.sampler.sample_drift(n, entity_type)
        elif strategy == "mixed":
            return self.sampler.sample_mixed(n, entity_type)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def schedule_retrain(
        self,
        components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Schedule retraining of model components.
        
        Args:
            components: List of components to retrain
            
        Returns:
            Retraining results
        """
        logger.info(f"Scheduling retraining for: {components}")
        
        results = self.retraining_pipeline.retrain_all_models(components)
        
        logger.info(f"Retraining completed: {list(results.keys())}")
        return results
    
    def route_variant(
        self,
        context: Dict[str, Any],
        model_a: str = "current",
        model_b: str = "new"
    ) -> str:
        """
        Route request to model variant for A/B testing.
        
        Args:
            context: Request context
            model_a: Model A name
            model_b: Model B name
            
        Returns:
            Variant name ("A" or "B")
        """
        return self.ab_router.route_variant(context, model_a, model_b)
    
    def evaluate_ab_experiment(
        self,
        experiment_name: str,
        outcome_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate A/B experiment results.
        
        Args:
            experiment_name: Name of experiment
            outcome_data: Outcome data
            
        Returns:
            Evaluation results
        """
        return self.ab_evaluator.evaluate_experiment(experiment_name, outcome_data)
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return self.feedback_store.get_stats()
    
    def get_model_versions(self) -> Dict[str, List[str]]:
        """Get available model versions."""
        return self.model_registry.list_models()
    
    def export_training_data(
        self,
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Export feedback data for training."""
        return self.feedback_store.export_for_training(entity_types, min_confidence)


# Convenience functions
def log_feedback(
    repo: str,
    entity_type: str,
    entity_id: str,
    suggestion: Dict[str, Any],
    final_decision: Dict[str, Any],
    signal: str,
    **kwargs
) -> bool:
    """Log feedback (convenience function)."""
    api = HITLAPI()
    return api.log_feedback(
        repo, entity_type, entity_id, suggestion, 
        final_decision, signal, **kwargs
    )


def sample_active_learning(
    n: int,
    strategy: str = "uncertainty",
    entity_type: Optional[str] = None
) -> List[FeedbackEntry]:
    """Sample active learning examples (convenience function)."""
    api = HITLAPI()
    return api.sample_active_learning(n, strategy, entity_type)


def schedule_retrain(components: Optional[List[str]] = None) -> Dict[str, Any]:
    """Schedule retraining (convenience function)."""
    api = HITLAPI()
    return api.schedule_retrain(components)


def route_variant(
    context: Dict[str, Any],
    model_a: str = "current",
    model_b: str = "new"
) -> str:
    """Route variant (convenience function)."""
    api = HITLAPI()
    return api.route_variant(context, model_a, model_b)
