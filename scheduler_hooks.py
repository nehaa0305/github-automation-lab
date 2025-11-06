"""Scheduler hooks for ML-related jobs."""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import asyncio

from .ml_client import MLClient
from .policy_integration import PolicyManager

logger = logging.getLogger(__name__)


class SchedulerHooks:
    """Scheduler hooks for ML jobs."""
    
    def __init__(
        self,
        ml_client: Optional[MLClient] = None,
        policy_manager: Optional[PolicyManager] = None
    ):
        self.ml_client = ml_client or MLClient()
        self.policy_manager = policy_manager or PolicyManager()
        self.job_registry = {}
    
    def register_ml_jobs(self, scheduler) -> None:
        """
        Register ML-related jobs with existing scheduler.
        
        Args:
            scheduler: Existing job scheduler instance
        """
        logger.info("Registering ML jobs with scheduler...")
        
        # Register retraining job
        if self.policy_manager.get_retraining_config().get("enabled", False):
            schedule = self.policy_manager.get_retraining_config().get("schedule", "weekly")
            scheduler.register_job(
                name="weekly_retrain_models",
                func=self.weekly_retrain_models,
                schedule=schedule,
                description="Retrain ML models with accumulated feedback"
            )
            self.job_registry["weekly_retrain_models"] = True
        
        # Register index refresh job
        scheduler.register_job(
            name="daily_index_refresh",
            func=self.daily_index_refresh,
            schedule="daily",
            description="Refresh embeddings and FAISS indices"
        )
        self.job_registry["daily_index_refresh"] = True
        
        # Register feedback aggregation job
        scheduler.register_job(
            name="feedback_aggregation_job",
            func=self.feedback_aggregation_job,
            schedule="hourly",
            description="Aggregate and process feedback data"
        )
        self.job_registry["feedback_aggregation_job"] = True
        
        # Register model evaluation job
        scheduler.register_job(
            name="model_evaluation_job",
            func=self.model_evaluation_job,
            schedule="daily",
            description="Evaluate model performance and generate reports"
        )
        self.job_registry["model_evaluation_job"] = True
        
        logger.info(f"Registered {len(self.job_registry)} ML jobs")
    
    def weekly_retrain_models(self) -> Dict[str, Any]:
        """
        Weekly retraining job.
        
        Returns:
            Job execution result
        """
        logger.info("Starting weekly model retraining...")
        
        result = {
            "job_name": "weekly_retrain_models",
            "start_time": datetime.utcnow().isoformat(),
            "success": False,
            "retrained_models": [],
            "errors": []
        }
        
        try:
            # Check if we have enough feedback data
            min_samples = self.policy_manager.get_retraining_config().get("min_feedback_samples", 100)
            
            # This would check actual feedback count
            feedback_count = self._get_feedback_count()
            
            if feedback_count < min_samples:
                result["message"] = f"Insufficient feedback data: {feedback_count} < {min_samples}"
                result["success"] = True  # Not an error, just skipped
                return result
            
            # Retrain models
            retrain_result = self.ml_client._call_module(
                "hitl_learning",
                "retrain",
                {"components": ["linking", "labeling", "rag", "risk"]}
            )
            
            if retrain_result.get("success", False):
                result["retrained_models"] = retrain_result.get("retrained_models", [])
                result["success"] = True
                logger.info(f"Successfully retrained models: {result['retrained_models']}")
            else:
                result["errors"].append(retrain_result.get("error", "Unknown retraining error"))
        
        except Exception as e:
            logger.error(f"Error in weekly retraining: {e}")
            result["errors"].append(str(e))
        
        result["end_time"] = datetime.utcnow().isoformat()
        return result
    
    def daily_index_refresh(self) -> Dict[str, Any]:
        """
        Daily index refresh job.
        
        Returns:
            Job execution result
        """
        logger.info("Starting daily index refresh...")
        
        result = {
            "job_name": "daily_index_refresh",
            "start_time": datetime.utcnow().isoformat(),
            "success": False,
            "refreshed_indices": [],
            "errors": []
        }
        
        try:
            # Refresh embeddings index
            embeddings_result = self.ml_client._call_module(
                "embeddings_index",
                "refresh_index",
                {}
            )
            
            if embeddings_result.get("success", False):
                result["refreshed_indices"].append("embeddings")
            
            # Refresh FAISS indices
            faiss_result = self.ml_client._call_module(
                "embeddings_index",
                "refresh_faiss",
                {}
            )
            
            if faiss_result.get("success", False):
                result["refreshed_indices"].append("faiss")
            
            result["success"] = len(result["refreshed_indices"]) > 0
            
        except Exception as e:
            logger.error(f"Error in daily index refresh: {e}")
            result["errors"].append(str(e))
        
        result["end_time"] = datetime.utcnow().isoformat()
        return result
    
    def feedback_aggregation_job(self) -> Dict[str, Any]:
        """
        Hourly feedback aggregation job.
        
        Returns:
            Job execution result
        """
        logger.info("Starting feedback aggregation...")
        
        result = {
            "job_name": "feedback_aggregation_job",
            "start_time": datetime.utcnow().isoformat(),
            "success": False,
            "aggregated_count": 0,
            "errors": []
        }
        
        try:
            # Aggregate feedback data
            aggregation_result = self.ml_client._call_module(
                "hitl_learning",
                "aggregate_feedback",
                {}
            )
            
            if aggregation_result.get("success", False):
                result["aggregated_count"] = aggregation_result.get("count", 0)
                result["success"] = True
            else:
                result["errors"].append(aggregation_result.get("error", "Unknown aggregation error"))
        
        except Exception as e:
            logger.error(f"Error in feedback aggregation: {e}")
            result["errors"].append(str(e))
        
        result["end_time"] = datetime.utcnow().isoformat()
        return result
    
    def model_evaluation_job(self) -> Dict[str, Any]:
        """
        Daily model evaluation job.
        
        Returns:
            Job execution result
        """
        logger.info("Starting model evaluation...")
        
        result = {
            "job_name": "model_evaluation_job",
            "start_time": datetime.utcnow().isoformat(),
            "success": False,
            "evaluation_results": {},
            "errors": []
        }
        
        try:
            # Evaluate models
            evaluation_result = self.ml_client._call_module(
                "hitl_learning",
                "evaluate_models",
                {}
            )
            
            if evaluation_result.get("success", False):
                result["evaluation_results"] = evaluation_result.get("results", {})
                result["success"] = True
            else:
                result["errors"].append(evaluation_result.get("error", "Unknown evaluation error"))
        
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            result["errors"].append(str(e))
        
        result["end_time"] = datetime.utcnow().isoformat()
        return result
    
    def _get_feedback_count(self) -> int:
        """Get current feedback count (placeholder)."""
        # This would query the actual feedback store
        return 150  # Placeholder
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of all registered ML jobs."""
        return {
            "registered_jobs": list(self.job_registry.keys()),
            "total_jobs": len(self.job_registry),
            "timestamp": datetime.utcnow().isoformat()
        }
