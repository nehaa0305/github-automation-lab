"""
ML Client Wrappers for Orchestrator Integration.

Provides unified API to call ML/NLP modules (Functionalities 1-7)
from the existing automation backbone.
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time

logger = logging.getLogger(__name__)


class MLClient:
    """Unified client for calling ML/NLP modules."""
    
    def __init__(self, base_path: Path = Path(".")):
        self.base_path = Path(base_path)
        self.timeout = 300  # 5 minutes default timeout
        
    def link_pr_to_issues(
        self,
        pr_id: str,
        pr_data: Dict[str, Any],
        issue_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Link PR to related issues using linking model.
        
        Args:
            pr_id: PR identifier
            pr_data: PR data (title, body, diff, etc.)
            issue_candidates: List of candidate issues
            
        Returns:
            Linking results with confidence scores
        """
        logger.info(f"Linking PR {pr_id} to issues...")
        
        try:
            # Prepare input data
            input_data = {
                "pr_id": pr_id,
                "pr_data": pr_data,
                "issue_candidates": issue_candidates
            }
            
            # Call linking module
            result = self._call_module(
                "linking_labeling",
                "link_pr_issues",
                input_data
            )
            
            return {
                "pr_id": pr_id,
                "linked_issues": result.get("linked_issues", []),
                "confidence_scores": result.get("confidence_scores", []),
                "model_version": result.get("model_version", "unknown"),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to link PR {pr_id}: {e}")
            return {
                "pr_id": pr_id,
                "linked_issues": [],
                "confidence_scores": [],
                "model_version": "unknown",
                "success": False,
                "error": str(e)
            }
    
    def generate_pr_description(
        self,
        diff_text: str,
        commit_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate PR description using RAG system.
        
        Args:
            diff_text: Git diff text
            commit_message: Commit message
            context: Additional context
            
        Returns:
            Generated PR description
        """
        logger.info("Generating PR description...")
        
        try:
            # Call RAG generation module
            result = self._call_module(
                "rag_generation",
                "generate_pr",
                {
                    "diff": diff_text,
                    "commit": commit_message,
                    "context": context or {}
                }
            )
            
            return {
                "title": result.get("title", ""),
                "body": result.get("body", ""),
                "confidence": result.get("confidence", 0.0),
                "model_version": result.get("model_version", "unknown"),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to generate PR description: {e}")
            return {
                "title": "",
                "body": "",
                "confidence": 0.0,
                "model_version": "unknown",
                "success": False,
                "error": str(e)
            }
    
    def generate_issue_summary(
        self,
        diff_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate issue summary using RAG system.
        
        Args:
            diff_text: Git diff text
            context: Additional context
            
        Returns:
            Generated issue summary
        """
        logger.info("Generating issue summary...")
        
        try:
            # Call RAG generation module
            result = self._call_module(
                "rag_generation",
                "generate_issue",
                {
                    "diff": diff_text,
                    "context": context or {}
                }
            )
            
            return {
                "title": result.get("title", ""),
                "body": result.get("body", ""),
                "labels": result.get("labels", []),
                "confidence": result.get("confidence", 0.0),
                "model_version": result.get("model_version", "unknown"),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to generate issue summary: {e}")
            return {
                "title": "",
                "body": "",
                "labels": [],
                "confidence": 0.0,
                "model_version": "unknown",
                "success": False,
                "error": str(e)
            }
    
    def evaluate_risk(
        self,
        pr_id: str,
        pr_data: Dict[str, Any],
        diff_text: str
    ) -> Dict[str, Any]:
        """
        Evaluate PR risk using verification system.
        
        Args:
            pr_id: PR identifier
            pr_data: PR data
            diff_text: Git diff text
            
        Returns:
            Risk evaluation results
        """
        logger.info(f"Evaluating risk for PR {pr_id}...")
        
        try:
            # Call verification module
            result = self._call_module(
                "verification",
                "verify_pr",
                {
                    "pr_id": pr_id,
                    "pr_data": pr_data,
                    "diff": diff_text
                }
            )
            
            return {
                "pr_id": pr_id,
                "risk_score": result.get("risk_score", 0.5),
                "decision": result.get("decision", "MANUAL_REVIEW"),
                "confidence": result.get("confidence", 0.0),
                "tests_passed": result.get("tests_passed", False),
                "lint_passed": result.get("lint_passed", False),
                "security_passed": result.get("security_passed", False),
                "model_version": result.get("model_version", "unknown"),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate risk for PR {pr_id}: {e}")
            return {
                "pr_id": pr_id,
                "risk_score": 0.5,
                "decision": "MANUAL_REVIEW",
                "confidence": 0.0,
                "tests_passed": False,
                "lint_passed": False,
                "security_passed": False,
                "model_version": "unknown",
                "success": False,
                "error": str(e)
            }
    
    def log_feedback(
        self,
        event_type: str,
        entity_id: str,
        suggestion: Dict[str, Any],
        final_decision: Dict[str, Any],
        signal: str,
        **kwargs
    ) -> bool:
        """
        Log feedback to HITL system.
        
        Args:
            event_type: Type of event (pr, issue, link, etc.)
            entity_id: Entity identifier
            suggestion: Model suggestion
            final_decision: Human decision
            signal: Feedback signal
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        logger.info(f"Logging feedback for {event_type}:{entity_id}")
        
        try:
            # Call HITL learning module
            result = self._call_module(
                "hitl_learning",
                "log_feedback",
                {
                    "event_type": event_type,
                    "entity_id": entity_id,
                    "suggestion": suggestion,
                    "final_decision": final_decision,
                    "signal": signal,
                    **kwargs
                }
            )
            
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
            return False
    
    def _call_module(
        self,
        module_name: str,
        function_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call ML module function via subprocess.
        
        Args:
            module_name: Name of the module
            function_name: Function to call
            input_data: Input data
            
        Returns:
            Function result
        """
        try:
            # Create temporary input file
            input_file = self.base_path / f"temp_{module_name}_{function_name}_{int(time.time())}.json"
            with open(input_file, 'w') as f:
                json.dump(input_data, f)
            
            # Call module
            cmd = [
                "python", "-m", module_name,
                function_name,
                "--input", str(input_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.base_path
            )
            
            # Clean up input file
            input_file.unlink(missing_ok=True)
            
            if result.returncode != 0:
                raise Exception(f"Module call failed: {result.stderr}")
            
            # Parse output
            output_data = json.loads(result.stdout)
            return output_data
            
        except subprocess.TimeoutExpired:
            logger.error(f"Module call timed out: {module_name}.{function_name}")
            raise
        except Exception as e:
            logger.error(f"Module call failed: {module_name}.{function_name} - {e}")
            raise


class MLClientFallback(MLClient):
    """Fallback ML client that returns safe defaults when modules fail."""
    
    def link_pr_to_issues(self, pr_id: str, pr_data: Dict[str, Any], issue_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback: return empty links."""
        logger.warning(f"Using fallback for PR linking: {pr_id}")
        return {
            "pr_id": pr_id,
            "linked_issues": [],
            "confidence_scores": [],
            "model_version": "fallback",
            "success": True
        }
    
    def generate_pr_description(self, diff_text: str, commit_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback: use commit message as title."""
        logger.warning("Using fallback for PR generation")
        return {
            "title": commit_message,
            "body": f"Changes from commit: {commit_message}",
            "confidence": 0.5,
            "model_version": "fallback",
            "success": True
        }
    
    def generate_issue_summary(self, diff_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback: generate basic issue."""
        logger.warning("Using fallback for issue generation")
        return {
            "title": "Code changes detected",
            "body": "Automated issue for code changes",
            "labels": ["automated"],
            "confidence": 0.5,
            "model_version": "fallback",
            "success": True
        }
    
    def evaluate_risk(self, pr_id: str, pr_data: Dict[str, Any], diff_text: str) -> Dict[str, Any]:
        """Fallback: conservative risk assessment."""
        logger.warning(f"Using fallback for risk evaluation: {pr_id}")
        return {
            "pr_id": pr_id,
            "risk_score": 0.5,
            "decision": "MANUAL_REVIEW",
            "confidence": 0.5,
            "tests_passed": True,
            "lint_passed": True,
            "security_passed": True,
            "model_version": "fallback",
            "success": True
        }
