"""Retrieval model for PR-issue linking using FAISS."""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss

from embeddings_index.search import IndexSearcher
from embeddings_index.models import EmbeddingModel

logger = logging.getLogger(__name__)


class RetrievalModel:
    """FAISS-based retrieval model for PR-issue linking."""
    
    def __init__(self, index_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize retrieval model.
        
        Args:
            index_dir: Directory containing FAISS indices
            model_name: Name of the embedding model
        """
        self.index_dir = index_dir
        self.searcher = IndexSearcher(index_dir, model_name)
        self.model = EmbeddingModel(model_name)
    
    def retrieve(self, query_text: str, top_k: int = 10, 
                 index_type: str = "pr_issues") -> List[Dict[str, Any]]:
        """
        Retrieve similar records for a query.
        
        Args:
            query_text: Query text
            top_k: Number of results to retrieve
            index_type: Type of index to search
            
        Returns:
            List of retrieved records with scores
        """
        try:
            results = self.searcher.search(query_text, index_type, top_k)
            logger.info(f"Retrieved {len(results)} results for query: {query_text[:50]}")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def retrieve_for_pr(self, pr_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve related issues for a PR.
        
        Args:
            pr_text: PR title and body
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved issues with scores
        """
        return self.retrieve(pr_text, top_k, "pr_issues")
    
    def retrieve_for_issue(self, issue_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve related PRs for an issue.
        
        Args:
            issue_text: Issue title and body
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved PRs with scores
        """
        return self.retrieve(issue_text, top_k, "pr_issues")
