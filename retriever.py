"""Context retrieval using FAISS indices."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def retrieve_context(
    input_text: str,
    index_dir: Path,
    retrieval_type: str = "pr",
    top_k: int = 5,
    index_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context from FAISS indices.
    
    Args:
        input_text: Input text (PR description, issue text, diff, etc.)
        index_dir: Directory containing FAISS indices
        retrieval_type: Type of retrieval ("pr" or "issue")
        top_k: Number of results to retrieve
        index_name: Specific index name (defaults based on retrieval_type)
        
    Returns:
        List of retrieved records with metadata and scores
    """
    from embeddings_index.search import IndexSearcher
    
    logger.info(f"Retrieving context for type '{retrieval_type}' with top_k={top_k}")
    
    # Map retrieval type to index name
    if index_name is None:
        if retrieval_type.lower() in ["pr", "pull_request", "pullrequest"]:
            index_name = "pr_issues"
        elif retrieval_type.lower() in ["issue", "issues"]:
            index_name = "pr_issues"  # Same index for now
        elif retrieval_type.lower() == "commit":
            index_name = "commits"
        elif retrieval_type.lower() == "code":
            index_name = "code"
        else:
            index_name = "pr_issues"
    
    try:
        searcher = IndexSearcher(index_dir)
        # Primary results (PRs/issues/commits based on retrieval_type)
        results = searcher.search(input_text, index_name, top_k)
        logger.info(f"Retrieved {len(results)} results from {index_name} index")

        # Also fetch code context to ground templates in repository code
        code_results: List[Dict[str, Any]] = []
        try:
            code_results = searcher.search(input_text, "code", min(top_k, 5))
            logger.info(f"Retrieved {len(code_results)} code context results")
        except Exception as e:
            logger.warning(f"Code context retrieval failed: {e}")

        # Format results with retrieval metadata
        formatted_results: List[Dict[str, Any]] = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "score": result.get("score", 0.0),
                "repo": result.get("repo", "unknown"),
                "id": result.get("record_id", "unknown"),
                "type": result.get("type", "pr_or_issue"),
                "title": result.get("title", ""),
                "text_preview": result.get("text_preview", ""),
                "language": result.get("language", ""),
                "source_dataset": result.get("source_dataset", "")
            })
        # Append code items with explicit type
        offset = len(formatted_results)
        for j, result in enumerate(code_results, 1):
            formatted_results.append({
                "rank": offset + j,
                "score": result.get("score", 0.0),
                "repo": result.get("repo", "unknown"),
                "id": result.get("record_id", "unknown"),
                "type": "code",
                "title": result.get("title", result.get("path", "")),
                "text_preview": result.get("text_preview", ""),
                "language": result.get("language", ""),
                "source_dataset": result.get("source_dataset", "")
            })

        return formatted_results
    
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []
