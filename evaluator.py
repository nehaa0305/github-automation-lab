"""Evaluation metrics for generated text."""

import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def compute_bleu(reference: str, generated: str) -> float:
    """
    Compute BLEU score.
    
    Args:
        reference: Reference text
        generated: Generated text
        
    Returns:
        BLEU score (0-1)
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu
        
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()
        
        if len(ref_tokens) == 0 or len(gen_tokens) == 0:
            return 0.0
        
        score = sentence_bleu([ref_tokens], gen_tokens)
        return float(score)
    except ImportError:
        logger.warning("NLTK not available, using simple BLEU approximation")
        return compute_simple_bleu(reference, generated)
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        return 0.0


def compute_simple_bleu(reference: str, generated: str) -> float:
    """Simple BLEU approximation using n-gram overlap."""
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()
    
    if len(ref_tokens) == 0 or len(gen_tokens) == 0:
        return 0.0
    
    # Unigram precision
    ref_unigrams = set(ref_tokens)
    gen_unigrams = set(gen_tokens)
    common = ref_unigrams & gen_unigrams
    
    precision = len(common) / len(gen_unigrams) if gen_unigrams else 0.0
    recall = len(common) / len(ref_unigrams) if ref_unigrams else 0.0
    
    # Simple F1-based approximation
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(reference: str, generated: str) -> float:
    """
    Compute ROUGE-L score (Longest Common Subsequence).
    
    Args:
        reference: Reference text
        generated: Generated text
        
    Returns:
        ROUGE-L F1 score (0-1)
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return float(scores['rougeL'].fmeasure)
    except ImportError:
        logger.warning("rouge-score not available, using simple LCS")
        return compute_simple_lcs(reference, generated)
    except Exception as e:
        logger.warning(f"ROUGE-L computation failed: {e}")
        return 0.0


def compute_simple_lcs(reference: str, generated: str) -> float:
    """Simple LCS-based F1 score."""
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()
    
    if len(ref_tokens) == 0 or len(gen_tokens) == 0:
        return 0.0
    
    # Compute LCS length
    lcs_length = _lcs_length(ref_tokens, gen_tokens)
    
    precision = lcs_length / len(gen_tokens) if gen_tokens else 0.0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute LCS length using dynamic programming."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def compute_meteor(reference: str, generated: str) -> float:
    """
    Compute METEOR score.
    
    Args:
        reference: Reference text
        generated: Generated text
        
    Returns:
        METEOR score (0-1)
    """
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
        
        ref_tokens = nltk.word_tokenize(reference.lower())
        gen_tokens = nltk.word_tokenize(generated.lower())
        
        score = meteor_score([ref_tokens], gen_tokens)
        return float(score)
    except ImportError:
        logger.warning("NLTK not available for METEOR, using approximation")
        return compute_rouge_l(reference, generated) * 0.9  # Approximate
    except Exception as e:
        logger.warning(f"METEOR computation failed: {e}")
        return 0.0


def compute_semantic_similarity(reference: str, generated: str) -> float:
    """
    Compute semantic similarity using embeddings.
    
    Args:
        reference: Reference text
        generated: Generated text
        
    Returns:
        Cosine similarity (0-1)
    """
    try:
        from embeddings_index.models import EmbeddingModel
        
        model = EmbeddingModel()
        ref_emb = model.encode_single(reference)
        gen_emb = model.encode_single(generated)
        
        # Cosine similarity (vectors already normalized)
        similarity = np.dot(ref_emb[0], gen_emb[0])
        
        # Normalize to 0-1 range (from -1 to 1)
        return float((similarity + 1) / 2)
    except Exception as e:
        logger.warning(f"Semantic similarity computation failed: {e}")
        return 0.0


def evaluate_generation(
    references: List[str],
    predictions: List[str],
    model_name: str = "Unknown"
) -> Dict[str, Any]:
    """
    Evaluate generated text against references.
    
    Args:
        references: List of reference texts
        predictions: List of generated texts
        model_name: Name of the model
        
    Returns:
        Dictionary with all metrics
    """
    if len(references) != len(predictions):
        logger.warning(f"Mismatch: {len(references)} references vs {len(predictions)} predictions")
        min_len = min(len(references), len(predictions))
        references = references[:min_len]
        predictions = predictions[:min_len]
    
    logger.info(f"Evaluating {len(references)} samples for {model_name}")
    
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    semantic_scores = []
    
    for ref, pred in zip(references, predictions):
        bleu_scores.append(compute_bleu(ref, pred))
        rouge_scores.append(compute_rouge_l(ref, pred))
        meteor_scores.append(compute_meteor(ref, pred))
        semantic_scores.append(compute_semantic_similarity(ref, pred))
    
    metrics = {
        "model": model_name,
        "num_samples": len(references),
        "bleu": {
            "mean": float(np.mean(bleu_scores)),
            "std": float(np.std(bleu_scores)),
            "scores": [float(s) for s in bleu_scores]
        },
        "rouge_l": {
            "mean": float(np.mean(rouge_scores)),
            "std": float(np.std(rouge_scores)),
            "scores": [float(s) for s in rouge_scores]
        },
        "meteor": {
            "mean": float(np.mean(meteor_scores)),
            "std": float(np.std(meteor_scores)),
            "scores": [float(s) for s in meteor_scores]
        },
        "semantic_similarity": {
            "mean": float(np.mean(semantic_scores)),
            "std": float(np.std(semantic_scores)),
            "scores": [float(s) for s in semantic_scores]
        }
    }
    
    logger.info(f"{model_name} - BLEU: {metrics['bleu']['mean']:.4f}, "
                f"ROUGE-L: {metrics['rouge_l']['mean']:.4f}, "
                f"METEOR: {metrics['meteor']['mean']:.4f}, "
                f"Semantic: {metrics['semantic_similarity']['mean']:.4f}")
    
    return metrics






