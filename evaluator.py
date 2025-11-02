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






