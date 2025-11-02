"""Deduplication utilities."""

from typing import List, Dict, Set, Tuple
from collections import defaultdict
import hashlib
import logging

logger = logging.getLogger(__name__)


def compute_text_hash(title: str, body: str) -> str:
    """
    Compute hash of title and body combined.
    
    Args:
        title: Title text
        body: Body text
        
    Returns:
        MD5 hash string
    """
    combined = f"{title}|{body}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def deduplicate_records(records: List[Dict]) -> List[Dict]:
    """
    Deduplicate records based on (repo, type, id) and text duplicates.
    
    Deduplication strategy:
    1. Drop exact duplicates based on (repo, type, id)
    2. For text duplicates (identical title+body), keep earliest timestamp
    
    Args:
        records: List of record dictionaries
        
    Returns:
        Deduplicated list of records
    """
    if not records:
        return []
    
    # Step 1: Group by (repo, type, id) - exact duplicates
    by_key: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for record in records:
        key = (record['repo'], record['type'], record['id'])
        by_key[key].append(record)
    
    # Keep first occurrence for exact duplicates
    exact_deduped = []
    for key, group in by_key.items():
        if len(group) == 1:
            exact_deduped.append(group[0])
        else:
            # Multiple records with same key - keep first one
            logger.warning(f"Found {len(group)} exact duplicates for {key}, keeping first")
            exact_deduped.append(group[0])
    
    # Step 2: Remove text duplicates (same title+body)
    # Group by text hash
    by_text_hash: Dict[str, List[Dict]] = defaultdict(list)
    for record in exact_deduped:
        title = record.get('title', '')
        body = record.get('body', '')
        text_hash = compute_text_hash(title, body)
        by_text_hash[text_hash].append(record)
    
    text_deduped = []
    for text_hash, group in by_text_hash.items():
        if len(group) == 1:
            text_deduped.append(group[0])
        else:
            # Multiple records with same text - keep earliest timestamp
            group_sorted = sorted(
                group,
                key=lambda r: r.get('timestamp', ''),
                reverse=False  # Earliest first (empty strings sort first)
            )
            logger.warning(f"Found {len(group)} text duplicates, keeping earliest timestamp")
            text_deduped.append(group_sorted[0])
    
    return text_deduped


