"""Record normalization utilities."""

from datetime import datetime
from typing import Any, Dict, Optional
import re

from .language import detect_language, detect_language_from_diff


def normalize_timestamp(ts: Any) -> Optional[str]:
    """
    Convert timestamp to ISO 8601 format.
    
    Args:
        ts: Timestamp in various formats
        
    Returns:
        ISO 8601 string or None
    """
    if not ts:
        return None
    
    # Already ISO format
    if isinstance(ts, str):
        if 'T' in ts or ts.startswith('20'):
            return ts
        
        # Try to parse common formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']:
            try:
                dt = datetime.strptime(ts, fmt)
                return dt.isoformat() + 'Z'
            except ValueError:
                continue
    
    # Unix timestamp
    if isinstance(ts, (int, float)):
        try:
            dt = datetime.fromtimestamp(ts)
            return dt.isoformat() + 'Z'
        except (ValueError, OSError):
            return None
    
    return None


def extract_linked_ids(text: str) -> list:
    """
    Extract linked issue/PR IDs from text.
    
    Args:
        text: Text to search
        
    Returns:
        List of linked IDs
    """
    if not text:
        return []
    
    patterns = [
        r'#(\d+)',
        r'(?:fixes?|closes?|resolves?|relates? to)\s+#(\d+)',
        r'(?:see|ref)\s+#(\d+)',
    ]
    
    linked_ids = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Handle tuple from groups
            match = match if isinstance(match, str) else match[0] if match else None
            if match:
                linked_ids.append(match)
    
    return list(set(linked_ids))


def trim_diff(diff: str, max_chars: int = 50000, context: int = 20000) -> str:
    """
    Trim large diffs keeping start and end context.
    
    Args:
        diff: Diff content
        max_chars: Maximum character limit
        context: Characters to keep from start/end
        
    Returns:
        Trimmed diff string
    """
    if not diff or len(diff) <= max_chars:
        return diff
    
    start = diff[:context]
    end = diff[-context:]
    
    return f"{start}\n\n...<truncated {len(diff) - 2*context} characters>...\n\n{end}"


def normalize_record(record: Dict[str, Any], record_type: str, source_dataset: str) -> Dict[str, Any]:
    """
    Normalize a record to unified schema.
    
    Args:
        record: Raw record dictionary
        record_type: Type of record ('pr', 'issue', 'commit', 'code')
        source_dataset: Source dataset name
        
    Returns:
        Normalized record dictionary
    """
    normalized = {
        'repo': normalize_repo(record),
        'type': record_type,
        'id': str(record.get('number', record.get('id', record.get('sha', '')))),
        'title': record.get('title', record.get('subject', '')),
        'body': record.get('body', record.get('description', record.get('message', ''))),
        'linked_ids': extract_linked_ids(f"{record.get('title', '')} {record.get('body', '')}"),
        'commit_message': record.get('message', record.get('commit_message', '')),
        'diff': trim_diff(record.get('diff', record.get('patch', ''))),
        'filepath': record.get('file_path', record.get('path', record.get('filename', ''))),
        'code': record.get('code', record.get('content', record.get('snippet', ''))),
        'labels': normalize_labels(record.get('labels', [])),
        'language': detect_language_for_record(record, record_type),
        'timestamp': normalize_timestamp(record.get('created_at', record.get('date', record.get('timestamp')))),
        'source_dataset': source_dataset,
        'license': record.get('license', record.get('license_name', None)),
        'split': None  # Will be assigned later
    }
    
    return normalized


def normalize_repo(record: Dict[str, Any]) -> str:
    """Extract and normalize repository name."""
    # Try various field names
    for field in ['repo', 'repository', 'repo_name', 'full_name', 'repo_url']:
        if field in record and record[field]:
            repo = str(record[field])
            # Extract from URL if needed
            if 'github.com' in repo:
                match = re.search(r'github\.com/([^/]+/[^/]+)', repo)
                if match:
                    return match.group(1)
            if '/' in repo:
                return repo
    
    # Try to extract from URL fields
    for field in ['url', 'html_url', 'web_url']:
        if field in record and record[field]:
            match = re.search(r'github\.com/([^/]+/[^/]+)', str(record[field]))
            if match:
                return match.group(1)
    
    return 'unknown/unknown'


def normalize_labels(labels: Any) -> list:
    """Normalize labels to a list of strings."""
    if not labels:
        return []
    
    if isinstance(labels, list):
        return [str(l.get('name', l) if isinstance(l, dict) else l).lower().strip() 
                for l in labels if l]
    
    if isinstance(labels, str):
        return [l.strip().lower() for l in labels.split(',') if l.strip()]
    
    return [str(labels).lower().strip()]


def detect_language_for_record(record: Dict[str, Any], record_type: str) -> str:
    """Detect language for a record based on its type and content."""
    if record_type == 'code':
        return detect_language(record.get('file_path', record.get('path', record.get('filename'))))
    elif record_type == 'commit':
        diff = record.get('diff', record.get('patch', ''))
        return detect_language_from_diff(diff)
    else:
        return detect_language(record.get('file_path'))
