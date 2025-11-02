"""Feedback schema and data structures."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json


@dataclass
class FeedbackEntry:
    """Structured feedback entry."""
    
    ts: str  # ISO-8601 timestamp
    repo: str  # owner/repo
    entity_type: str  # pr|issue|link|label|merge_policy
    entity_id: str  # unique identifier
    suggestion: Dict[str, Any]  # model output snapshot
    final_decision: Dict[str, Any]  # human-edited result or action
    signal: str  # accept|edit|reject|approve|request_changes|merge|revert
    edit_distance: float  # text edit ratio 0..1
    confidence: float  # model confidence at time
    model_versions: Dict[str, str]  # linking, labeling, rag, risk versions
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format."""
        return json.dumps(asdict(self), ensure_ascii=False)
    
    @classmethod
    def from_jsonl(cls, line: str) -> 'FeedbackEntry':
        """Create from JSONL line."""
        data = json.loads(line.strip())
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FeedbackSchema:
    """Schema validation and utilities."""
    
    ENTITY_TYPES = {"pr", "issue", "link", "label", "merge_policy"}
    SIGNALS = {
        "accept", "edit", "reject", "approve", "request_changes", 
        "merge", "revert", "ignore", "timeout"
    }
    
    @classmethod
    def validate_entry(cls, entry: FeedbackEntry) -> bool:
        """Validate feedback entry."""
        if entry.entity_type not in cls.ENTITY_TYPES:
            return False
        
        if entry.signal not in cls.SIGNALS:
            return False
        
        if not (0.0 <= entry.edit_distance <= 1.0):
            return False
        
        if not (0.0 <= entry.confidence <= 1.0):
            return False
        
        return True
    
    @classmethod
    def create_entry(
        cls,
        repo: str,
        entity_type: str,
        entity_id: str,
        suggestion: Dict[str, Any],
        final_decision: Dict[str, Any],
        signal: str,
        edit_distance: float = 0.0,
        confidence: float = 1.0,
        model_versions: Optional[Dict[str, str]] = None
    ) -> FeedbackEntry:
        """Create validated feedback entry."""
        
        if model_versions is None:
            model_versions = {}
        
        entry = FeedbackEntry(
            ts=datetime.utcnow().isoformat() + "Z",
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
        
        if not cls.validate_entry(entry):
            raise ValueError(f"Invalid feedback entry: {entry}")
        
        return entry
