"""Dataset builder orchestrator."""

import json
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

from .loaders import (
    load_pr_issues,
    load_commits,
    load_code_snippets,
    load_issue_labels
)
from .utils import (
    normalize_record,
    deduplicate_records,
    assign_split,
    save_jsonl
)

logger = logging.getLogger(__name__)


def build_dataset(
    pr_issues_path: Optional[Path] = None,
    commits_path: Optional[Path] = None,
    code_path: Optional[Path] = None,
    issue_labels_path: Optional[Path] = None,
    output_dir: Path = Path('normalized_dataset')
) -> None:
    """
    Build unified dataset from multiple input sources.
    
    Args:
        pr_issues_path: Path to PR/issues JSONL file
        commits_path: Path to commits JSONL file
        code_path: Path to code snippets JSONL file
        issue_labels_path: Path to issue labels JSONL file
        output_dir: Output directory for normalized files
    """
    logger.info("="*60)
    logger.info("DATASET BUILDER")
    logger.info("="*60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all records
    all_records = defaultdict(list)
    
    # Load PR/Issues
    if pr_issues_path:
        logger.info(f"Loading PR/Issues from {pr_issues_path}")
        records = load_pr_issues(pr_issues_path, source_dataset="pr_issues")
        for record in records:
            # Determine if PR or issue
            record_type = 'pr' if record.get('pull_request') else 'issue'
            normalized = normalize_record(record, record_type, record.get('_source', 'pr_issues'))
            normalized['split'] = assign_split(normalized['repo'], normalized['id'])
            all_records[record_type].append(normalized)
    
    # Load Commits
    if commits_path:
        logger.info(f"Loading Commits from {commits_path}")
        records = load_commits(commits_path, source_dataset="commits")
        for record in records:
            normalized = normalize_record(record, 'commit', record.get('_source', 'commits'))
            normalized['split'] = assign_split(normalized['repo'], normalized['id'])
            all_records['commit'].append(normalized)
    
    # Load Code Snippets
    if code_path:
        logger.info(f"Loading Code from {code_path}")
        records = load_code_snippets(code_path, source_dataset="code_snippets")
        for record in records:
            normalized = normalize_record(record, 'code', record.get('_source', 'code_snippets'))
            normalized['split'] = assign_split(normalized['repo'], normalized['id'])
            all_records['code'].append(normalized)
    
    # Load Issue Labels
    if issue_labels_path:
        logger.info(f"Loading Issue Labels from {issue_labels_path}")
        records = load_issue_labels(issue_labels_path, source_dataset="issue_labels")
        for record in records:
            normalized = normalize_record(record, 'issue', record.get('_source', 'issue_labels'))
            normalized['split'] = assign_split(normalized['repo'], normalized['id'])
            all_records['issue_with_labels'].append(normalized)
    
    # Deduplicate and save
    logger.info("\n" + "="*60)
    logger.info("DEDUPLICATING AND SAVING")
    logger.info("="*60)
    
    # Save PR and Issues combined
    if all_records['pr'] or all_records['issue']:
        pr_issues = deduplicate_records(all_records['pr'] + all_records['issue'])
        save_jsonl(pr_issues, output_dir / 'pr_issues.jsonl')
        logger.info(f"Saved {len(pr_issues)} PR/Issue records")
    
    # Save Commits
    if all_records['commit']:
        commits = deduplicate_records(all_records['commit'])
        save_jsonl(commits, output_dir / 'commits.jsonl')
        logger.info(f"Saved {len(commits)} commit records")
    
    # Save Code Snippets
    if all_records['code']:
        code_snippets = deduplicate_records(all_records['code'])
        save_jsonl(code_snippets, output_dir / 'code_snippets.jsonl')
        logger.info(f"Saved {len(code_snippets)} code snippet records")
    
    # Save Issue Labels (only issues with non-empty labels)
    if all_records['issue_with_labels']:
        issue_labels = [r for r in all_records['issue_with_labels'] if r.get('labels')]
        issue_labels = deduplicate_records(issue_labels)
        save_jsonl(issue_labels, output_dir / 'issue_labels.jsonl')
        logger.info(f"Saved {len(issue_labels)} labeled issue records")
    
    # Generate catalog
    generate_catalog(all_records, output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("DATASET BUILD COMPLETE")
    logger.info("="*60)


def generate_catalog(all_records: dict, output_dir: Path) -> None:
    """Generate catalog.json with statistics."""
    catalog = {
        'schema': {
            'description': 'Unified GitHub dataset schema',
            'version': '1.0'
        },
        'statistics': {}
    }
    
    stats = catalog['statistics']
    
    # Count by type
    for record_type, records in all_records.items():
        if records:
            stats[record_type] = {
                'count': len(records),
                'with_labels': sum(1 for r in records if r.get('labels')),
                'with_timestamps': sum(1 for r in records if r.get('timestamp'))
            }
    
    # Save catalog
    catalog_path = output_dir / 'catalog.json'
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    logger.info(f"Generated catalog at {catalog_path}")











