"""
Example: Using the Issue Labeling Dataset

This example demonstrates how to load and use the issue labeling dataset
for training classification models.
"""

import json
from pathlib import Path
from typing import List, Dict
from collections import Counter

def load_issue_labeling_data(file_path: str) -> List[Dict]:
    """Load issue labeling data from JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    return records

def analyze_labels(records: List[Dict]):
    """Analyze label distribution in the dataset."""
    all_labels = []
    for record in records:
        labels = record.get('labels', [])
        all_labels.extend(labels)
    
    label_counts = Counter(all_labels)
    
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION")
    print("="*60)
    print(f"Total unique labels: {len(label_counts)}")
    print(f"Total label instances: {len(all_labels)}")
    print("\nTop 20 most common labels:")
    for label, count in label_counts.most_common(20):
        print(f"  {label}: {count}")
    print("="*60)

def filter_by_labels(records: List[Dict], target_labels: List[str]) -> List[Dict]:
    """Filter records that contain any of the target labels."""
    filtered = []
    for record in records:
        labels = record.get('labels', [])
        if any(label in labels for label in target_labels):
            filtered.append(record)
    return filtered

def get_label_statistics(records: List[Dict]):
    """Get statistics about labels in the dataset."""
    stats = {
        'total_issues': len(records),
        'issues_with_labels': sum(1 for r in records if r.get('labels')),
        'issues_without_labels': sum(1 for r in records if not r.get('labels')),
        'average_labels_per_issue': 0,
        'most_common_labels': []
    }
    
    all_labels = []
    for record in records:
        labels = record.get('labels', [])
        all_labels.extend(labels)
    
    if len(records) > 0:
        stats['average_labels_per_issue'] = len(all_labels) / len(records)
    
    label_counts = Counter(all_labels)
    stats['most_common_labels'] = label_counts.most_common(10)
    
    return stats

def print_sample_labeled_issue(record: Dict):
    """Print a sample labeled issue."""
    print("\n" + "="*60)
    print("SAMPLE LABELED ISSUE")
    print("="*60)
    print(f"Repository: {record.get('repo', 'N/A')}")
    print(f"Issue ID: {record.get('id', 'N/A')}")
    print(f"Type: {record.get('type', 'N/A')}")
    print(f"\nTitle: {record.get('title', 'N/A')[:100]}")
    print(f"\nBody: {record.get('body', 'N/A')[:200]}...")
    print(f"\nLabels: {', '.join(record.get('labels', []))}")
    print(f"Source Dataset: {record.get('source_dataset', 'N/A')}")
    print("="*60)

def main():
    """Main example function."""
    # Path to the issue labeling dataset
    dataset_path = Path('normalized_dataset/issue_labeling.jsonl')
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please run the dataset processor first:")
        print("  python -m app.unified_dataset_processor")
        return
    
    # Load the data
    print("Loading issue labeling dataset...")
    records = load_issue_labeling_data(dataset_path)
    print(f"Loaded {len(records)} labeled issues")
    
    # Analyze labels
    analyze_labels(records)
    
    # Get statistics
    stats = get_label_statistics(records)
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total issues: {stats['total_issues']}")
    print(f"Issues with labels: {stats['issues_with_labels']}")
    print(f"Issues without labels: {stats['issues_without_labels']}")
    print(f"Average labels per issue: {stats['average_labels_per_issue']:.2f}")
    print("="*60)
    
    # Filter by specific labels
    bug_issues = filter_by_labels(records, ['bug', 'error', 'fix'])
    print(f"\nFound {len(bug_issues)} issues with bug-related labels")
    
    enhancement_issues = filter_by_labels(records, ['enhancement', 'feature', 'improvement'])
    print(f"Found {len(enhancement_issues)} issues with enhancement-related labels")
    
    # Show a sample
    if records:
        print_sample_labeled_issue(records[0])

if __name__ == '__main__':
    main()


