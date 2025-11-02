"""Command-line interface for dataset builder."""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_command(args):
    """Build unified dataset from input files."""
    from .builder import build_dataset
    
    logger.info("Starting dataset build...")
    
    try:
        build_dataset(
            pr_issues_path=args.pr_issues,
            commits_path=args.commits,
            code_path=args.code,
            issue_labels_path=args.issue_labels,
            output_dir=args.out
        )
        logger.info("Dataset build completed successfully")
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)


def validate_command(args):
    """Validate dataset."""
    from .utils.validate import validate_dataset, generate_report
    
    logger.info("Validating dataset...")
    
    dataset_dir = Path(args.input)
    report = validate_dataset(dataset_dir)
    
    # Generate markdown report
    report_path = dataset_dir / 'REPORT.md'
    generate_report(report, report_path)
    
    logger.info(f"Validation complete. Report saved to {report_path}")
    
    # Exit with error if there are issues
    if report['errors']:
        logger.error(f"Validation found {len(report['errors'])} errors")
        sys.exit(1)


def sample_command(args):
    """Create stratified sample dataset."""
    logger.info("Creating sample dataset...")
    logger.warning("Sample command not yet implemented")
    # TODO: Implement sampling


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Dataset Builder for GitHub ML/NLP Project',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build unified dataset')
    build_parser.add_argument('--pr-issues', type=Path, help='Path to PR/issues JSONL file')
    build_parser.add_argument('--commits', type=Path, help='Path to commits JSONL file')
    build_parser.add_argument('--code', type=Path, help='Path to code snippets JSONL file')
    build_parser.add_argument('--issue-labels', type=Path, help='Path to issue labels JSONL file')
    build_parser.add_argument('--out', type=Path, default='normalized_dataset', 
                            help='Output directory (default: normalized_dataset)')
    build_parser.set_defaults(func=build_command)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('--in', dest='input', type=Path, default='normalized_dataset',
                               help='Input directory (default: normalized_dataset)')
    validate_parser.set_defaults(func=validate_command)
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Create sample dataset')
    sample_parser.add_argument('--in', dest='input', type=Path, required=True, help='Input directory')
    sample_parser.add_argument('--n', type=int, default=200, help='Number of samples per type')
    sample_parser.add_argument('--out', type=Path, required=True, help='Output directory')
    sample_parser.set_defaults(func=sample_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()











