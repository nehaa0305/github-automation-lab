"""CLI interface for HITL learning system."""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from .api import HITLAPI
from .feedback.schema import FeedbackSchema
from .reports.build import generate_hitl_reports

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ingest_command(args):
    """Ingest feedback from various sources."""
    logger.info("Ingesting feedback...")
    
    api = HITLAPI()
    
    if args.from_file:
        # Check if file exists
        if not Path(args.from_file).exists():
            logger.error(f"File not found: {args.from_file}")
            return
        
        # Ingest from file
        with open(args.from_file, 'r', encoding='utf-8') as f:
            if args.from_file.endswith('.jsonl'):
                # JSONL format
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        api.log_feedback(**data)
                    except Exception as e:
                        logger.warning(f"Failed to parse line: {e}")
            else:
                # JSON format
                data = json.load(f)
                for item in data:
                    try:
                        api.log_feedback(**item)
                    except Exception as e:
                        logger.warning(f"Failed to process item: {e}")
    
    logger.info("Ingestion completed")


def sample_command(args):
    """Sample examples for active learning."""
    logger.info(f"Sampling {args.n} examples using {args.strategy} strategy...")
    
    api = HITLAPI()
    samples = api.sample_active_learning(
        n=args.n,
        strategy=args.strategy,
        entity_type=args.entity_type
    )
    
    # Export for review
    from .active_learning.sampler import ActiveLearningSampler
    sampler = ActiveLearningSampler(api.feedback_store)
    output_path = Path(args.out) / "review_batch.jsonl"
    sampler.export_review_batch(samples, output_path)
    
    logger.info(f"Exported {len(samples)} samples to {output_path}")


def retrain_command(args):
    """Retrain model components."""
    logger.info(f"Retraining components: {args.components}")
    
    api = HITLAPI()
    results = api.schedule_retrain(args.components)
    
    # Save results
    output_path = Path(args.out) / "retraining_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Retraining completed. Results saved to {output_path}")


def calibrate_command(args):
    """Calibrate model thresholds."""
    logger.info("Calibrating model thresholds...")
    
    # This would implement actual calibration
    # For now, create a placeholder
    calibration_results = {
        "linking_model": {"threshold": 0.7, "calibration_score": 0.85},
        "labeling_model": {"threshold": 0.6, "calibration_score": 0.82},
        "rag_model": {"threshold": 0.5, "calibration_score": 0.78},
        "risk_model": {"threshold": 0.3, "calibration_score": 0.88}
    }
    
    output_path = Path(args.out) / "calibration_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    logger.info(f"Calibration completed. Results saved to {output_path}")


def ab_route_command(args):
    """Route requests for A/B testing."""
    logger.info(f"Setting up A/B routing with {args.traffic_split} split...")
    
    api = HITLAPI()
    
    # Example routing
    context = {
        "user_id": "test_user",
        "repo": "test/repo",
        "context_type": args.context
    }
    
    variant = api.route_variant(context)
    logger.info(f"Routed to variant: {variant}")


def report_command(args):
    """Generate HITL reports."""
    logger.info("Generating HITL reports...")
    
    api = HITLAPI()
    
    # Get data for reports
    feedback_stats = api.get_feedback_stats()
    model_versions = api.get_model_versions()
    
    # Generate reports
    report_path = generate_hitl_reports(
        feedback_stats=feedback_stats,
        model_versions=model_versions,
        output_dir=Path(args.out)
    )
    
    logger.info(f"Reports generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Human-in-the-Loop Learning System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest feedback from sources')
    ingest_parser.add_argument('--from', dest='from_file', help='Input file (JSON/JSONL)')
    ingest_parser.add_argument('--out', default='feedback', help='Output directory')
    ingest_parser.set_defaults(func=ingest_command)
    
    # sample command
    sample_parser = subparsers.add_parser('sample', help='Sample examples for active learning')
    sample_parser.add_argument('--strategy', default='uncertainty', 
                              choices=['uncertainty', 'diversity', 'drift', 'mixed'],
                              help='Sampling strategy')
    sample_parser.add_argument('--n', type=int, default=100, help='Number of samples')
    sample_parser.add_argument('--entity-type', help='Filter by entity type')
    sample_parser.add_argument('--out', default='feedback', help='Output directory')
    sample_parser.set_defaults(func=sample_command)
    
    # retrain command
    retrain_parser = subparsers.add_parser('retrain', help='Retrain model components')
    retrain_parser.add_argument('--components', nargs='*', 
                               choices=['linking', 'labeling', 'rag', 'risk'],
                               default=['linking', 'labeling', 'rag', 'risk'],
                               help='Components to retrain (space-separated)')
    retrain_parser.add_argument('--out', default='models/registry', help='Output directory')
    retrain_parser.set_defaults(func=retrain_command)
    
    # calibrate command
    calibrate_parser = subparsers.add_parser('calibrate', help='Calibrate model thresholds')
    calibrate_parser.add_argument('--out', default='models/registry', help='Output directory')
    calibrate_parser.set_defaults(func=calibrate_command)
    
    # ab-route command
    ab_parser = subparsers.add_parser('ab-route', help='Route requests for A/B testing')
    ab_parser.add_argument('--traffic-split', type=float, default=0.9, 
                          help='Traffic split ratio (0-1)')
    ab_parser.add_argument('--context', default='pr', help='Context type')
    ab_parser.set_defaults(func=ab_route_command)
    
    # report command
    report_parser = subparsers.add_parser('report', help='Generate HITL reports')
    report_parser.add_argument('--out', default='hitl_reports', help='Output directory')
    report_parser.set_defaults(func=report_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
