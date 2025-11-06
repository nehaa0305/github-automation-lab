"""Dashboard hooks for ML components."""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from .metrics_integration import MLMetricsIntegration
from .policy_integration import PolicyManager

logger = logging.getLogger(__name__)


class DashboardHooks:
    """Dashboard hooks for ML components."""
    
    def __init__(
        self,
        metrics_integration: Optional[MLMetricsIntegration] = None,
        policy_manager: Optional[PolicyManager] = None
    ):
        self.metrics_integration = metrics_integration or MLMetricsIntegration()
        self.policy_manager = policy_manager or PolicyManager()
    
    def get_ml_dashboard_data(self) -> Dict[str, Any]:
        """Get ML dashboard data."""
        return {
            "model_versions": self._get_model_versions(),
            "recent_metrics": self._get_recent_metrics(),
            "feedback_trends": self._get_feedback_trends(),
            "policy_status": self._get_policy_status(),
            "risk_distribution": self._get_risk_distribution(),
            "performance_summary": self._get_performance_summary()
        }
    
    def _get_model_versions(self) -> Dict[str, Any]:
        """Get current model versions."""
        # This would query the model registry
        return {
            "linking_model": "v1.2.0",
            "labeling_model": "v2.1.0",
            "rag_model": "v1.5.0",
            "risk_model": "v1.0.0"
        }
    
    def _get_recent_metrics(self) -> Dict[str, Any]:
        """Get recent metrics data."""
        summary = self.metrics_integration.metrics_collector.get_metrics_summary()
        
        return {
            "total_operations": sum(self.metrics_integration.metrics_collector.counters.values()),
            "success_rate": self._calculate_success_rate(),
            "avg_latency_ms": self._calculate_avg_latency(),
            "active_models": len(set(
                metric.labels.get("model_version", "unknown")
                for metric in self.metrics_integration.metrics_collector.metrics
            ))
        }
    
    def _get_feedback_trends(self) -> Dict[str, Any]:
        """Get feedback trends data."""
        # This would query the feedback store
        return {
            "total_feedback": 1250,
            "accept_rate": 0.78,
            "edit_rate": 0.15,
            "reject_rate": 0.07,
            "trend_7_days": [0.75, 0.76, 0.78, 0.79, 0.77, 0.78, 0.78]
        }
    
    def _get_policy_status(self) -> Dict[str, Any]:
        """Get policy status."""
        policy_summary = self.policy_manager.get_policy_summary()
        
        return {
            "risk_thresholds": policy_summary["risk_thresholds"],
            "ml_features_enabled": policy_summary["ml_features_enabled"],
            "retraining_enabled": policy_summary["retraining_enabled"],
            "ab_testing_enabled": policy_summary["ab_testing_enabled"]
        }
    
    def _get_risk_distribution(self) -> Dict[str, Any]:
        """Get risk score distribution."""
        # This would query recent risk evaluations
        return {
            "auto_merge": 45,
            "manual_review": 35,
            "block_merge": 20,
            "avg_risk_score": 0.42
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "pr_generation_success_rate": 0.92,
            "issue_generation_success_rate": 0.88,
            "linking_precision": 0.85,
            "linking_recall": 0.82,
            "risk_evaluation_accuracy": 0.89
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        counters = self.metrics_integration.metrics_collector.counters
        total = sum(counters.values())
        
        if total == 0:
            return 0.0
        
        success = sum(
            count for name, count in counters.items()
            if "success=true" in name or "success=True" in name
        )
        
        return success / total
    
    def _calculate_avg_latency(self) -> float:
        """Calculate average latency."""
        histograms = self.metrics_integration.metrics_collector.histograms
        latency_values = []
        
        for name, values in histograms.items():
            if "latency" in name:
                latency_values.extend(values)
        
        if not latency_values:
            return 0.0
        
        return sum(latency_values) / len(latency_values)
    
    def generate_dashboard_html(self, output_path: Path) -> Path:
        """Generate HTML dashboard."""
        data = self.get_ml_dashboard_data()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ML/NLP Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-card {{ 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            padding: 20px; 
            margin: 10px 0; 
            background: #f9f9f9;
        }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-bottom: 10px; }}
        .chart-container {{ width: 400px; height: 300px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>ML/NLP Automation Dashboard</h1>
    <p>Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="metric-card">
        <div class="metric-label">Total Operations</div>
        <div class="metric-value">{data['recent_metrics']['total_operations']}</div>
    </div>
    
    <div class="metric-card">
        <div class="metric-label">Success Rate</div>
        <div class="metric-value">{data['recent_metrics']['success_rate']:.1%}</div>
    </div>
    
    <div class="metric-card">
        <div class="metric-label">Average Latency</div>
        <div class="metric-value">{data['recent_metrics']['avg_latency_ms']:.0f}ms</div>
    </div>
    
    <div class="metric-card">
        <div class="metric-label">Active Models</div>
        <div class="metric-value">{data['recent_metrics']['active_models']}</div>
    </div>
    
    <h2>Model Versions</h2>
    <ul>
        {''.join(f'<li><strong>{name}:</strong> {version}</li>' for name, version in data['model_versions'].items())}
    </ul>
    
    <h2>Feedback Trends</h2>
    <div class="metric-card">
        <div class="metric-label">Accept Rate</div>
        <div class="metric-value">{data['feedback_trends']['accept_rate']:.1%}</div>
    </div>
    
    <h2>Risk Distribution</h2>
    <div class="metric-card">
        <div class="metric-label">Auto Merge</div>
        <div class="metric-value">{data['risk_distribution']['auto_merge']}%</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Manual Review</div>
        <div class="metric-value">{data['risk_distribution']['manual_review']}%</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Block Merge</div>
        <div class="metric-value">{data['risk_distribution']['block_merge']}%</div>
    </div>
    
    <h2>Performance Summary</h2>
    <ul>
        <li>PR Generation Success Rate: {data['performance_summary']['pr_generation_success_rate']:.1%}</li>
        <li>Issue Generation Success Rate: {data['performance_summary']['issue_generation_success_rate']:.1%}</li>
        <li>Linking Precision: {data['performance_summary']['linking_precision']:.1%}</li>
        <li>Linking Recall: {data['performance_summary']['linking_recall']:.1%}</li>
        <li>Risk Evaluation Accuracy: {data['performance_summary']['risk_evaluation_accuracy']:.1%}</li>
    </ul>
    
    <script>
        // Auto-refresh every 5 minutes
        setTimeout(() => location.reload(), 300000);
    </script>
</body>
</html>
        """
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated dashboard HTML: {output_path}")
        return output_path
    
    def get_api_endpoints(self) -> List[Dict[str, Any]]:
        """Get API endpoints for ML dashboard."""
        return [
            {
                "path": "/api/ml/dashboard",
                "method": "GET",
                "description": "Get ML dashboard data"
            },
            {
                "path": "/api/ml/metrics",
                "method": "GET",
                "description": "Get ML metrics in Prometheus format"
            },
            {
                "path": "/api/ml/models",
                "method": "GET",
                "description": "Get model versions and status"
            },
            {
                "path": "/api/ml/feedback",
                "method": "GET",
                "description": "Get feedback trends"
            },
            {
                "path": "/api/ml/policy",
                "method": "GET",
                "description": "Get policy configuration"
            }
        ]
