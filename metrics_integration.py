"""Metrics integration for ML components."""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MLMetric:
    """ML metric definition."""
    name: str
    value: float
    timestamp: str
    labels: Dict[str, str]
    metric_type: str = "gauge"  # gauge, counter, histogram


class MetricsCollector:
    """Collects and stores ML metrics."""
    
    def __init__(self):
        self.metrics = []
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: str = "gauge"
    ) -> None:
        """Record a metric."""
        metric = MLMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow().isoformat(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        self.metrics.append(metric)
        
        # Update internal stores
        if metric_type == "counter":
            self.counters[name] = self.counters.get(name, 0) + value
        elif metric_type == "gauge":
            self.gauges[name] = value
        elif metric_type == "histogram":
            if name not in self.histograms:
                self.histograms[name] = []
            self.histograms[name].append(value)
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, 1.0, labels, "counter")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        self.record_metric(name, value, labels, "gauge")
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value."""
        self.record_metric(name, value, labels, "histogram")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_metrics": len(self.metrics),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {name: len(values) for name, values in self.histograms.items()},
            "timestamp": datetime.utcnow().isoformat()
        }


class MLMetricsIntegration:
    """Integrates ML metrics with existing monitoring."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.start_times = {}
    
    def record_pr_generation_metrics(
        self,
        success: bool,
        latency_ms: float,
        confidence: float,
        model_version: str
    ) -> None:
        """Record PR generation metrics."""
        labels = {
            "model_version": model_version,
            "success": str(success)
        }
        
        self.metrics_collector.increment_counter("ml_pr_gen_total", labels)
        self.metrics_collector.record_histogram("ml_pr_gen_latency_ms", latency_ms, labels)
        self.metrics_collector.record_histogram("ml_pr_gen_confidence", confidence, labels)
    
    def record_issue_generation_metrics(
        self,
        success: bool,
        latency_ms: float,
        confidence: float,
        model_version: str
    ) -> None:
        """Record issue generation metrics."""
        labels = {
            "model_version": model_version,
            "success": str(success)
        }
        
        self.metrics_collector.increment_counter("ml_issue_gen_total", labels)
        self.metrics_collector.record_histogram("ml_issue_gen_latency_ms", latency_ms, labels)
        self.metrics_collector.record_histogram("ml_issue_gen_confidence", confidence, labels)
    
    def record_linking_metrics(
        self,
        success: bool,
        latency_ms: float,
        precision: float,
        recall: float,
        model_version: str
    ) -> None:
        """Record linking metrics."""
        labels = {
            "model_version": model_version,
            "success": str(success)
        }
        
        self.metrics_collector.increment_counter("ml_linking_total", labels)
        self.metrics_collector.record_histogram("ml_linking_latency_ms", latency_ms, labels)
        self.metrics_collector.record_histogram("ml_linking_precision", precision, labels)
        self.metrics_collector.record_histogram("ml_linking_recall", recall, labels)
    
    def record_risk_evaluation_metrics(
        self,
        success: bool,
        latency_ms: float,
        risk_score: float,
        decision: str,
        model_version: str
    ) -> None:
        """Record risk evaluation metrics."""
        labels = {
            "model_version": model_version,
            "success": str(success),
            "decision": decision
        }
        
        self.metrics_collector.increment_counter("ml_risk_eval_total", labels)
        self.metrics_collector.record_histogram("ml_risk_eval_latency_ms", latency_ms, labels)
        self.metrics_collector.record_histogram("ml_risk_score", risk_score, labels)
    
    def record_feedback_metrics(
        self,
        signal: str,
        entity_type: str,
        success: bool
    ) -> None:
        """Record feedback metrics."""
        labels = {
            "signal": signal,
            "entity_type": entity_type,
            "success": str(success)
        }
        
        self.metrics_collector.increment_counter("ml_feedback_total", labels)
    
    def record_retraining_metrics(
        self,
        success: bool,
        duration_seconds: float,
        models_retrained: List[str]
    ) -> None:
        """Record retraining metrics."""
        labels = {
            "success": str(success),
            "models_count": str(len(models_retrained))
        }
        
        self.metrics_collector.increment_counter("ml_retrain_total", labels)
        self.metrics_collector.record_histogram("ml_retrain_duration_seconds", duration_seconds, labels)
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration in milliseconds."""
        if operation not in self.start_times:
            return 0.0
        
        duration_ms = (time.time() - self.start_times[operation]) * 1000
        del self.start_times[operation]
        return duration_ms
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        lines = []
        
        # Counters
        for name, value in self.metrics_collector.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in self.metrics_collector.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Histograms (simplified)
        for name, values in self.metrics_collector.histograms.items():
            if values:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {len(values)}")
                lines.append(f"{name}_sum {sum(values)}")
                lines.append(f"{name}_avg {sum(values) / len(values)}")
        
        return "\n".join(lines)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for system health report."""
        summary = self.metrics_collector.get_metrics_summary()
        
        # Calculate health indicators
        total_operations = sum(self.metrics_collector.counters.values())
        success_rate = 0.0
        
        if total_operations > 0:
            success_operations = sum(
                count for name, count in self.metrics_collector.counters.items()
                if "success=true" in name or "success=True" in name
            )
            success_rate = success_operations / total_operations
        
        return {
            "ml_operations_total": total_operations,
            "ml_success_rate": success_rate,
            "ml_metrics_count": summary["total_metrics"],
            "ml_models_active": len(set(
                metric.labels.get("model_version", "unknown")
                for metric in self.metrics_collector.metrics
            )),
            "timestamp": datetime.utcnow().isoformat()
        }
