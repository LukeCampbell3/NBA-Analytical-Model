"""
Metrics tracking for production monitoring.
"""

from typing import Dict, List
from datetime import datetime
import json
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """
    Tracks system metrics over time.
    
    Tracks:
    - Inference latency
    - Throughput
    - Error rates
    - Model performance
    """
    
    def __init__(self, metrics_dir: str = "logs/metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.current_metrics = []
    
    def record_metric(self, metric_name: str, value: float, 
                     tags: Dict = None):
        """Record a metric value."""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'name': metric_name,
            'value': value,
            'tags': tags or {}
        }
        self.current_metrics.append(metric)
    
    def flush_metrics(self):
        """Flush metrics to file."""
        if not self.current_metrics:
            return
        
        metrics_file = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(metrics_file, 'a') as f:
            for metric in self.current_metrics:
                f.write(json.dump(metric) + '\n')
        
        self.current_metrics = []
        logger.debug(f"Flushed metrics to {metrics_file}")
