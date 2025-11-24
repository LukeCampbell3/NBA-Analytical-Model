"""
Drift monitoring for data and model performance.

Detects population shift, covariance shift, and calibration drift.
"""

from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftMonitor:
    """
    Monitors data and model drift.
    
    Implements:
    - Population shift detection (PSI)
    - Covariance shift detection
    - Calibration drift detection
    - Auto-refresh triggers
    """
    
    def __init__(self, config: Dict, baseline_dir: str = "artifacts/baselines"):
        """
        Initialize drift monitor.
        
        Args:
            config: Configuration dictionary with thresholds
            baseline_dir: Directory for baseline statistics
        """
        self.config = config
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.thresholds = config.get('drift_management', {}).get('monitors', {})
        self.actions = config.get('drift_management', {}).get('actions', {})
        self.action_callbacks = {}
        
        logger.info("Initialized DriftMonitor")
    
    def register_action(self, action_name: str, callback: Callable):
        """
        Register callback for drift action.
        
        Args:
            action_name: Name of action (e.g., 'frontier_refit')
            callback: Function to call when action is triggered
        """
        self.action_callbacks[action_name] = callback
        logger.info(f"Registered action callback: {action_name}")
    
    def trigger_action(self, action_name: str, **kwargs):
        """
        Trigger a drift action.
        
        Args:
            action_name: Name of action to trigger
            **kwargs: Additional arguments for callback
        """
        if action_name in self.action_callbacks:
            logger.warning(f"Triggering action: {action_name}")
            self.action_callbacks[action_name](**kwargs)
        else:
            logger.warning(f"No callback registered for action: {action_name}")
    
    def calculate_psi(self, baseline: np.ndarray, 
                     current: np.ndarray,
                     bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures distribution shift between baseline and current data.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        
        Args:
            baseline: Baseline distribution
            current: Current distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Remove NaN values
        baseline = baseline[~np.isnan(baseline)]
        current = current[~np.isnan(current)]
        
        if len(baseline) == 0 or len(current) == 0:
            logger.warning("Empty arrays for PSI calculation")
            return 0.0
        
        # Create bins based on baseline
        bin_edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Compute distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
        current_dist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize
        baseline_dist = baseline_dist / len(baseline)
        current_dist = current_dist / len(current)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_dist = baseline_dist + epsilon
        current_dist = current_dist + epsilon
        
        # Calculate PSI
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
        
        return float(psi)
    
    def check_population_shift(self, current_data: pd.DataFrame,
                              baseline_data: Optional[pd.DataFrame] = None,
                              baseline_file: Optional[str] = None) -> Dict:
        """
        Check for population mean shift using PSI.
        
        Args:
            current_data: Current data
            baseline_data: Baseline data (optional if baseline_file provided)
            baseline_file: Path to baseline file
            
        Returns:
            Dictionary with PSI scores per column
        """
        # Load baseline if not provided
        if baseline_data is None and baseline_file:
            baseline_path = self.baseline_dir / baseline_file
            if baseline_path.exists():
                baseline_data = pd.read_parquet(baseline_path)
            else:
                logger.warning(f"Baseline file not found: {baseline_path}")
                return {}
        
        if baseline_data is None:
            logger.warning("No baseline data available")
            return {}
        
        psi_scores = {}
        threshold = self.thresholds.get('population_mean_shift', {}).get('threshold', 0.2)
        
        # Check numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in baseline_data.columns:
                continue
            
            psi = self.calculate_psi(
                baseline_data[col].values,
                current_data[col].values
            )
            psi_scores[col] = psi
            
            if psi > threshold:
                logger.warning(f"Population shift detected in {col}: PSI={psi:.3f}")
                self.trigger_action('frontier_refit', column=col, psi=psi)
        
        return psi_scores
    
    def check_covariance_shift(self, current_data: pd.DataFrame,
                               baseline_data: Optional[pd.DataFrame] = None) -> float:
        """
        Check for covariance shift using Frobenius norm.
        
        Args:
            current_data: Current data
            baseline_data: Baseline data
            
        Returns:
            Relative change in Frobenius norm
        """
        if baseline_data is None:
            logger.warning("No baseline data for covariance shift check")
            return 0.0
        
        # Select numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in baseline_data.columns]
        
        if len(common_cols) < 2:
            logger.warning("Not enough columns for covariance shift check")
            return 0.0
        
        # Compute covariance matrices
        baseline_cov = baseline_data[common_cols].cov().values
        current_cov = current_data[common_cols].cov().values
        
        # Compute Frobenius norms
        baseline_norm = np.linalg.norm(baseline_cov, 'fro')
        current_norm = np.linalg.norm(current_cov, 'fro')
        
        # Relative change
        if baseline_norm > 0:
            relative_change = abs(current_norm - baseline_norm) / baseline_norm
        else:
            relative_change = 0.0
        
        threshold = self.thresholds.get('covariance_shift', {}).get('threshold', 0.1)
        
        if relative_change > threshold:
            logger.warning(f"Covariance shift detected: {relative_change:.3f}")
            self.trigger_action('posterior_recenter', change=relative_change)
        
        return relative_change
    
    def calculate_ece(self, predictions: np.ndarray, 
                     actuals: np.ndarray,
                     n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            predictions: Predicted probabilities
            actuals: Actual outcomes (0 or 1)
            n_bins: Number of bins
            
        Returns:
            ECE value
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            
            if np.sum(bin_mask) == 0:
                continue
            
            bin_preds = predictions[bin_mask]
            bin_actuals = actuals[bin_mask]
            
            bin_confidence = np.mean(bin_preds)
            bin_accuracy = np.mean(bin_actuals)
            bin_weight = len(bin_preds) / len(predictions)
            
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return ece
    
    def check_calibration_shift(self, predictions: np.ndarray,
                                actuals: np.ndarray,
                                baseline_ece: Optional[float] = None) -> float:
        """
        Check for calibration drift.
        
        Args:
            predictions: Current predictions
            actuals: Current actuals
            baseline_ece: Baseline ECE (optional)
            
        Returns:
            Change in ECE
        """
        current_ece = self.calculate_ece(predictions, actuals)
        
        if baseline_ece is None:
            baseline_ece = self.load_baseline_ece()
        
        if baseline_ece is None:
            logger.warning("No baseline ECE available")
            self.save_baseline_ece(current_ece)
            return 0.0
        
        delta = abs(current_ece - baseline_ece)
        threshold = self.thresholds.get('calibration_shift', {}).get('threshold', 0.02)
        
        if delta > threshold:
            logger.warning(f"Calibration drift detected: ΔECE={delta:.3f}")
            self.trigger_action('calibration_refresh', delta=delta)
        
        return delta
    
    def save_baseline_ece(self, ece: float):
        """Save baseline ECE."""
        baseline_file = self.baseline_dir / "baseline_ece.json"
        with open(baseline_file, 'w') as f:
            json.dump({'ece': ece, 'timestamp': datetime.now().isoformat()}, f)
    
    def load_baseline_ece(self) -> Optional[float]:
        """Load baseline ECE."""
        baseline_file = self.baseline_dir / "baseline_ece.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                data = json.load(f)
                return data.get('ece')
        return None
    
    def save_baseline_data(self, data: pd.DataFrame, name: str):
        """Save baseline data for future comparisons."""
        baseline_file = self.baseline_dir / f"{name}.parquet"
        data.to_parquet(baseline_file)
        logger.info(f"Saved baseline data to {baseline_file}")
    
    def get_drift_summary(self) -> Dict:
        """Get summary of all drift metrics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'thresholds': self.thresholds,
            'registered_actions': list(self.action_callbacks.keys())
        }
