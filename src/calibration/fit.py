"""
Calibration module for probabilistic predictions.

This module provides calibration functionality for model predictions including:
- Probability Integral Transform (PIT) computation
- Isotonic regression for per-statistic calibration
- Copula fitting for multivariate dependencies
- Correlated sampling from fitted copulas
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression
from scipy import stats
from scipy.stats import norm, multivariate_normal
import joblib


@dataclass
class CopulaModel:
    """Gaussian copula model for multivariate dependencies."""
    correlation_matrix: np.ndarray
    stat_names: List[str]
    
    def __post_init__(self):
        """Validate correlation matrix."""
        if self.correlation_matrix.shape[0] != self.correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")
        if len(self.stat_names) != self.correlation_matrix.shape[0]:
            raise ValueError("Number of stat names must match correlation matrix dimension")


class Calibrator:
    """
    Calibrator for probabilistic predictions.
    
    Provides methods for:
    - Computing PIT values for calibration diagnostics
    - Fitting isotonic regression models for calibration
    - Applying calibration transformations
    - Fitting copulas for multivariate dependencies
    - Sampling from copulas with calibrated marginals
    """
    
    def __init__(self):
        """Initialize calibrator with empty model storage."""
        self.isotonic_models: Dict[str, IsotonicRegression] = {}
        self.copula_model: Optional[CopulaModel] = None
    
    def compute_pit(self, y_true: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
        """
        Compute Probability Integral Transform values.
        
        The PIT is the empirical CDF of the predicted distribution evaluated at
        the true outcome. For well-calibrated predictions, PIT values should be
        uniformly distributed on [0, 1].
        
        Args:
            y_true: True outcome values, shape (n_instances,)
            y_samples: Predicted samples, shape (n_instances, n_samples)
        
        Returns:
            PIT values, shape (n_instances,)
            
        Example:
            >>> y_true = np.array([10.0, 20.0, 15.0])
            >>> y_samples = np.array([[8, 9, 11, 12], [18, 19, 21, 22], [14, 15, 16, 17]])
            >>> pit = calibrator.compute_pit(y_true, y_samples)
            >>> # pit[0] ≈ 0.5 (10.0 is at median of [8, 9, 11, 12])
        """
        if y_true.shape[0] != y_samples.shape[0]:
            raise ValueError(f"Shape mismatch: y_true has {y_true.shape[0]} instances, "
                           f"y_samples has {y_samples.shape[0]}")
        
        n_instances = y_true.shape[0]
        pit_values = np.zeros(n_instances)
        
        for i in range(n_instances):
            # Compute empirical CDF at y_true[i]
            # F(y) = P(Y <= y) = fraction of samples <= y
            pit_values[i] = np.mean(y_samples[i] <= y_true[i])
        
        return pit_values
    
    def fit_isotonic(self, stat: str, pits: np.ndarray) -> IsotonicRegression:
        """
        Fit isotonic regression for per-statistic calibration.
        
        Isotonic regression maps predicted quantiles to calibrated quantiles
        by fitting a monotonic function. This corrects systematic over/under-
        confidence in predictions.
        
        Args:
            stat: Statistic name (e.g., "PTS", "REB", "AST")
            pits: PIT values from validation set, shape (n_instances,)
        
        Returns:
            Fitted IsotonicRegression model
            
        Example:
            >>> pits = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            >>> model = calibrator.fit_isotonic("PTS", pits)
            >>> calibrator.isotonic_models["PTS"] = model
        """
        if len(pits) == 0:
            raise ValueError("Cannot fit isotonic regression with empty PIT array")
        
        # Sort PIT values for isotonic regression
        # We want to map predicted quantiles (sorted PITs) to uniform quantiles
        sorted_pits = np.sort(pits)
        n = len(sorted_pits)
        
        # Target: uniform quantiles [1/(n+1), 2/(n+1), ..., n/(n+1)]
        uniform_quantiles = np.arange(1, n + 1) / (n + 1)
        
        # Fit isotonic regression: sorted_pits -> uniform_quantiles
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(sorted_pits, uniform_quantiles)
        
        # Store model
        self.isotonic_models[stat] = model
        
        return model
    
    def apply_calibration(
        self, 
        stat: str, 
        samples: np.ndarray, 
        model: Optional[IsotonicRegression] = None
    ) -> np.ndarray:
        """
        Apply calibration transformation to new predictions.
        
        Transforms predicted samples using the fitted isotonic regression model
        to produce calibrated predictions.
        
        Args:
            stat: Statistic name
            samples: Predicted samples, shape (n_instances, n_samples) or (n_samples,)
            model: Isotonic regression model (if None, uses stored model for stat)
        
        Returns:
            Calibrated samples with same shape as input
            
        Example:
            >>> samples = np.array([[8, 10, 12, 14], [18, 20, 22, 24]])
            >>> calibrated = calibrator.apply_calibration("PTS", samples)
        """
        if model is None:
            if stat not in self.isotonic_models:
                raise ValueError(f"No calibration model found for stat '{stat}'. "
                               f"Call fit_isotonic() first.")
            model = self.isotonic_models[stat]
        
        original_shape = samples.shape
        samples_flat = samples.flatten()
        
        # For each sample, compute its quantile in the empirical distribution
        # then map through isotonic regression
        n_total = len(samples_flat)
        calibrated_flat = np.zeros(n_total)
        
        # Sort samples to compute quantiles
        sorted_indices = np.argsort(samples_flat)
        sorted_samples = samples_flat[sorted_indices]
        
        # Compute empirical quantiles
        empirical_quantiles = np.arange(1, n_total + 1) / (n_total + 1)
        
        # Apply isotonic transformation to quantiles
        calibrated_quantiles = model.predict(empirical_quantiles)
        
        # Map back to sample values using inverse transform
        # Use linear interpolation to map calibrated quantiles back to values
        calibrated_flat[sorted_indices] = np.interp(
            calibrated_quantiles,
            empirical_quantiles,
            sorted_samples
        )
        
        return calibrated_flat.reshape(original_shape)
    
    def fit_copula(self, stats_matrix: np.ndarray, stat_names: List[str]) -> CopulaModel:
        """
        Fit Gaussian copula to capture multivariate dependencies.
        
        A copula models the dependency structure between variables independently
        of their marginal distributions. This allows us to generate correlated
        predictions (e.g., high PTS often correlates with high AST for playmakers).
        
        Args:
            stats_matrix: Matrix of statistics, shape (n_instances, n_stats)
                         Each column is a different statistic
            stat_names: Names of statistics corresponding to columns
        
        Returns:
            Fitted CopulaModel
            
        Example:
            >>> stats = np.array([[25, 5, 7], [30, 4, 8], [20, 6, 6]])  # PTS, REB, AST
            >>> copula = calibrator.fit_copula(stats, ["PTS", "REB", "AST"])
        """
        if stats_matrix.shape[1] != len(stat_names):
            raise ValueError(f"Number of columns ({stats_matrix.shape[1]}) must match "
                           f"number of stat names ({len(stat_names)})")
        
        n_instances, n_stats = stats_matrix.shape
        
        if n_instances < 2:
            raise ValueError("Need at least 2 instances to fit copula")
        
        # Transform each marginal to uniform using empirical CDF
        uniform_matrix = np.zeros_like(stats_matrix)
        
        for j in range(n_stats):
            # Compute empirical CDF for each statistic
            sorted_indices = np.argsort(stats_matrix[:, j])
            ranks = np.empty(n_instances)
            ranks[sorted_indices] = np.arange(1, n_instances + 1)
            
            # Transform to uniform [0, 1]
            uniform_matrix[:, j] = ranks / (n_instances + 1)
        
        # Transform uniform to standard normal using inverse CDF
        normal_matrix = norm.ppf(uniform_matrix)
        
        # Compute correlation matrix of normal-transformed data
        correlation_matrix = np.corrcoef(normal_matrix.T)
        
        # Ensure correlation matrix is valid (symmetric, positive semi-definite)
        # Add small regularization if needed
        min_eigenval = np.min(np.linalg.eigvals(correlation_matrix))
        if min_eigenval < 1e-6:
            correlation_matrix += np.eye(n_stats) * (1e-6 - min_eigenval)
        
        copula_model = CopulaModel(
            correlation_matrix=correlation_matrix,
            stat_names=stat_names
        )
        
        self.copula_model = copula_model
        
        return copula_model
    
    def sample_copula(
        self, 
        model: CopulaModel, 
        marginals: Dict[str, np.ndarray],
        n_samples: int = 1000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample from fitted copula with given marginal distributions.
        
        Generates correlated samples by:
        1. Sampling from multivariate normal with fitted correlation
        2. Transforming to uniform via normal CDF
        3. Transforming to target marginals via inverse empirical CDF
        
        Args:
            model: Fitted CopulaModel
            marginals: Dictionary mapping stat names to marginal samples
                      Each value is array of shape (n_marginal_samples,)
            n_samples: Number of correlated samples to generate
            seed: Random seed for reproducibility
        
        Returns:
            Correlated samples, shape (n_samples, n_stats)
            
        Example:
            >>> marginals = {
            ...     "PTS": np.random.normal(25, 5, 1000),
            ...     "REB": np.random.normal(6, 2, 1000),
            ...     "AST": np.random.normal(7, 3, 1000)
            ... }
            >>> samples = calibrator.sample_copula(copula, marginals, n_samples=500)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_stats = len(model.stat_names)
        
        # Validate marginals
        for stat_name in model.stat_names:
            if stat_name not in marginals:
                raise ValueError(f"Missing marginal distribution for '{stat_name}'")
        
        # Sample from multivariate normal with fitted correlation
        mean = np.zeros(n_stats)
        normal_samples = multivariate_normal.rvs(
            mean=mean,
            cov=model.correlation_matrix,
            size=n_samples
        )
        
        # Handle case where n_samples=1 (returns 1D array)
        if n_samples == 1:
            normal_samples = normal_samples.reshape(1, -1)
        
        # Transform to uniform via normal CDF
        uniform_samples = norm.cdf(normal_samples)
        
        # Transform to target marginals via inverse empirical CDF
        correlated_samples = np.zeros((n_samples, n_stats))
        
        for j, stat_name in enumerate(model.stat_names):
            marginal_data = marginals[stat_name]
            sorted_marginal = np.sort(marginal_data)
            
            # Map uniform samples to marginal values using linear interpolation
            # uniform value u maps to quantile u of the marginal distribution
            quantile_positions = uniform_samples[:, j] * (len(sorted_marginal) - 1)
            correlated_samples[:, j] = np.interp(
                quantile_positions,
                np.arange(len(sorted_marginal)),
                sorted_marginal
            )
        
        return correlated_samples
    
    def save(self, path: str) -> None:
        """
        Save calibrator models to disk.
        
        Args:
            path: File path to save models (e.g., "artifacts/calibrator.pkl")
        """
        data = {
            'isotonic_models': self.isotonic_models,
            'copula_model': self.copula_model
        }
        joblib.dump(data, path)
    
    @classmethod
    def load(cls, path: str) -> 'Calibrator':
        """
        Load calibrator models from disk.
        
        Args:
            path: File path to load models from
        
        Returns:
            Loaded Calibrator instance
        """
        data = joblib.load(path)
        calibrator = cls()
        calibrator.isotonic_models = data['isotonic_models']
        calibrator.copula_model = data['copula_model']
        return calibrator
