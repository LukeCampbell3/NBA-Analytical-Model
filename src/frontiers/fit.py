"""
Frontier fitting module for NBA player performance prediction.

This module implements efficiency frontier fitting using quantile regression
to define trade-off boundaries between performance attributes.
"""

import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from statsmodels.regression.quantile_regression import QuantReg
from scipy.interpolate import interp1d


@dataclass
class Halfspace:
    """
    Represents a halfspace constraint: a^T x <= b
    
    Attributes:
        normal: Normal vector (a)
        offset: Offset scalar (b)
    """
    normal: np.ndarray
    offset: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'normal': self.normal.tolist(),
            'offset': float(self.offset)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Halfspace':
        """Create from dictionary."""
        return cls(
            normal=np.array(data['normal']),
            offset=float(data['offset'])
        )


@dataclass
class FrontierModel:
    """
    Represents a fitted efficiency frontier model.
    
    Attributes:
        x_attr: Name of x-axis attribute
        y_attr: Name of y-axis attribute
        strata: Stratification parameters (role, opponent_scheme_bin, etc.)
        quantile: Quantile level used for fitting (e.g., 0.9 for 90th percentile)
        coefficients: Fitted quantile regression coefficients
        x_range: Valid range for x attribute (min, max)
        y_range: Valid range for y attribute (min, max)
    """
    x_attr: str
    y_attr: str
    strata: Dict[str, Any]
    quantile: float
    coefficients: np.ndarray
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'x_attr': self.x_attr,
            'y_attr': self.y_attr,
            'strata': self.strata,
            'quantile': self.quantile,
            'coefficients': self.coefficients.tolist(),
            'x_range': list(self.x_range),
            'y_range': list(self.y_range)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrontierModel':
        """Create from dictionary."""
        return cls(
            x_attr=data['x_attr'],
            y_attr=data['y_attr'],
            strata=data['strata'],
            quantile=data['quantile'],
            coefficients=np.array(data['coefficients']),
            x_range=tuple(data['x_range']),
            y_range=tuple(data['y_range'])
        )


class FrontierFitter:
    """
    Fits efficiency frontiers using quantile regression.
    
    The frontier represents the trade-off boundary between two performance
    attributes (e.g., scoring efficiency vs. volume). Frontiers are stratified
    by role and opponent scheme to capture context-specific constraints.
    """
    
    def __init__(self, min_samples: int = 30):
        """
        Initialize FrontierFitter.
        
        Args:
            min_samples: Minimum number of samples required to fit a frontier
        """
        self.min_samples = min_samples
    
    def fit_frontier(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        strata: Dict[str, Any],
        quantile: float = 0.9
    ) -> FrontierModel:
        """
        Fit an efficiency frontier using quantile regression.
        
        Args:
            data: DataFrame with player performance data
            x: Name of x-axis attribute column
            y: Name of y-axis attribute column
            strata: Dictionary of stratification parameters (e.g., {'role': 'starter'})
            quantile: Quantile level for frontier (default: 0.9 for 90th percentile)
        
        Returns:
            FrontierModel with fitted coefficients
        
        Raises:
            ValueError: If insufficient data or invalid parameters
        """
        # Filter data by strata
        filtered_data = data.copy()
        for key, value in strata.items():
            if key in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[key] == value]
        
        if len(filtered_data) < self.min_samples:
            raise ValueError(
                f"Insufficient data for strata {strata}: "
                f"got {len(filtered_data)} samples, need {self.min_samples}"
            )
        
        # Extract x and y values
        if x not in filtered_data.columns or y not in filtered_data.columns:
            raise ValueError(f"Columns {x} or {y} not found in data")
        
        X_vals = filtered_data[x].values
        y_vals = filtered_data[y].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(X_vals) | np.isnan(y_vals))
        X_vals = X_vals[valid_mask]
        y_vals = y_vals[valid_mask]
        
        if len(X_vals) < self.min_samples:
            raise ValueError(
                f"Insufficient valid data after removing NaNs: "
                f"got {len(X_vals)} samples, need {self.min_samples}"
            )
        
        # Prepare design matrix (add intercept)
        X_design = np.column_stack([np.ones(len(X_vals)), X_vals])
        
        # Fit quantile regression
        model = QuantReg(y_vals, X_design)
        result = model.fit(q=quantile)
        coefficients = result.params
        
        # Store valid ranges
        x_range = (float(np.min(X_vals)), float(np.max(X_vals)))
        y_range = (float(np.min(y_vals)), float(np.max(y_vals)))
        
        return FrontierModel(
            x_attr=x,
            y_attr=y,
            strata=strata,
            quantile=quantile,
            coefficients=coefficients,
            x_range=x_range,
            y_range=y_range
        )
    
    def linearize_frontier(
        self,
        model: FrontierModel,
        n_segments: int = 10
    ) -> List[Halfspace]:
        """
        Convert fitted frontier to halfspace representation.
        
        The frontier curve is approximated by piecewise linear segments,
        each represented as a halfspace constraint.
        
        Args:
            model: Fitted FrontierModel
            n_segments: Number of linear segments for approximation
        
        Returns:
            List of Halfspace constraints representing the frontier
        """
        # Generate grid of x values
        x_min, x_max = model.x_range
        x_grid = np.linspace(x_min, x_max, n_segments + 1)
        
        # Compute y values on frontier using fitted model
        # y = coef[0] + coef[1] * x
        y_grid = model.coefficients[0] + model.coefficients[1] * x_grid
        
        # Create halfspaces from consecutive points
        halfspaces = []
        
        for i in range(n_segments):
            x1, y1 = x_grid[i], y_grid[i]
            x2, y2 = x_grid[i + 1], y_grid[i + 1]
            
            # Compute normal vector perpendicular to segment
            # Segment direction: (x2-x1, y2-y1)
            # Normal (pointing inward/below frontier): (y2-y1, -(x2-x1))
            dx = x2 - x1
            dy = y2 - y1
            
            # Normal vector (perpendicular, pointing below the frontier)
            normal = np.array([dy, -dx])
            
            # Normalize
            norm_length = np.linalg.norm(normal)
            if norm_length > 1e-10:
                normal = normal / norm_length
            
            # Compute offset: b = a^T * point
            # Use midpoint of segment
            midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            offset = np.dot(normal, midpoint)
            
            halfspaces.append(Halfspace(normal=normal, offset=offset))
        
        # Add boundary constraints to ensure feasible region
        # Lower bound on x: x >= x_min => -x <= -x_min
        halfspaces.append(Halfspace(
            normal=np.array([-1.0, 0.0]),
            offset=-x_min
        ))
        
        # Upper bound on x: x <= x_max
        halfspaces.append(Halfspace(
            normal=np.array([1.0, 0.0]),
            offset=x_max
        ))
        
        # Lower bound on y: y >= 0 (assuming non-negative attributes)
        halfspaces.append(Halfspace(
            normal=np.array([0.0, -1.0]),
            offset=0.0
        ))
        
        return halfspaces
    
    def save_frontier(self, model: FrontierModel, path: str) -> None:
        """
        Save frontier model to disk.
        
        Args:
            model: FrontierModel to save
            path: File path for saving (will create parent directories if needed)
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary and save with joblib
        model_dict = model.to_dict()
        joblib.dump(model_dict, path)
    
    def load_frontier(self, path: str) -> FrontierModel:
        """
        Load frontier model from disk.
        
        Args:
            path: File path to load from
        
        Returns:
            Loaded FrontierModel
        
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Frontier model not found at {path}")
        
        model_dict = joblib.load(path)
        return FrontierModel.from_dict(model_dict)
