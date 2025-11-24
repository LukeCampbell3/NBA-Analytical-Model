"""
Feature engineering module for NBA player statistics.

This module provides functionality for computing rolling statistics,
player posteriors, and feature transformations.
"""

from .transform import (
    FeatureTransform,
    PosteriorParams,
    RobustScalerParams
)

__all__ = [
    'FeatureTransform',
    'PosteriorParams',
    'RobustScalerParams'
]
