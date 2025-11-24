"""
Positional Tracking Module

This module provides functionality for processing player tracking data and deriving
spatial features for capability region construction. It is currently scaffolded and
disabled by default (positional.enabled=false in config).

When tracking data becomes available, this module can be enabled to replace proxy
features with measured spatial overlaps and enhance prediction accuracy.

Modules:
    - ingest_tracking: Load and validate tracking data
    - derive_features: Compute spatial features from tracking coordinates
    - build_spatial_region: Construct spatial capability volumes (SCV)
    - simulate_play_states: Simulate player movements and defensive rotations

Usage:
    from src.positional import ingest_tracking, derive_features
    
    # When enabled=true in config
    tracking_data = ingest_tracking.load_tracking_data(game_id)
    spatial_features = derive_features.compute_spatial_features(tracking_data)
"""

from typing import Dict, Any

__version__ = "0.1.0"
__all__ = [
    "ingest_tracking",
    "derive_features", 
    "build_spatial_region",
    "simulate_play_states"
]


def is_enabled(config: Dict[str, Any]) -> bool:
    """
    Check if positional tracking module is enabled in configuration.
    
    Args:
        config: System configuration dictionary
        
    Returns:
        True if positional tracking is enabled, False otherwise
    """
    return config.get("positional", {}).get("enabled", False)


def get_tracking_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get positional tracking configuration.
    
    Args:
        config: System configuration dictionary
        
    Returns:
        Positional tracking configuration dictionary
    """
    return config.get("positional", {})
