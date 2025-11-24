"""
League baseline manager for computing and storing league-wide priors.
"""

import json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LeagueBaselineManager:
    """
    Computes and manages league baseline statistics.
    
    Computes role-specific means and covariances from historical data.
    """
    
    def __init__(self, output_dir: str = "artifacts/priors"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_baseline(self, df: pd.DataFrame, season: int,
                        stats_columns: list = None) -> Dict:
        """
        Compute league baseline from historical data.
        
        Args:
            df: DataFrame with player-game data
            season: Season year
            stats_columns: List of stat columns to include
            
        Returns:
            Dictionary with role-specific baselines
        """
        if stats_columns is None:
            stats_columns = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
        
        logger.info(f"Computing league baseline for season {season}")
        
        baseline = {}
        
        for role in ['starter', 'rotation', 'bench']:
            role_data = df[df['role'] == role][stats_columns]
            
            if len(role_data) == 0:
                logger.warning(f"No data for role '{role}', skipping")
                continue
            
            # Compute mean and covariance
            mu = role_data.mean().values
            Sigma = role_data.cov().values
            
            baseline[role] = {
                'mu': mu.tolist(),
                'Sigma': Sigma.tolist(),
                'n_samples': len(role_data)
            }
            
            logger.info(f"Role '{role}': n={len(role_data)}, "
                       f"mean_pts={mu[0]:.1f}")
        
        return baseline
    
    def save_baseline(self, baseline: Dict, season: int):
        """Save baseline to file."""
        output_file = self.output_dir / f"league_baseline_v{season}.json"
        
        with open(output_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        logger.info(f"Saved league baseline to {output_file}")
    
    def load_baseline(self, season: int) -> Dict:
        """Load baseline from file."""
        baseline_file = self.output_dir / f"league_baseline_v{season}.json"
        
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_file}")
        
        with open(baseline_file, 'r') as f:
            return json.load(f)
