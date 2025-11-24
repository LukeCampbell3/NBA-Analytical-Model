"""
Cold-start priors for new players and teams.

Implements hierarchical Bayes priors with role-based initialization.
"""

from datetime import datetime
from typing import Optional, Dict
import numpy as np
from pathlib import Path
import json

from src.features.transform import PosteriorParams
from src.simulation.global_sim import OpponentContext
from src.priors.role_inference import RoleInferenceModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ColdStartPriors:
    """
    Manages priors for new players and teams.
    
    Uses hierarchical Bayes approach:
    - League baseline by role
    - Wider uncertainty for new players
    - Exponential decay as games accumulate
    """
    
    def __init__(self, season: int, priors_dir: str = "artifacts/priors"):
        """
        Initialize cold-start priors.
        
        Args:
            season: Season year
            priors_dir: Directory containing prior files
        """
        self.season = season
        self.priors_dir = Path(priors_dir)
        self.league_baseline = self._load_league_baseline()
        self.role_priors = self._load_role_priors()
        self.role_inference = RoleInferenceModel()
        
        logger.info(f"Initialized ColdStartPriors for season {season}")
    
    def _load_league_baseline(self) -> Dict:
        """Load league baseline priors."""
        baseline_file = self.priors_dir / f"league_baseline_v{self.season}.json"
        
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"League baseline not found, using defaults")
            return self._get_default_baseline()
    
    def _get_default_baseline(self) -> Dict:
        """Get default league baseline if file not found."""
        return {
            'starter': {
                'mu': [25.0, 5.5, 4.5, 1.0, 0.5, 2.0],  # PTS, REB, AST, STL, BLK, TOV
                'Sigma': [
                    [36.0, 3.0, 2.0, 0.5, 0.2, 1.0],
                    [3.0, 9.0, 1.0, 0.3, 0.1, 0.5],
                    [2.0, 1.0, 16.0, 0.4, 0.1, 0.8],
                    [0.5, 0.3, 0.4, 1.5, 0.1, 0.3],
                    [0.2, 0.1, 0.1, 0.1, 0.8, 0.1],
                    [1.0, 0.5, 0.8, 0.3, 0.1, 4.0]
                ]
            },
            'rotation': {
                'mu': [15.0, 4.0, 2.5, 0.7, 0.4, 1.5],
                'Sigma': [
                    [25.0, 2.0, 1.5, 0.4, 0.15, 0.8],
                    [2.0, 6.0, 0.8, 0.2, 0.08, 0.4],
                    [1.5, 0.8, 9.0, 0.3, 0.08, 0.6],
                    [0.4, 0.2, 0.3, 1.0, 0.08, 0.2],
                    [0.15, 0.08, 0.08, 0.08, 0.5, 0.08],
                    [0.8, 0.4, 0.6, 0.2, 0.08, 2.5]
                ]
            },
            'bench': {
                'mu': [8.0, 2.5, 1.5, 0.4, 0.2, 1.0],
                'Sigma': [
                    [16.0, 1.5, 1.0, 0.3, 0.1, 0.6],
                    [1.5, 4.0, 0.6, 0.15, 0.05, 0.3],
                    [1.0, 0.6, 4.0, 0.2, 0.05, 0.4],
                    [0.3, 0.15, 0.2, 0.6, 0.05, 0.15],
                    [0.1, 0.05, 0.05, 0.05, 0.3, 0.05],
                    [0.6, 0.3, 0.4, 0.15, 0.05, 1.5]
                ]
            },
            'unknown': {
                'mu': [12.0, 3.5, 2.0, 0.6, 0.3, 1.2],
                'Sigma': [
                    [30.0, 2.5, 1.8, 0.45, 0.18, 0.9],
                    [2.5, 7.5, 0.9, 0.25, 0.09, 0.45],
                    [1.8, 0.9, 12.0, 0.35, 0.09, 0.7],
                    [0.45, 0.25, 0.35, 1.2, 0.09, 0.25],
                    [0.18, 0.09, 0.09, 0.09, 0.6, 0.09],
                    [0.9, 0.45, 0.7, 0.25, 0.09, 3.0]
                ]
            }
        }
    
    def _load_role_priors(self) -> Dict:
        """Load role distribution priors."""
        return {
            'G': {'starter': 0.35, 'rotation': 0.40, 'bench': 0.25},
            'W': {'starter': 0.30, 'rotation': 0.45, 'bench': 0.25},
            'BIG': {'starter': 0.25, 'rotation': 0.40, 'bench': 0.35},
            'unknown': {'starter': 0.30, 'rotation': 0.40, 'bench': 0.30}
        }
    
    def get_player_prior(self, player_id: str,
                        role: Optional[str] = None,
                        n_games: int = 0,
                        player_info: Optional[Dict] = None) -> PosteriorParams:
        """
        Get prior for new or low-data player.
        
        Args:
            player_id: Player identifier
            role: Player role (if known)
            n_games: Number of games played
            player_info: Additional player info (height, weight, etc.)
            
        Returns:
            PosteriorParams with appropriate prior
        """
        # Infer role if not provided
        if role is None or role == 'unknown':
            if player_info:
                role = self.role_inference.infer_role(player_info)
                logger.info(f"Inferred role '{role}' for player {player_id}")
            else:
                role = 'unknown'
                logger.warning(f"No role info for player {player_id}, using 'unknown'")
        
        # Get baseline for role
        baseline = self.league_baseline.get(role, self.league_baseline['unknown'])
        mu = np.array(baseline['mu'])
        Sigma = np.array(baseline['Sigma'])
        
        # Widen uncertainty for new players
        if n_games < 10:
            scale = 1.35 * np.exp(-n_games / 8.0)
            Sigma = Sigma * scale
            logger.info(f"Widening prior for {player_id} (n_games={n_games}, scale={scale:.2f})")
        
        return PosteriorParams(
            mu=mu,
            Sigma=Sigma,
            player_id=player_id,
            as_of_date=datetime.now()
        )
    
    def get_team_prior(self, team_id: str) -> OpponentContext:
        """
        Get prior for new team.
        
        Uses league medians for all features.
        
        Args:
            team_id: Team identifier
            
        Returns:
            OpponentContext with league median values
        """
        logger.info(f"Using league median prior for new team {team_id}")
        
        return OpponentContext(
            opponent_id=team_id,
            scheme_drop_rate=0.40,  # league median
            scheme_switch_rate=0.30,
            scheme_ice_rate=0.20,
            blitz_rate=0.15,
            rim_deterrence_index=1.0,
            def_reb_strength=1.0,
            foul_discipline_index=1.0,
            pace=100.0,
            help_nail_freq=0.35
        )
    
    def update_with_data(self, player_id: str, 
                        recent_games: np.ndarray,
                        prior: PosteriorParams,
                        n_games: int) -> PosteriorParams:
        """
        Update prior with recent game data using Bayesian update.
        
        Args:
            player_id: Player identifier
            recent_games: Array of recent game stats (n_games x n_features)
            prior: Current prior
            n_games: Total games played
            
        Returns:
            Updated PosteriorParams
        """
        if len(recent_games) == 0:
            return prior
        
        # Compute sample statistics
        sample_mean = np.mean(recent_games, axis=0)
        sample_cov = np.cov(recent_games.T)
        n = len(recent_games)
        
        # Bayesian update (conjugate prior)
        # Weight prior less as more data accumulates
        prior_weight = max(0.1, 1.0 / (1.0 + n / 5.0))
        data_weight = 1.0 - prior_weight
        
        # Update mean
        updated_mu = prior_weight * prior.mu + data_weight * sample_mean
        
        # Update covariance (simple weighted average)
        updated_Sigma = prior_weight * prior.Sigma + data_weight * sample_cov
        
        logger.info(f"Updated prior for {player_id} with {n} games "
                   f"(prior_weight={prior_weight:.2f})")
        
        return PosteriorParams(
            mu=updated_mu,
            Sigma=updated_Sigma,
            player_id=player_id,
            as_of_date=datetime.now()
        )
