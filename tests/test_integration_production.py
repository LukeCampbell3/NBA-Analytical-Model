"""
Integration tests for production-ready features.

Tests the complete flow with contracts, cold-start priors, and fallbacks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.data_loader import DataLoader
from src.features.transform import FeatureTransform
from src.regions.build import RegionBuilder
from src.contracts import PlayersPerGameContract
from src.priors import ColdStartPriors


class TestProductionIntegration:
    """Test production features integration."""
    
    def test_new_player_complete_flow(self):
        """Test complete flow for a new player with no historical data."""
        
        # 1. Initialize with production features (season auto-detected)
        loader = DataLoader(
            data_dir="Data",
            use_contracts=True
        )
        
        transform = FeatureTransform(
            window_games=20,
            decay_half_life=7,
            use_cold_start=True
        )
        
        builder = RegionBuilder(regularization=1e-6)
        
        # 2. Create empty DataFrame for new player
        empty_df = pd.DataFrame({
            'player_id': [],
            'game_id': [],
            'date': [],
            'team_id': [],
            'opponent_id': [],
            'minutes': []
        })
        
        # 3. Compute posterior with fallback
        posterior = transform.compute_player_posteriors_with_fallback(
            df=empty_df,
            player_id="rookie_2024_001",
            role="unknown",
            player_info={'usage': 0.22, 'three_pa_rate': 0.38}
        )
        
        # 4. Verify posterior is valid
        assert posterior is not None
        assert posterior.mu is not None
        assert posterior.Sigma is not None
        assert len(posterior.mu) > 0
        assert posterior.Sigma.shape[0] == posterior.Sigma.shape[1]
        
        # 5. Build region
        ellipsoid = builder.credible_ellipsoid(
            mu=posterior.mu,
            Sigma=posterior.Sigma,
            alpha=0.80
        )
        
        # 6. Verify ellipsoid is valid
        assert ellipsoid is not None
        assert ellipsoid.center is not None
        assert ellipsoid.shape_matrix is not None
    
    def test_schema_evolution(self):
        """Test handling of schema changes with aliases."""
        
        # Create DataFrame with old column names
        df_old = pd.DataFrame({
            'player_id': ['test_player'],
            'game_id': ['G001'],
            'date': [datetime(2024, 1, 1)],
            'team_id': ['GSW'],
            'opponent_id': ['LAL'],
            'minutes': [30.0],
            'usage_rate': [0.25],  # Old name
            'ts': [0.58],          # Old name
            'threepr': [0.42]      # Old name
        })
        
        # Load with contracts (season auto-detected)
        loader = DataLoader(use_contracts=True)
        df_new = loader.load_with_contract(
            df_old,
            PlayersPerGameContract,
            source="test"
        )
        
        # Verify aliases were applied
        assert 'usage' in df_new.columns
        assert 'ts_pct' in df_new.columns
        assert 'three_pa_rate' in df_new.columns
    
    def test_insufficient_data_fallback(self):
        """Test fallback when player has < 3 games."""
        
        # Create DataFrame with only 2 games
        df_partial = pd.DataFrame({
            'player_id': ['partial_player', 'partial_player'],
            'game_id': ['G001', 'G002'],
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 3)],
            'team_id': ['GSW', 'GSW'],
            'opponent_id': ['LAL', 'BOS'],
            'minutes': [25.0, 28.0],
            'TS%': [0.55, 0.60],
            'USG%': [0.22, 0.24],
            'AST%': [0.15, 0.18],
            'TOV%': [0.12, 0.10],
            'TRB%': [0.08, 0.09],
            'STL%': [0.02, 0.03],
            'BLK%': [0.01, 0.01]
        })
        
        # Compute posterior with fallback (season auto-detected)
        transform = FeatureTransform(use_cold_start=True)
        posterior = transform.compute_player_posteriors_with_fallback(
            df=df_partial,
            player_id="partial_player",
            role="rotation"
        )
        
        # Verify posterior is valid (should use prior + update)
        assert posterior is not None
        assert posterior.mu is not None
        assert len(posterior.mu) > 0
    
    def test_singular_matrix_fallback(self):
        """Test fallback when covariance matrix is singular."""
        
        # Create a singular covariance matrix
        dimension = 5
        mu = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        
        # Singular matrix (rank deficient)
        Sigma = np.zeros((dimension, dimension))
        Sigma[0, 0] = 1.0
        # Rest are zeros (singular)
        
        # Build ellipsoid with automatic regularization
        builder = RegionBuilder(regularization=1e-6)
        ellipsoid = builder.credible_ellipsoid(
            mu=mu,
            Sigma=Sigma,
            alpha=0.80
        )
        
        # Should succeed with regularization
        assert ellipsoid is not None
        assert ellipsoid.shape_matrix is not None
    
    def test_missing_columns_fallback(self):
        """Test fallback when DataFrame is missing columns."""
        
        # Create DataFrame with minimal columns
        df_minimal = pd.DataFrame({
            'player_id': ['minimal_player'],
            'game_id': ['G001'],
            'date': [datetime(2024, 1, 1)],
            'team_id': ['GSW'],
            'opponent_id': ['LAL'],
            'minutes': [30.0]
            # Missing: usage, ts_pct, etc.
        })
        
        # Load with contracts (should add defaults, season auto-detected)
        loader = DataLoader(use_contracts=True)
        df_complete = loader.load_with_contract(
            df_minimal,
            PlayersPerGameContract,
            source="test",
            apply_fallbacks=True
        )
        
        # Verify defaults were added
        assert 'usage' in df_complete.columns
        assert 'three_pa_rate' in df_complete.columns
        assert df_complete['usage'].iloc[0] == 0.18  # Default value
    
    def test_backward_compatibility(self):
        """Test that old code still works without contracts."""
        
        # Old-style initialization (no contracts)
        loader_old = DataLoader(data_dir="Data", use_contracts=False)
        transform_old = FeatureTransform(use_cold_start=False)
        
        # Should work without errors
        assert loader_old is not None
        assert transform_old is not None
        assert not loader_old.use_contracts
        assert transform_old._priors_cache is None


class TestColdStartPriors:
    """Test cold-start priors functionality."""
    
    def test_get_player_prior(self):
        """Test getting prior for new player."""
        priors = ColdStartPriors(season=2024)  # Season still needed for priors
        
        prior = priors.get_player_prior(
            player_id="test_player",
            role="starter",
            n_games=0
        )
        
        assert prior is not None
        assert prior.mu is not None
        assert prior.Sigma is not None
        assert len(prior.mu) > 0
    
    def test_prior_uncertainty_scaling(self):
        """Test that uncertainty widens for new players."""
        priors = ColdStartPriors(season=2024)
        
        # Get prior for player with 0 games
        prior_0 = priors.get_player_prior(
            player_id="test_player",
            role="starter",
            n_games=0
        )
        
        # Get prior for player with 5 games
        prior_5 = priors.get_player_prior(
            player_id="test_player",
            role="starter",
            n_games=5
        )
        
        # Uncertainty should be wider for 0 games
        assert np.trace(prior_0.Sigma) > np.trace(prior_5.Sigma)
    
    def test_role_inference(self):
        """Test role inference from player info."""
        priors = ColdStartPriors(season=2024)
        
        # High usage should suggest starter
        prior_high_usage = priors.get_player_prior(
            player_id="test_player",
            role=None,  # Will be inferred
            n_games=0,
            player_info={'usage': 0.28, 'minutes': 32}
        )
        
        assert prior_high_usage is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
