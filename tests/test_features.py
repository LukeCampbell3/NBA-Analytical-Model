"""
Unit tests for feature engineering functionality.

Tests cover:
- Rolling window computation with exponential decay
- Posterior computation (mu, Sigma)
- Scaler transformations
- Context joining
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.transform import FeatureTransform, PosteriorParams, RobustScalerParams


class TestFeatureTransform:
    """Test suite for FeatureTransform class."""
    
    @pytest.fixture
    def feature_transform(self):
        """Create a FeatureTransform instance for testing."""
        return FeatureTransform(window_games=20, decay_half_life=7)
    
    @pytest.fixture
    def sample_player_data(self):
        """Create sample player data for testing."""
        n_games = 30
        dates = pd.date_range('2024-01-01', periods=n_games)
        
        df = pd.DataFrame({
            'Date': dates,
            'Player': ['Test_Player'] * n_games,
            'PTS': np.random.randint(15, 35, n_games),
            'TRB': np.random.randint(3, 10, n_games),
            'AST': np.random.randint(2, 8, n_games),
            'STL': np.random.randint(0, 3, n_games),
            'BLK': np.random.randint(0, 2, n_games),
            'TOV': np.random.randint(1, 4, n_games),
            'MP': np.random.randint(25, 38, n_games),
            'TS%': np.random.uniform(0.50, 0.65, n_games),
            'USG%': np.random.uniform(20, 30, n_games),
            'AST%': np.random.uniform(15, 25, n_games),
            'TOV%': np.random.uniform(10, 15, n_games),
            'TRB%': np.random.uniform(8, 12, n_games),
            'STL%': np.random.uniform(1, 3, n_games),
            'BLK%': np.random.uniform(1, 3, n_games)
        })
        
        return df
    
    def test_init(self):
        """Test FeatureTransform initialization."""
        ft = FeatureTransform(window_games=15, decay_half_life=5)
        assert ft.window_games == 15
        assert ft.decay_half_life == 5
        assert ft.decay_lambda > 0
    
    def test_compute_exponential_weights(self, feature_transform):
        """Test exponential weight computation."""
        weights = feature_transform._compute_exponential_weights(10)
        
        assert len(weights) == 10
        assert weights[-1] == 1.0  # Most recent game has weight 1
        assert weights[0] < weights[-1]  # Older games have lower weight
        assert all(weights > 0)  # All weights positive
    
    def test_compute_rolling_features(self, feature_transform, sample_player_data):
        """Test rolling feature computation."""
        df_rolling = feature_transform.compute_rolling_features(
            sample_player_data,
            player_id='test_player'
        )
        
        # Check that rolling columns are added
        assert 'PTS_rolling_mean' in df_rolling.columns
        assert 'PTS_rolling_std' in df_rolling.columns
        assert 'player_id' in df_rolling.columns
        
        # Check that early games have NaN (no history)
        assert pd.isna(df_rolling['PTS_rolling_mean'].iloc[0])
        
        # Check that later games have valid values
        assert not pd.isna(df_rolling['PTS_rolling_mean'].iloc[-1])
        assert df_rolling['PTS_rolling_mean'].iloc[-1] > 0
    
    def test_compute_player_posteriors(self, feature_transform, sample_player_data):
        """Test posterior computation."""
        posteriors = feature_transform.compute_player_posteriors(
            sample_player_data,
            player_id='test_player'
        )
        
        assert isinstance(posteriors, PosteriorParams)
        assert posteriors.player_id == 'test_player'
        assert len(posteriors.mu) == len(feature_transform.CORE_ATTRIBUTES)
        assert posteriors.Sigma.shape == (len(posteriors.mu), len(posteriors.mu))
        
        # Check that Sigma is symmetric
        assert np.allclose(posteriors.Sigma, posteriors.Sigma.T)
        
        # Check that Sigma is positive definite (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvals(posteriors.Sigma)
        assert all(eigenvalues > 0)
    
    def test_compute_player_posteriors_insufficient_data(self, feature_transform):
        """Test posterior computation with insufficient data."""
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=2),
            'TS%': [0.6, 0.65],
            'USG%': [25, 27],
            'AST%': [20, 22],
            'TOV%': [12, 11],
            'TRB%': [10, 9],
            'STL%': [2, 2.5],
            'BLK%': [1.5, 1.8]
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            feature_transform.compute_player_posteriors(df)
    
    def test_compute_scalers(self, feature_transform, sample_player_data):
        """Test scaler computation."""
        features = ['PTS', 'TRB', 'AST']
        scaler_params = feature_transform.compute_scalers(
            sample_player_data[features],
            feature_names=features
        )
        
        assert isinstance(scaler_params, RobustScalerParams)
        assert len(scaler_params.center) == len(features)
        assert len(scaler_params.scale) == len(features)
        assert scaler_params.feature_names == features
    
    def test_apply_scalers(self, feature_transform, sample_player_data):
        """Test scaler application."""
        features = ['PTS', 'TRB', 'AST']
        
        # Compute scaler parameters
        scaler_params = feature_transform.compute_scalers(
            sample_player_data[features],
            feature_names=features
        )
        
        # Apply scaling
        df_scaled = feature_transform.apply_scalers(
            sample_player_data[features],
            scaler_params
        )
        
        # Check that scaling was applied
        assert df_scaled.shape == sample_player_data[features].shape
        
        # Check that scaled values are different from original
        assert not np.allclose(df_scaled['PTS'].values, sample_player_data['PTS'].values)
    
    def test_join_context_opponent(self, feature_transform, sample_player_data):
        """Test joining with opponent features."""
        # Create opponent data
        df_opponent = pd.DataFrame({
            'Date': sample_player_data['Date'],
            'opponent_id': ['OPP'] * len(sample_player_data),
            'def_rating': np.random.uniform(105, 115, len(sample_player_data)),
            'pace': np.random.uniform(95, 105, len(sample_player_data))
        })
        
        df_joined = feature_transform.join_context(
            sample_player_data,
            df_opponent=df_opponent
        )
        
        # Check that opponent features are added
        assert 'def_rating' in df_joined.columns or 'def_rating_opp' in df_joined.columns
        assert len(df_joined) == len(sample_player_data)
    
    def test_join_context_rotation(self, feature_transform, sample_player_data):
        """Test joining with rotation priors."""
        # Create rotation data
        df_rotation = pd.DataFrame({
            'Date': sample_player_data['Date'],
            'player_id': ['test_player'] * len(sample_player_data),
            'role': ['starter'] * len(sample_player_data),
            'exp_minutes': np.random.uniform(30, 35, len(sample_player_data))
        })
        
        # Add player_id to sample data
        sample_player_data['player_id'] = 'test_player'
        
        df_joined = feature_transform.join_context(
            sample_player_data,
            df_rotation=df_rotation
        )
        
        # Check that rotation features are added
        assert 'role' in df_joined.columns or 'role_rot' in df_joined.columns
        assert len(df_joined) == len(sample_player_data)
    
    def test_compute_all_features(self, feature_transform, sample_player_data):
        """Test full feature engineering pipeline."""
        df_features, posteriors = feature_transform.compute_all_features(
            sample_player_data,
            player_id='test_player'
        )
        
        # Check that rolling features are computed
        assert 'PTS_rolling_mean' in df_features.columns
        
        # Check that posteriors are computed
        assert isinstance(posteriors, PosteriorParams)
        assert posteriors.player_id == 'test_player'
        
        # Check that data is preserved
        assert len(df_features) == len(sample_player_data)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
