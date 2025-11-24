"""
Unit tests for local rebound model.

Tests cover:
- Featurization of rebound events
- Model training
- Predictions in [0,1] range
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.local_models.rebound import ReboundModel


class TestReboundModel:
    """Test suite for ReboundModel class."""
    
    @pytest.fixture
    def rebound_model(self):
        """Create a ReboundModel instance for testing."""
        return ReboundModel()
    
    @pytest.fixture
    def sample_rebound_data(self):
        """Create sample rebound event data for testing."""
        n_events = 100
        
        df = pd.DataFrame({
            'player_height': np.random.uniform(72, 84, n_events),
            'player_wingspan': np.random.uniform(75, 90, n_events),
            'distance_to_basket': np.random.uniform(0, 20, n_events),
            'nearby_players': np.random.randint(1, 5, n_events),
            'box_out_quality': np.random.uniform(0, 1, n_events),
            'opponent_avg_height': np.random.uniform(72, 84, n_events),
            'player_speed': np.random.uniform(3, 8, n_events),
            'rebound_success': np.random.randint(0, 2, n_events)
        })
        
        return df
    
    def test_init(self, rebound_model):
        """Test ReboundModel initialization."""
        assert rebound_model.model is None
        assert len(rebound_model.feature_columns) == 4
        assert 'time_to_ball_proxy' in rebound_model.feature_columns
        assert 'crowd_index' in rebound_model.feature_columns
        assert 'reach_margin' in rebound_model.feature_columns
        assert 'seal_angle_proxy' in rebound_model.feature_columns
    
    def test_featurize_rebound(self, rebound_model, sample_rebound_data):
        """Test rebound featurization."""
        df_features = rebound_model.featurize_rebound(sample_rebound_data)
        
        # Check that all required features are present
        for feature in rebound_model.feature_columns:
            assert feature in df_features.columns
        
        # Check that features are numeric
        assert df_features['time_to_ball_proxy'].dtype in [np.float64, np.float32]
        assert df_features['crowd_index'].dtype in [np.float64, np.float32, np.int64, np.int32]
        assert df_features['reach_margin'].dtype in [np.float64, np.float32]
        assert df_features['seal_angle_proxy'].dtype in [np.float64, np.float32]
        
        # Check that seal_angle_proxy is in [0, 1]
        assert df_features['seal_angle_proxy'].min() >= 0
        assert df_features['seal_angle_proxy'].max() <= 1
    
    def test_featurize_rebound_minimal_data(self, rebound_model):
        """Test featurization with minimal required columns."""
        df_minimal = pd.DataFrame({
            'player_height': [78, 82, 75],
            'distance_to_basket': [5, 10, 15]
        })
        
        df_features = rebound_model.featurize_rebound(df_minimal)
        
        # Should still create all required features with defaults
        for feature in rebound_model.feature_columns:
            assert feature in df_features.columns
    
    def test_fit_rebound_logit(self, rebound_model, sample_rebound_data):
        """Test model fitting."""
        # Featurize data
        df_features = rebound_model.featurize_rebound(sample_rebound_data)
        
        # Fit model
        model = rebound_model.fit_rebound_logit(
            df_features,
            target_col='rebound_success',
            cv_folds=3,
            random_state=42
        )
        
        # Check that model is fitted
        assert model is not None
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Check coefficient dimensions
        assert model.coef_.shape[1] == len(rebound_model.feature_columns)
    
    def test_fit_rebound_logit_missing_features(self, rebound_model):
        """Test fitting with missing features raises error."""
        df_incomplete = pd.DataFrame({
            'time_to_ball_proxy': [1, 2, 3],
            'crowd_index': [2, 3, 4],
            'rebound_success': [0, 1, 0]
        })
        
        with pytest.raises(ValueError, match="Missing required features"):
            rebound_model.fit_rebound_logit(df_incomplete)
    
    def test_fit_rebound_logit_missing_target(self, rebound_model, sample_rebound_data):
        """Test fitting with missing target raises error."""
        df_features = rebound_model.featurize_rebound(sample_rebound_data)
        df_features = df_features.drop(columns=['rebound_success'], errors='ignore')
        
        with pytest.raises(ValueError, match="Target column"):
            rebound_model.fit_rebound_logit(df_features, target_col='rebound_success')
    
    def test_predict_rebound_prob(self, rebound_model, sample_rebound_data):
        """Test rebound probability prediction."""
        # Featurize and fit
        df_features = rebound_model.featurize_rebound(sample_rebound_data)
        model = rebound_model.fit_rebound_logit(
            df_features,
            target_col='rebound_success',
            cv_folds=2,
            random_state=42
        )
        
        # Predict on same data
        probs = rebound_model.predict_rebound_prob(model, df_features)
        
        # Check predictions
        assert len(probs) == len(df_features)
        assert all(probs >= 0)
        assert all(probs <= 1)
        assert probs.dtype in [np.float64, np.float32]
    
    def test_predict_rebound_prob_missing_features(self, rebound_model, sample_rebound_data):
        """Test prediction with missing features raises error."""
        # Fit model first
        df_features = rebound_model.featurize_rebound(sample_rebound_data)
        model = rebound_model.fit_rebound_logit(
            df_features,
            target_col='rebound_success',
            cv_folds=2,
            random_state=42
        )
        
        # Try to predict with incomplete data
        df_incomplete = pd.DataFrame({
            'time_to_ball_proxy': [1, 2, 3],
            'crowd_index': [2, 3, 4]
        })
        
        with pytest.raises(ValueError, match="Missing required features"):
            rebound_model.predict_rebound_prob(model, df_incomplete)
    
    def test_get_feature_importance(self, rebound_model, sample_rebound_data):
        """Test feature importance extraction."""
        # Fit model
        df_features = rebound_model.featurize_rebound(sample_rebound_data)
        model = rebound_model.fit_rebound_logit(
            df_features,
            target_col='rebound_success',
            cv_folds=2,
            random_state=42
        )
        
        # Get feature importance
        importance = rebound_model.get_feature_importance(model)
        
        # Check structure
        assert isinstance(importance, dict)
        assert len(importance) == len(rebound_model.feature_columns)
        
        # Check that all features are present
        for feature in rebound_model.feature_columns:
            assert feature in importance
            assert isinstance(importance[feature], (float, np.floating))
    
    def test_save_and_load_model(self, rebound_model, sample_rebound_data, tmp_path):
        """Test model persistence."""
        # Fit model
        df_features = rebound_model.featurize_rebound(sample_rebound_data)
        model = rebound_model.fit_rebound_logit(
            df_features,
            target_col='rebound_success',
            cv_folds=2,
            random_state=42
        )
        
        # Save model
        model_path = tmp_path / "rebound_model.pkl"
        rebound_model.save_model(model, str(model_path))
        
        # Check file exists
        assert model_path.exists()
        
        # Load model
        loaded_model = rebound_model.load_model(str(model_path))
        
        # Check that loaded model works
        probs_original = rebound_model.predict_rebound_prob(model, df_features)
        probs_loaded = rebound_model.predict_rebound_prob(loaded_model, df_features)
        
        # Predictions should be identical
        assert np.allclose(probs_original, probs_loaded)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
