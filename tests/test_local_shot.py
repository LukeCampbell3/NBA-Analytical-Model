"""
Unit tests for local shot model.

Tests cover:
- Featurization of shot events
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

from src.local_models.shot import ShotModel


class TestShotModel:
    """Test suite for ShotModel class."""
    
    @pytest.fixture
    def shot_model(self):
        """Create a ShotModel instance for testing."""
        return ShotModel()
    
    @pytest.fixture
    def sample_shot_data(self):
        """Create sample shot event data for testing."""
        n_events = 100
        
        df = pd.DataFrame({
            'shooter_ts_pct': np.random.uniform(0.45, 0.65, n_events),
            'shooter_recent_ts': np.random.uniform(0.45, 0.65, n_events),
            'shot_distance': np.random.uniform(0, 30, n_events),
            'shooter_dribbles_before_shot': np.random.randint(0, 8, n_events),
            'shooter_time_of_possession': np.random.uniform(0, 5, n_events),
            'shooter_catch_and_shoot_pct': np.random.uniform(0.2, 0.8, n_events),
            'opponent_rim_protection': np.random.uniform(0.9, 1.1, n_events),
            'opponent_blk_pct': np.random.uniform(0.02, 0.08, n_events),
            'defender_distance': np.random.uniform(0, 8, n_events),
            'shot_made': np.random.randint(0, 2, n_events)
        })
        
        return df
    
    def test_init(self, shot_model):
        """Test ShotModel initialization."""
        assert shot_model.model is None
        assert len(shot_model.feature_columns) == 4
        assert 'shooter_ts_context' in shot_model.feature_columns
        assert 'distance_bin' in shot_model.feature_columns
        assert 'pullup_vs_catch_proxy' in shot_model.feature_columns
        assert 'opponent_rim_deterrence' in shot_model.feature_columns
    
    def test_featurize_shot(self, shot_model, sample_shot_data):
        """Test shot featurization."""
        df_features = shot_model.featurize_shot(sample_shot_data)
        
        # Check that all required features are present
        for feature in shot_model.feature_columns:
            assert feature in df_features.columns
        
        # Check that features are numeric
        assert df_features['shooter_ts_context'].dtype in [np.float64, np.float32]
        assert df_features['distance_bin'].dtype in [np.float64, np.float32]
        assert df_features['pullup_vs_catch_proxy'].dtype in [np.float64, np.float32]
        assert df_features['opponent_rim_deterrence'].dtype in [np.float64, np.float32]
        
        # Check that distance_bin is in valid range [0, 1, 2]
        assert df_features['distance_bin'].min() >= 0
        assert df_features['distance_bin'].max() <= 2
        
        # Check that pullup_vs_catch_proxy is in [0, 1]
        assert df_features['pullup_vs_catch_proxy'].min() >= 0
        assert df_features['pullup_vs_catch_proxy'].max() <= 1
        
        # Check that opponent_rim_deterrence is in [0, 1]
        assert df_features['opponent_rim_deterrence'].min() >= 0
        assert df_features['opponent_rim_deterrence'].max() <= 1
    
    def test_featurize_shot_minimal_data(self, shot_model):
        """Test featurization with minimal required columns."""
        df_minimal = pd.DataFrame({
            'shooter_ts_pct': [0.55, 0.60, 0.50]
        })
        
        df_features = shot_model.featurize_shot(df_minimal)
        
        # Should still create all required features with defaults
        for feature in shot_model.feature_columns:
            assert feature in df_features.columns
    
    def test_fit_shot_logit(self, shot_model, sample_shot_data):
        """Test model fitting."""
        # Featurize data
        df_features = shot_model.featurize_shot(sample_shot_data)
        
        # Fit model
        model = shot_model.fit_shot_logit(
            df_features,
            target_col='shot_made',
            cv_folds=3,
            random_state=42
        )
        
        # Check that model is fitted
        assert model is not None
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Check coefficient dimensions
        assert model.coef_.shape[1] == len(shot_model.feature_columns)
    
    def test_fit_shot_logit_missing_features(self, shot_model):
        """Test fitting with missing features raises error."""
        df_incomplete = pd.DataFrame({
            'shooter_ts_context': [0.55, 0.60, 0.50],
            'distance_bin': [0, 1, 2],
            'shot_made': [0, 1, 0]
        })
        
        with pytest.raises(ValueError, match="Missing required features"):
            shot_model.fit_shot_logit(df_incomplete)
    
    def test_fit_shot_logit_missing_target(self, shot_model, sample_shot_data):
        """Test fitting with missing target raises error."""
        df_features = shot_model.featurize_shot(sample_shot_data)
        df_features = df_features.drop(columns=['shot_made'], errors='ignore')
        
        with pytest.raises(ValueError, match="Target column"):
            shot_model.fit_shot_logit(df_features, target_col='shot_made')
    
    def test_predict_shot_prob(self, shot_model, sample_shot_data):
        """Test shot probability prediction."""
        # Featurize and fit
        df_features = shot_model.featurize_shot(sample_shot_data)
        model = shot_model.fit_shot_logit(
            df_features,
            target_col='shot_made',
            cv_folds=2,
            random_state=42
        )
        
        # Predict on same data
        probs = shot_model.predict_shot_prob(model, df_features)
        
        # Check predictions
        assert len(probs) == len(df_features)
        assert all(probs >= 0)
        assert all(probs <= 1)
        assert probs.dtype in [np.float64, np.float32]
    
    def test_predict_shot_prob_missing_features(self, shot_model, sample_shot_data):
        """Test prediction with missing features raises error."""
        # Fit model first
        df_features = shot_model.featurize_shot(sample_shot_data)
        model = shot_model.fit_shot_logit(
            df_features,
            target_col='shot_made',
            cv_folds=2,
            random_state=42
        )
        
        # Try to predict with incomplete data
        df_incomplete = pd.DataFrame({
            'shooter_ts_context': [0.55, 0.60, 0.50],
            'distance_bin': [0, 1, 2]
        })
        
        with pytest.raises(ValueError, match="Missing required features"):
            shot_model.predict_shot_prob(model, df_incomplete)
    
    def test_get_feature_importance(self, shot_model, sample_shot_data):
        """Test feature importance extraction."""
        # Fit model
        df_features = shot_model.featurize_shot(sample_shot_data)
        model = shot_model.fit_shot_logit(
            df_features,
            target_col='shot_made',
            cv_folds=2,
            random_state=42
        )
        
        # Get feature importance
        importance = shot_model.get_feature_importance(model)
        
        # Check structure
        assert isinstance(importance, dict)
        assert len(importance) == len(shot_model.feature_columns)
        
        # Check that all features are present
        for feature in shot_model.feature_columns:
            assert feature in importance
            assert isinstance(importance[feature], (float, np.floating))
    
    def test_save_and_load_model(self, shot_model, sample_shot_data, tmp_path):
        """Test model persistence."""
        # Fit model
        df_features = shot_model.featurize_shot(sample_shot_data)
        model = shot_model.fit_shot_logit(
            df_features,
            target_col='shot_made',
            cv_folds=2,
            random_state=42
        )
        
        # Save model
        model_path = tmp_path / "shot_model.pkl"
        shot_model.save_model(model, str(model_path))
        
        # Check file exists
        assert model_path.exists()
        
        # Load model
        loaded_model = shot_model.load_model(str(model_path))
        
        # Check that loaded model works
        probs_original = shot_model.predict_shot_prob(model, df_features)
        probs_loaded = shot_model.predict_shot_prob(loaded_model, df_features)
        
        # Predictions should be identical
        assert np.allclose(probs_original, probs_loaded)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
