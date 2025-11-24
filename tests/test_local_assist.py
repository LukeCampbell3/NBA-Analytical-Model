"""
Unit tests for local assist model.

Tests cover:
- Featurization of assist events
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

from src.local_models.assist import AssistModel


class TestAssistModel:
    """Test suite for AssistModel class."""
    
    @pytest.fixture
    def assist_model(self):
        """Create an AssistModel instance for testing."""
        return AssistModel()
    
    @pytest.fixture
    def sample_assist_data(self):
        """Create sample assist event data for testing."""
        n_events = 100
        
        df = pd.DataFrame({
            'passer_usage': np.random.uniform(0.15, 0.35, n_events),
            'passer_ast_pct': np.random.uniform(0.10, 0.40, n_events),
            'passer_recent_ast_pct': np.random.uniform(0.10, 0.40, n_events),
            'receiver_shot_quality': np.random.uniform(0.45, 0.65, n_events),
            'receiver_ts_pct': np.random.uniform(0.50, 0.65, n_events),
            'receiver_catch_shoot_pct': np.random.uniform(0.3, 0.8, n_events),
            'opponent_help_freq': np.random.uniform(0.3, 0.7, n_events),
            'opponent_nail_freq': np.random.uniform(0.2, 0.6, n_events),
            'lane_congestion': np.random.uniform(0, 1, n_events),
            'defender_help_distance': np.random.uniform(0, 15, n_events),
            'assist_success': np.random.randint(0, 2, n_events)
        })
        
        return df
    
    def test_init(self, assist_model):
        """Test AssistModel initialization."""
        assert assist_model.model is None
        assert len(assist_model.feature_columns) == 5
        assert 'passer_usage' in assist_model.feature_columns
        assert 'passer_ast_pct' in assist_model.feature_columns
        assert 'receiver_shot_quality_proxy' in assist_model.feature_columns
        assert 'opponent_help_nail_freq' in assist_model.feature_columns
        assert 'lane_risk_proxy' in assist_model.feature_columns
    
    def test_featurize_assist(self, assist_model, sample_assist_data):
        """Test assist featurization."""
        df_features = assist_model.featurize_assist(sample_assist_data)
        
        # Check that all required features are present
        for feature in assist_model.feature_columns:
            assert feature in df_features.columns
        
        # Check that features are numeric
        assert df_features['passer_usage'].dtype in [np.float64, np.float32]
        assert df_features['passer_ast_pct'].dtype in [np.float64, np.float32]
        assert df_features['receiver_shot_quality_proxy'].dtype in [np.float64, np.float32]
        assert df_features['opponent_help_nail_freq'].dtype in [np.float64, np.float32]
        assert df_features['lane_risk_proxy'].dtype in [np.float64, np.float32]
        
        # Check that passer_usage is in reasonable range
        assert df_features['passer_usage'].min() >= 0
        assert df_features['passer_usage'].max() <= 1
        
        # Check that passer_ast_pct is in reasonable range
        assert df_features['passer_ast_pct'].min() >= 0
        assert df_features['passer_ast_pct'].max() <= 1
        
        # Check that receiver_shot_quality_proxy is in [0, 1]
        assert df_features['receiver_shot_quality_proxy'].min() >= 0
        assert df_features['receiver_shot_quality_proxy'].max() <= 1
        
        # Check that opponent_help_nail_freq is in [0, 1]
        assert df_features['opponent_help_nail_freq'].min() >= 0
        assert df_features['opponent_help_nail_freq'].max() <= 1
        
        # Check that lane_risk_proxy is in [0, 1]
        assert df_features['lane_risk_proxy'].min() >= 0
        assert df_features['lane_risk_proxy'].max() <= 1
    
    def test_featurize_assist_minimal_data(self, assist_model):
        """Test featurization with minimal required columns."""
        df_minimal = pd.DataFrame({
            'passer_usage': [0.25, 0.30, 0.20],
            'passer_ast_pct': [0.25, 0.30, 0.20]
        })
        
        df_features = assist_model.featurize_assist(df_minimal)
        
        # Should still create all required features with defaults
        for feature in assist_model.feature_columns:
            assert feature in df_features.columns
    
    def test_fit_assist_logit(self, assist_model, sample_assist_data):
        """Test model fitting."""
        # Featurize data
        df_features = assist_model.featurize_assist(sample_assist_data)
        
        # Fit model
        model = assist_model.fit_assist_logit(
            df_features,
            target_col='assist_success',
            cv_folds=3,
            random_state=42
        )
        
        # Check that model is fitted
        assert model is not None
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Check coefficient dimensions
        assert model.coef_.shape[1] == len(assist_model.feature_columns)
    
    def test_fit_assist_logit_missing_features(self, assist_model):
        """Test fitting with missing features raises error."""
        df_incomplete = pd.DataFrame({
            'passer_usage': [0.25, 0.30, 0.20],
            'passer_ast_pct': [0.25, 0.30, 0.20],
            'assist_success': [0, 1, 0]
        })
        
        with pytest.raises(ValueError, match="Missing required features"):
            assist_model.fit_assist_logit(df_incomplete)
    
    def test_fit_assist_logit_missing_target(self, assist_model, sample_assist_data):
        """Test fitting with missing target raises error."""
        df_features = assist_model.featurize_assist(sample_assist_data)
        df_features = df_features.drop(columns=['assist_success'], errors='ignore')
        
        with pytest.raises(ValueError, match="Target column"):
            assist_model.fit_assist_logit(df_features, target_col='assist_success')
    
    def test_predict_assist_prob(self, assist_model, sample_assist_data):
        """Test assist probability prediction."""
        # Featurize and fit
        df_features = assist_model.featurize_assist(sample_assist_data)
        model = assist_model.fit_assist_logit(
            df_features,
            target_col='assist_success',
            cv_folds=2,
            random_state=42
        )
        
        # Predict on same data
        probs = assist_model.predict_assist_prob(model, df_features)
        
        # Check predictions
        assert len(probs) == len(df_features)
        assert all(probs >= 0)
        assert all(probs <= 1)
        assert probs.dtype in [np.float64, np.float32]
    
    def test_predict_assist_prob_missing_features(self, assist_model, sample_assist_data):
        """Test prediction with missing features raises error."""
        # Fit model first
        df_features = assist_model.featurize_assist(sample_assist_data)
        model = assist_model.fit_assist_logit(
            df_features,
            target_col='assist_success',
            cv_folds=2,
            random_state=42
        )
        
        # Try to predict with incomplete data
        df_incomplete = pd.DataFrame({
            'passer_usage': [0.25, 0.30, 0.20],
            'passer_ast_pct': [0.25, 0.30, 0.20]
        })
        
        with pytest.raises(ValueError, match="Missing required features"):
            assist_model.predict_assist_prob(model, df_incomplete)
    
    def test_get_feature_importance(self, assist_model, sample_assist_data):
        """Test feature importance extraction."""
        # Fit model
        df_features = assist_model.featurize_assist(sample_assist_data)
        model = assist_model.fit_assist_logit(
            df_features,
            target_col='assist_success',
            cv_folds=2,
            random_state=42
        )
        
        # Get feature importance
        importance = assist_model.get_feature_importance(model)
        
        # Check structure
        assert isinstance(importance, dict)
        assert len(importance) == len(assist_model.feature_columns)
        
        # Check that all features are present
        for feature in assist_model.feature_columns:
            assert feature in importance
            assert isinstance(importance[feature], (float, np.floating))
    
    def test_save_and_load_model(self, assist_model, sample_assist_data, tmp_path):
        """Test model persistence."""
        # Fit model
        df_features = assist_model.featurize_assist(sample_assist_data)
        model = assist_model.fit_assist_logit(
            df_features,
            target_col='assist_success',
            cv_folds=2,
            random_state=42
        )
        
        # Save model
        model_path = tmp_path / "assist_model.pkl"
        assist_model.save_model(model, str(model_path))
        
        # Check file exists
        assert model_path.exists()
        
        # Load model
        loaded_model = assist_model.load_model(str(model_path))
        
        # Check that loaded model works
        probs_original = assist_model.predict_assist_prob(model, df_features)
        probs_loaded = assist_model.predict_assist_prob(loaded_model, df_features)
        
        # Predictions should be identical
        assert np.allclose(probs_original, probs_loaded)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
