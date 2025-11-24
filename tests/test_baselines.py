"""
Unit tests for baseline models.

Tests cover:
- Feature building (rolling means/variances, opponent features, role, pace)
- Ridge regression training and prediction
- XGBoost training and prediction (if available)
- MLP training and prediction
- Model saving and loading
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.models import BaselineModels, BaselineModelConfig


class TestBaselineModels:
    """Test suite for BaselineModels class."""
    
    @pytest.fixture
    def baseline_models(self):
        """Create a BaselineModels instance for testing."""
        config = BaselineModelConfig(
            ridge_alpha=1.0,
            xgb_max_depth=3,  # Smaller for faster tests
            xgb_n_estimators=50,  # Fewer estimators for faster tests
            mlp_hidden_layers=[32, 16],  # Smaller layers for faster tests
            mlp_max_iter=100  # Fewer iterations for faster tests
        )
        return BaselineModels(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample player data for testing."""
        n_games = 50
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
            'AST%': np.random.uniform(15, 35, n_games),
            'TOV%': np.random.uniform(8, 15, n_games),
            'TRB%': np.random.uniform(8, 15, n_games),
            'STL%': np.random.uniform(1, 3, n_games),
            'BLK%': np.random.uniform(1, 3, n_games),
            'role': ['starter'] * n_games,
            'pace': np.random.uniform(95, 105, n_games),
            'home_away': ['home' if i % 2 == 0 else 'away' for i in range(n_games)]
        })
        
        return df
    
    def test_build_features(self, baseline_models, sample_data):
        """Test feature building from player data."""
        X = baseline_models.build_features(sample_data)
        
        # Check that features were created
        assert len(X) == len(sample_data)
        assert len(X.columns) > 0
        
        # Check for rolling features
        assert any('rolling' in col for col in X.columns)
        
        # Check for role features
        assert any('role' in col for col in X.columns)
        
        # Check no NaN values
        assert not X.isna().any().any()
    
    def test_train_ridge(self, baseline_models, sample_data):
        """Test Ridge regression training."""
        X = baseline_models.build_features(sample_data)
        y = sample_data['PTS'].values
        
        model = baseline_models.train_ridge(X, y)
        
        # Check model was trained
        assert model is not None
        assert hasattr(model, 'coef_')
        
        # Check predictions
        predictions = baseline_models.predict(model, X, model_type='ridge')
        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()
    
    def test_train_mlp(self, baseline_models, sample_data):
        """Test MLP training."""
        X = baseline_models.build_features(sample_data)
        y = sample_data['PTS'].values
        
        model = baseline_models.train_mlp(X, y)
        
        # Check model was trained
        assert model is not None
        assert hasattr(model, 'coefs_')
        
        # Check predictions
        predictions = baseline_models.predict(model, X, model_type='mlp')
        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()
    
    def test_train_xgboost(self, baseline_models, sample_data):
        """Test XGBoost training (if available)."""
        if not baseline_models.xgboost_available:
            pytest.skip("XGBoost not installed")
        
        X = baseline_models.build_features(sample_data)
        y = sample_data['PTS'].values
        
        model = baseline_models.train_xgboost(X, y)
        
        # Check model was trained
        assert model is not None
        
        # Check predictions
        predictions = baseline_models.predict(model, X, model_type='xgboost')
        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()
    
    def test_save_and_load_model(self, baseline_models, sample_data):
        """Test model saving and loading."""
        X = baseline_models.build_features(sample_data)
        y = sample_data['PTS'].values
        
        # Train a model
        model = baseline_models.train_ridge(X, y)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_model.joblib'
            metadata = {'stat': 'PTS', 'model_type': 'ridge'}
            baseline_models.save_model(model, model_path, metadata)
            
            # Check file was created
            assert model_path.exists()
            
            # Load model
            loaded_model, loaded_metadata = baseline_models.load_model(model_path)
            
            # Check metadata
            assert loaded_metadata['stat'] == 'PTS'
            assert loaded_metadata['model_type'] == 'ridge'
            
            # Check predictions match
            pred_original = baseline_models.predict(model, X, model_type='ridge')
            pred_loaded = baseline_models.predict(loaded_model, X, model_type='ridge')
            np.testing.assert_array_almost_equal(pred_original, pred_loaded)
    
    def test_train_all_models(self, baseline_models, sample_data):
        """Test training all models for multiple stats."""
        X = baseline_models.build_features(sample_data)
        
        # Prepare targets
        y_dict = {
            'PTS': sample_data['PTS'].values,
            'TRB': sample_data['TRB'].values,
            'AST': sample_data['AST'].values
        }
        
        # Train all models
        models_dict = baseline_models.train_all_models(X, y_dict)
        
        # Check structure
        assert 'PTS' in models_dict
        assert 'TRB' in models_dict
        assert 'AST' in models_dict
        
        # Check that ridge and mlp were trained
        assert 'ridge' in models_dict['PTS']
        assert 'mlp' in models_dict['PTS']
        
        # Check XGBoost if available
        if baseline_models.xgboost_available:
            assert 'xgboost' in models_dict['PTS']
