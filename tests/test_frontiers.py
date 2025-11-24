"""
Unit tests for frontier fitting module.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.frontiers.fit import FrontierFitter, FrontierModel, Halfspace


class TestHalfspace:
    """Tests for Halfspace class."""
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        normal = np.array([1.0, 2.0])
        offset = 3.5
        
        hs = Halfspace(normal=normal, offset=offset)
        hs_dict = hs.to_dict()
        hs_restored = Halfspace.from_dict(hs_dict)
        
        np.testing.assert_array_almost_equal(hs_restored.normal, normal)
        assert hs_restored.offset == offset


class TestFrontierModel:
    """Tests for FrontierModel class."""
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        model = FrontierModel(
            x_attr='usage',
            y_attr='efficiency',
            strata={'role': 'starter'},
            quantile=0.9,
            coefficients=np.array([10.0, 0.5]),
            x_range=(0.0, 1.0),
            y_range=(0.0, 100.0)
        )
        
        model_dict = model.to_dict()
        model_restored = FrontierModel.from_dict(model_dict)
        
        assert model_restored.x_attr == model.x_attr
        assert model_restored.y_attr == model.y_attr
        assert model_restored.strata == model.strata
        assert model_restored.quantile == model.quantile
        np.testing.assert_array_almost_equal(model_restored.coefficients, model.coefficients)
        assert model_restored.x_range == model.x_range
        assert model_restored.y_range == model.y_range


class TestFrontierFitter:
    """Tests for FrontierFitter class."""
    
    @pytest.fixture
    def toy_data(self):
        """Create toy dataset for testing."""
        np.random.seed(42)
        n = 100
        
        # Generate synthetic data with a frontier relationship
        # y = 50 + 20*x + noise, where higher x leads to higher y
        x_vals = np.random.uniform(0, 1, n)
        y_vals = 50 + 20 * x_vals + np.random.normal(0, 5, n)
        
        df = pd.DataFrame({
            'usage': x_vals,
            'efficiency': y_vals,
            'role': ['starter'] * 50 + ['bench'] * 50
        })
        
        return df
    
    def test_fit_frontier_basic(self, toy_data):
        """Test basic frontier fitting."""
        fitter = FrontierFitter(min_samples=10)
        
        model = fitter.fit_frontier(
            data=toy_data,
            x='usage',
            y='efficiency',
            strata={'role': 'starter'},
            quantile=0.9
        )
        
        assert model.x_attr == 'usage'
        assert model.y_attr == 'efficiency'
        assert model.strata == {'role': 'starter'}
        assert model.quantile == 0.9
        assert len(model.coefficients) == 2  # intercept + slope
        assert model.x_range[0] < model.x_range[1]
        assert model.y_range[0] < model.y_range[1]
    
    def test_fit_frontier_insufficient_data(self, toy_data):
        """Test that insufficient data raises error."""
        fitter = FrontierFitter(min_samples=200)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            fitter.fit_frontier(
                data=toy_data,
                x='usage',
                y='efficiency',
                strata={'role': 'starter'},
                quantile=0.9
            )
    
    def test_fit_frontier_missing_columns(self, toy_data):
        """Test that missing columns raise error."""
        fitter = FrontierFitter()
        
        with pytest.raises(ValueError, match="not found in data"):
            fitter.fit_frontier(
                data=toy_data,
                x='nonexistent',
                y='efficiency',
                strata={'role': 'starter'},
                quantile=0.9
            )
    
    def test_linearize_frontier(self, toy_data):
        """Test frontier linearization."""
        fitter = FrontierFitter(min_samples=10)
        
        model = fitter.fit_frontier(
            data=toy_data,
            x='usage',
            y='efficiency',
            strata={'role': 'starter'},
            quantile=0.9
        )
        
        halfspaces = fitter.linearize_frontier(model, n_segments=5)
        
        # Should have n_segments + 3 halfspaces (segments + 3 boundary constraints)
        assert len(halfspaces) == 5 + 3
        
        # Check that all halfspaces have valid structure
        for hs in halfspaces:
            assert isinstance(hs, Halfspace)
            assert len(hs.normal) == 2
            assert isinstance(hs.offset, (float, np.floating))
    
    def test_save_and_load_frontier(self, toy_data):
        """Test saving and loading frontier models."""
        fitter = FrontierFitter(min_samples=10)
        
        model = fitter.fit_frontier(
            data=toy_data,
            x='usage',
            y='efficiency',
            strata={'role': 'starter'},
            quantile=0.9
        )
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_frontier.pkl'
            fitter.save_frontier(model, str(path))
            
            # Load and verify
            loaded_model = fitter.load_frontier(str(path))
            
            assert loaded_model.x_attr == model.x_attr
            assert loaded_model.y_attr == model.y_attr
            assert loaded_model.strata == model.strata
            assert loaded_model.quantile == model.quantile
            np.testing.assert_array_almost_equal(
                loaded_model.coefficients,
                model.coefficients
            )
            assert loaded_model.x_range == model.x_range
            assert loaded_model.y_range == model.y_range
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        fitter = FrontierFitter()
        
        with pytest.raises(FileNotFoundError):
            fitter.load_frontier('nonexistent_file.pkl')
