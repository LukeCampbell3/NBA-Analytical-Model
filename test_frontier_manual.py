"""
Manual test script for frontier fitting module.
Run this after installing dependencies: pip install -r requirements.txt
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.frontiers.fit import FrontierFitter, FrontierModel, Halfspace


def test_halfspace():
    """Test Halfspace serialization."""
    print("Testing Halfspace...")
    normal = np.array([1.0, 2.0])
    offset = 3.5
    
    hs = Halfspace(normal=normal, offset=offset)
    hs_dict = hs.to_dict()
    hs_restored = Halfspace.from_dict(hs_dict)
    
    assert np.allclose(hs_restored.normal, normal)
    assert hs_restored.offset == offset
    print("✓ Halfspace serialization works")


def test_frontier_model():
    """Test FrontierModel serialization."""
    print("\nTesting FrontierModel...")
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
    assert np.allclose(model_restored.coefficients, model.coefficients)
    assert model_restored.x_range == model.x_range
    assert model_restored.y_range == model.y_range
    print("✓ FrontierModel serialization works")


def test_frontier_fitting():
    """Test frontier fitting on toy data."""
    print("\nTesting FrontierFitter...")
    
    # Create toy dataset
    np.random.seed(42)
    n = 100
    x_vals = np.random.uniform(0, 1, n)
    y_vals = 50 + 20 * x_vals + np.random.normal(0, 5, n)
    
    df = pd.DataFrame({
        'usage': x_vals,
        'efficiency': y_vals,
        'role': ['starter'] * 50 + ['bench'] * 50
    })
    
    # Fit frontier
    fitter = FrontierFitter(min_samples=10)
    model = fitter.fit_frontier(
        data=df,
        x='usage',
        y='efficiency',
        strata={'role': 'starter'},
        quantile=0.9
    )
    
    assert model.x_attr == 'usage'
    assert model.y_attr == 'efficiency'
    assert model.strata == {'role': 'starter'}
    assert model.quantile == 0.9
    assert len(model.coefficients) == 2
    print(f"✓ Fitted frontier with coefficients: {model.coefficients}")
    
    # Test linearization
    halfspaces = fitter.linearize_frontier(model, n_segments=5)
    assert len(halfspaces) == 8  # 5 segments + 3 boundary constraints
    print(f"✓ Generated {len(halfspaces)} halfspaces")
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'test_frontier.pkl'
        fitter.save_frontier(model, str(path))
        loaded_model = fitter.load_frontier(str(path))
        
        assert loaded_model.x_attr == model.x_attr
        assert np.allclose(loaded_model.coefficients, model.coefficients)
        print("✓ Save/load works correctly")


def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")
    
    df = pd.DataFrame({
        'x': np.random.rand(5),
        'y': np.random.rand(5),
        'role': ['starter'] * 5
    })
    
    fitter = FrontierFitter(min_samples=100)
    
    try:
        fitter.fit_frontier(df, 'x', 'y', {'role': 'starter'}, 0.9)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Insufficient data" in str(e)
        print("✓ Insufficient data error handled correctly")
    
    fitter2 = FrontierFitter(min_samples=1)
    try:
        fitter2.fit_frontier(df, 'nonexistent', 'y', {'role': 'starter'}, 0.9)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found in data" in str(e)
        print("✓ Missing column error handled correctly")
    
    try:
        fitter.load_frontier('nonexistent.pkl')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ Missing file error handled correctly")


if __name__ == '__main__':
    print("=" * 60)
    print("Running Frontier Fitting Tests")
    print("=" * 60)
    
    try:
        test_halfspace()
        test_frontier_model()
        test_frontier_fitting()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
