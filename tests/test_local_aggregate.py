"""
Unit tests for local model aggregation and blending.

Tests cover:
- Aggregation of event probabilities to box expectations
- Blending of global and local distributions
- Recalibration logic
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.local_models.aggregate import LocalAggregator


class TestLocalAggregator:
    """Test suite for LocalAggregator class."""
    
    @pytest.fixture
    def aggregator(self):
        """Create a LocalAggregator instance with default weights."""
        return LocalAggregator(global_weight=0.6, local_weight=0.4)
    
    @pytest.fixture
    def sample_local_probs(self):
        """Create sample local event probabilities."""
        return {
            'rebound_prob': np.random.uniform(0.1, 0.5, 50),
            'assist_prob': np.random.uniform(0.05, 0.3, 80),
            'shot_prob': np.random.uniform(0.3, 0.6, 100)
        }
    
    @pytest.fixture
    def sample_global_summary(self):
        """Create sample global simulation distributions."""
        n_trials = 1000
        return {
            'PTS': np.random.normal(25, 5, n_trials),
            'REB': np.random.normal(8, 2, n_trials),
            'AST': np.random.normal(6, 2, n_trials),
            'FGA': np.random.normal(18, 3, n_trials),
            'FGM': np.random.normal(9, 2, n_trials)
        }
    
    def test_init_default_weights(self):
        """Test initialization with default weights."""
        agg = LocalAggregator()
        assert agg.global_weight == 0.6
        assert agg.local_weight == 0.4
        assert agg.recalibration_method == 'variance_scaling'
    
    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        agg = LocalAggregator(global_weight=0.7, local_weight=0.3)
        assert agg.global_weight == 0.7
        assert agg.local_weight == 0.3
    
    def test_init_invalid_weights(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            LocalAggregator(global_weight=0.5, local_weight=0.6)
    
    def test_local_to_box_expectations(self, aggregator, sample_local_probs):
        """Test conversion of local probabilities to box expectations."""
        expectations = aggregator.local_to_box_expectations(
            local_probs=sample_local_probs,
            minutes=32.0,
            usage=0.25,
            pace=100.0,
            possessions_per_minute=1.0
        )
        
        # Check that expected stats are present
        assert 'REB' in expectations
        assert 'AST' in expectations
        assert 'PTS' in expectations
        assert 'FGA' in expectations
        assert 'FGM' in expectations
        
        # Check that values are non-negative
        for stat, value in expectations.items():
            assert value >= 0, f"{stat} should be non-negative"
        
        # Check reasonable ranges
        assert expectations['PTS'] > 0
        assert expectations['REB'] > 0
        assert expectations['AST'] > 0
        assert expectations['FGA'] > expectations['FGM']  # Attempts > Makes
    
    def test_local_to_box_expectations_empty_probs(self, aggregator):
        """Test with empty probability arrays."""
        empty_probs = {
            'rebound_prob': np.array([]),
            'assist_prob': np.array([]),
            'shot_prob': np.array([])
        }
        
        expectations = aggregator.local_to_box_expectations(
            local_probs=empty_probs,
            minutes=30.0,
            usage=0.20,
            pace=100.0
        )
        
        # Should return zeros for all stats
        assert expectations['REB'] == 0.0
        assert expectations['AST'] == 0.0
        assert expectations['PTS'] == 0.0
    
    def test_blend_global_local(self, aggregator, sample_global_summary):
        """Test blending of global and local distributions."""
        local_expect = {
            'PTS': 28.0,
            'REB': 9.0,
            'AST': 7.0,
            'FGA': 20.0,
            'FGM': 10.0
        }
        
        blended = aggregator.blend_global_local(
            global_summary=sample_global_summary,
            local_expect=local_expect
        )
        
        # Check that all stats are present
        for stat in local_expect.keys():
            assert stat in blended
        
        # Check that blended distributions have same length as global
        for stat in local_expect.keys():
            assert len(blended[stat]) == len(sample_global_summary[stat])
        
        # Check that blended mean is between global and local
        for stat in local_expect.keys():
            global_mean = np.mean(sample_global_summary[stat])
            local_value = local_expect[stat]
            blended_mean = np.mean(blended[stat])
            
            # Blended mean should be weighted average
            expected_mean = 0.6 * global_mean + 0.4 * local_value
            assert np.isclose(blended_mean, expected_mean, rtol=0.1)
    
    def test_blend_global_local_no_common_stats(self, aggregator):
        """Test blending with no common statistics raises error."""
        global_summary = {'PTS': np.random.normal(25, 5, 100)}
        local_expect = {'DIFFERENT_STAT': 10.0}
        
        with pytest.raises(ValueError, match="No common statistics"):
            aggregator.blend_global_local(global_summary, local_expect)
    
    def test_blend_global_local_variance_scaling(self, sample_global_summary):
        """Test variance scaling recalibration method."""
        agg = LocalAggregator(
            global_weight=0.6,
            local_weight=0.4,
            recalibration_method='variance_scaling'
        )
        
        local_expect = {'PTS': 28.0}
        blended = agg.blend_global_local(
            global_summary={'PTS': sample_global_summary['PTS']},
            local_expect=local_expect
        )
        
        # Variance should be reduced (more certainty from local info)
        global_std = np.std(sample_global_summary['PTS'])
        blended_std = np.std(blended['PTS'])
        assert blended_std < global_std
    
    def test_blend_global_local_bootstrap(self, sample_global_summary):
        """Test bootstrap recalibration method."""
        agg = LocalAggregator(
            global_weight=0.6,
            local_weight=0.4,
            recalibration_method='bootstrap'
        )
        
        local_expect = {'PTS': 28.0}
        blended = agg.blend_global_local(
            global_summary={'PTS': sample_global_summary['PTS']},
            local_expect=local_expect
        )
        
        # Should have same number of samples
        assert len(blended['PTS']) == len(sample_global_summary['PTS'])
        
        # Mean should be close to weighted average
        global_mean = np.mean(sample_global_summary['PTS'])
        blended_mean = np.mean(blended['PTS'])
        expected_mean = 0.6 * global_mean + 0.4 * 28.0
        assert np.isclose(blended_mean, expected_mean, rtol=0.2)
    
    def test_blend_global_local_no_recalibration(self, sample_global_summary):
        """Test no recalibration method."""
        agg = LocalAggregator(
            global_weight=0.6,
            local_weight=0.4,
            recalibration_method='none'
        )
        
        local_expect = {'PTS': 28.0}
        blended = agg.blend_global_local(
            global_summary={'PTS': sample_global_summary['PTS']},
            local_expect=local_expect
        )
        
        # Should still blend means correctly
        global_mean = np.mean(sample_global_summary['PTS'])
        blended_mean = np.mean(blended['PTS'])
        expected_mean = 0.6 * global_mean + 0.4 * 28.0
        assert np.isclose(blended_mean, expected_mean, rtol=0.1)
    
    def test_compute_blend_diagnostics(self, aggregator, sample_global_summary):
        """Test computation of blending diagnostics."""
        local_expect = {'PTS': 28.0, 'REB': 9.0}
        blended = aggregator.blend_global_local(
            global_summary=sample_global_summary,
            local_expect=local_expect
        )
        
        diagnostics = aggregator.compute_blend_diagnostics(
            global_summary=sample_global_summary,
            local_expect=local_expect,
            blended=blended
        )
        
        # Check that diagnostics are computed for blended stats
        assert 'PTS' in diagnostics
        assert 'REB' in diagnostics
        
        # Check diagnostic metrics
        for stat in ['PTS', 'REB']:
            assert 'shift_magnitude' in diagnostics[stat]
            assert 'uncertainty_ratio' in diagnostics[stat]
            assert 'blend_quality' in diagnostics[stat]
            assert 'global_mean' in diagnostics[stat]
            assert 'blended_mean' in diagnostics[stat]
            
            # Blend quality should be high (close to 1.0)
            assert diagnostics[stat]['blend_quality'] > 0.8
    
    def test_aggregate_and_blend_full_pipeline(
        self, 
        aggregator, 
        sample_local_probs, 
        sample_global_summary
    ):
        """Test complete aggregation and blending pipeline."""
        result = aggregator.aggregate_and_blend(
            global_summary=sample_global_summary,
            local_probs=sample_local_probs,
            minutes=32.0,
            usage=0.25,
            pace=100.0
        )
        
        # Check result structure
        assert 'blended' in result
        assert 'local_expect' in result
        assert 'diagnostics' in result
        
        # Check blended distributions
        assert isinstance(result['blended'], dict)
        assert len(result['blended']) > 0
        
        # Check local expectations
        assert isinstance(result['local_expect'], dict)
        assert 'PTS' in result['local_expect']
        assert 'REB' in result['local_expect']
        assert 'AST' in result['local_expect']
        
        # Check diagnostics
        assert isinstance(result['diagnostics'], dict)
        for stat in result['blended'].keys():
            if stat in sample_global_summary:
                assert stat in result['diagnostics']


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
