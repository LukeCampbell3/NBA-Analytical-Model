"""
Unit tests for benchmarking module.

Tests cover:
- Metric computation on toy data
- Comparison table structure
- Efficiency measurement
"""

import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.compare import BenchmarkRunner


class TestBenchmarkRunner:
    """Test suite for BenchmarkRunner class."""
    
    @pytest.fixture
    def benchmark_runner(self):
        """Create a BenchmarkRunner instance for testing."""
        return BenchmarkRunner()
    
    @pytest.fixture
    def toy_predictions(self):
        """Create toy prediction data for testing."""
        n_samples = 100
        
        # True values
        y_true = pd.DataFrame({
            'PTS': np.random.randint(10, 40, n_samples),
            'REB': np.random.randint(2, 15, n_samples),
            'AST': np.random.randint(1, 12, n_samples),
            'STL': np.random.randint(0, 4, n_samples),
            'BLK': np.random.randint(0, 3, n_samples)
        })
        
        # Model predictions (distributions)
        predictions = {}
        
        # Model 1: Good predictions (close to true)
        predictions['model_good'] = {
            stat: np.random.normal(y_true[stat].values[:, np.newaxis], 3, (n_samples, 1000))
            for stat in y_true.columns
        }
        
        # Model 2: Poor predictions (far from true)
        predictions['model_poor'] = {
            stat: np.random.normal(y_true[stat].values[:, np.newaxis] + 10, 8, (n_samples, 1000))
            for stat in y_true.columns
        }
        
        # Model 3: Point predictions (for baseline comparison)
        predictions['model_point'] = {
            stat: y_true[stat].values + np.random.normal(0, 2, n_samples)
            for stat in y_true.columns
        }
        
        return y_true, predictions
    
    def test_init(self, benchmark_runner):
        """Test BenchmarkRunner initialization."""
        assert benchmark_runner.metrics == [
            'mae', 'rmse', 'crps', 'coverage_50', 'coverage_80',
            'ece', 'tail_recall_p95'
        ]
        assert benchmark_runner.efficiency_metrics == [
            'train_time_sec', 'infer_time_ms_per_player',
            'adaptation_time_ms', 'memory_mb'
        ]
    
    def test_compute_mae(self, benchmark_runner):
        """Test MAE computation."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])
        
        mae = benchmark_runner._compute_mae(y_true, y_pred)
        
        expected_mae = np.mean(np.abs(y_true - y_pred))
        assert np.isclose(mae, expected_mae)
        assert mae == 2.0
    
    def test_compute_rmse(self, benchmark_runner):
        """Test RMSE computation."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])
        
        rmse = benchmark_runner._compute_rmse(y_true, y_pred)
        
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert np.isclose(rmse, expected_rmse)
    
    def test_compute_crps(self, benchmark_runner):
        """Test CRPS computation."""
        y_true = np.array([25, 30, 35])
        
        # Create distributions (3 samples, 100 trials each)
        y_samples = np.array([
            np.random.normal(25, 3, 100),
            np.random.normal(30, 3, 100),
            np.random.normal(35, 3, 100)
        ])
        
        crps = benchmark_runner._compute_crps(y_true, y_samples)
        
        # CRPS should be non-negative
        assert crps >= 0
        
        # CRPS should be reasonable (not too large)
        assert crps < 10
    
    def test_compute_coverage(self, benchmark_runner):
        """Test coverage computation."""
        y_true = np.array([25, 30, 35, 40, 45])
        
        # Create distributions where true values are at median
        y_samples = np.array([
            np.random.normal(25, 5, 1000),
            np.random.normal(30, 5, 1000),
            np.random.normal(35, 5, 1000),
            np.random.normal(40, 5, 1000),
            np.random.normal(45, 5, 1000)
        ])
        
        coverage_50 = benchmark_runner._compute_coverage(y_true, y_samples, alpha=0.50)
        coverage_80 = benchmark_runner._compute_coverage(y_true, y_samples, alpha=0.80)
        
        # Coverage should be close to target alpha
        assert 0.3 <= coverage_50 <= 0.7  # Allow some variance
        assert 0.6 <= coverage_80 <= 1.0
        
        # Higher alpha should have higher coverage
        assert coverage_80 >= coverage_50
    
    def test_compute_ece(self, benchmark_runner):
        """Test ECE (Expected Calibration Error) computation."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.75, 0.15, 0.25])
        
        ece = benchmark_runner._compute_ece(y_true, y_prob, n_bins=5)
        
        # ECE should be between 0 and 1
        assert 0 <= ece <= 1
    
    def test_compute_tail_recall(self, benchmark_runner):
        """Test tail recall computation."""
        y_true = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        
        # Create distributions
        y_samples = np.array([
            np.random.normal(val, 5, 100) for val in y_true
        ])
        
        tail_recall = benchmark_runner._compute_tail_recall(
            y_true, y_samples, percentile=95
        )
        
        # Tail recall should be between 0 and 1
        assert 0 <= tail_recall <= 1
    
    def test_compute_accuracy_metrics_distributions(
        self, benchmark_runner, toy_predictions
    ):
        """Test accuracy metrics computation with distributions."""
        y_true, predictions = toy_predictions
        
        metrics = benchmark_runner.compute_accuracy_metrics(
            y_true, predictions['model_good']
        )
        
        # Check that all stats are present
        for stat in y_true.columns:
            assert stat in metrics
        
        # Check that all metrics are computed for each stat
        for stat in y_true.columns:
            assert 'mae' in metrics[stat]
            assert 'rmse' in metrics[stat]
            assert 'crps' in metrics[stat]
            assert 'coverage_50' in metrics[stat]
            assert 'coverage_80' in metrics[stat]
            assert 'ece' in metrics[stat]
            assert 'tail_recall_p95' in metrics[stat]
            
            # Check that metrics are reasonable
            assert metrics[stat]['mae'] >= 0
            assert metrics[stat]['rmse'] >= metrics[stat]['mae']
            assert 0 <= metrics[stat]['coverage_50'] <= 1
            assert 0 <= metrics[stat]['coverage_80'] <= 1
            assert 0 <= metrics[stat]['ece'] <= 1
            assert 0 <= metrics[stat]['tail_recall_p95'] <= 1
    
    def test_compute_accuracy_metrics_point_predictions(
        self, benchmark_runner, toy_predictions
    ):
        """Test accuracy metrics with point predictions."""
        y_true, predictions = toy_predictions
        
        metrics = benchmark_runner.compute_accuracy_metrics(
            y_true, predictions['model_point']
        )
        
        # Should still compute MAE and RMSE
        for stat in y_true.columns:
            assert 'mae' in metrics[stat]
            assert 'rmse' in metrics[stat]
            assert metrics[stat]['mae'] >= 0
            assert metrics[stat]['rmse'] >= 0
    
    def test_compute_accuracy_metrics_good_vs_poor(
        self, benchmark_runner, toy_predictions
    ):
        """Test that good model has better metrics than poor model."""
        y_true, predictions = toy_predictions
        
        metrics_good = benchmark_runner.compute_accuracy_metrics(
            y_true, predictions['model_good']
        )
        metrics_poor = benchmark_runner.compute_accuracy_metrics(
            y_true, predictions['model_poor']
        )
        
        # Good model should have lower MAE and RMSE
        for stat in y_true.columns:
            assert metrics_good[stat]['mae'] < metrics_poor[stat]['mae']
            assert metrics_good[stat]['rmse'] < metrics_poor[stat]['rmse']
    
    def test_compute_efficiency_metrics(self, benchmark_runner):
        """Test efficiency metrics computation."""
        
        def dummy_inference_fn(n_players=10):
            """Dummy inference function for testing."""
            import time
            time.sleep(0.01)  # Simulate some work
            return {'result': 'done'}
        
        metrics = benchmark_runner.compute_efficiency_metrics(
            model_name='test_model',
            inference_fn=dummy_inference_fn,
            n_players=5
        )
        
        # Check that all efficiency metrics are present
        assert 'train_time_sec' in metrics
        assert 'infer_time_ms_per_player' in metrics
        assert 'memory_mb' in metrics
        
        # Check that metrics are reasonable
        assert metrics['infer_time_ms_per_player'] > 0
        assert metrics['memory_mb'] > 0
    
    def test_compare_models(self, benchmark_runner, toy_predictions):
        """Test model comparison table generation."""
        y_true, predictions = toy_predictions
        
        # Compute metrics for all models
        results = {}
        for model_name, preds in predictions.items():
            results[model_name] = benchmark_runner.compute_accuracy_metrics(
                y_true, preds
            )
        
        # Generate comparison table
        comparison_df = benchmark_runner.compare_models(results)
        
        # Check table structure
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == len(predictions)  # One row per model
        
        # Check that model names are in index or columns
        assert all(model in comparison_df.index or model in comparison_df.columns 
                  for model in predictions.keys())
        
        # Check that metrics are present
        assert 'PTS_mae' in comparison_df.columns or 'mae' in str(comparison_df.columns)
    
    def test_compare_models_with_efficiency(self, benchmark_runner, toy_predictions):
        """Test model comparison with efficiency metrics."""
        y_true, predictions = toy_predictions
        
        # Compute accuracy metrics
        results = {}
        for model_name, preds in predictions.items():
            results[model_name] = benchmark_runner.compute_accuracy_metrics(
                y_true, preds
            )
        
        # Add dummy efficiency metrics
        efficiency = {
            'model_good': {
                'train_time_sec': 10.5,
                'infer_time_ms_per_player': 2.0,
                'memory_mb': 150.0
            },
            'model_poor': {
                'train_time_sec': 5.2,
                'infer_time_ms_per_player': 1.5,
                'memory_mb': 100.0
            },
            'model_point': {
                'train_time_sec': 2.1,
                'infer_time_ms_per_player': 0.5,
                'memory_mb': 50.0
            }
        }
        
        # Generate comparison with efficiency
        comparison_df = benchmark_runner.compare_models(
            results, efficiency_metrics=efficiency
        )
        
        # Check that efficiency metrics are included
        assert 'train_time_sec' in comparison_df.columns or \
               any('train_time' in str(col) for col in comparison_df.columns)
    
    def test_run_eval_window(self, benchmark_runner, toy_predictions):
        """Test evaluation window execution."""
        y_true, predictions = toy_predictions
        
        # Create a simple evaluation window
        window_df = y_true.copy()
        window_df['game_id'] = [f'game_{i}' for i in range(len(y_true))]
        window_df['player_id'] = [f'player_{i % 10}' for i in range(len(y_true))]
        
        # Mock model functions
        def model_good_fn(df):
            return predictions['model_good']
        
        def model_poor_fn(df):
            return predictions['model_poor']
        
        models = {
            'model_good': model_good_fn,
            'model_poor': model_poor_fn
        }
        
        # Run evaluation
        results = benchmark_runner.run_eval_window(
            window_df=window_df,
            models=models,
            target_stats=['PTS', 'REB', 'AST']
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'model_good' in results
        assert 'model_poor' in results
        
        # Check that metrics are computed
        for model_name in models.keys():
            assert 'accuracy' in results[model_name]
            assert isinstance(results[model_name]['accuracy'], dict)
    
    def test_ablation_study(self, benchmark_runner):
        """Test ablation study execution."""
        
        # Create simple config grid
        config_grid = [
            {'blend_weight': 0.5, 'state_amplitude': 1.0},
            {'blend_weight': 0.6, 'state_amplitude': 1.1},
            {'blend_weight': 0.7, 'state_amplitude': 1.2}
        ]
        
        # Mock evaluation function
        def eval_fn(config):
            # Return dummy metrics based on config
            return {
                'PTS_mae': 5.0 + config['blend_weight'],
                'REB_mae': 2.0 + config['state_amplitude']
            }
        
        # Run ablation study
        ablation_df = benchmark_runner.ablation_study(
            config_grid=config_grid,
            eval_fn=eval_fn
        )
        
        # Check results
        assert isinstance(ablation_df, pd.DataFrame)
        assert len(ablation_df) == len(config_grid)
        
        # Check that config parameters are in DataFrame
        assert 'blend_weight' in ablation_df.columns
        assert 'state_amplitude' in ablation_df.columns
        
        # Check that metrics are in DataFrame
        assert 'PTS_mae' in ablation_df.columns
        assert 'REB_mae' in ablation_df.columns
    
    def test_compute_statistical_significance(self, benchmark_runner):
        """Test statistical significance testing."""
        
        # Create two sets of predictions
        y_true = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
        
        errors_model1 = np.abs(y_true - (y_true + np.random.normal(0, 2, 10)))
        errors_model2 = np.abs(y_true - (y_true + np.random.normal(0, 5, 10)))
        
        # Test significance
        p_value, is_significant = benchmark_runner.compute_statistical_significance(
            errors_model1, errors_model2, alpha=0.05
        )
        
        # Check results
        assert 0 <= p_value <= 1
        assert isinstance(is_significant, bool)
    
    def test_generate_benchmark_summary(self, benchmark_runner, toy_predictions):
        """Test benchmark summary generation."""
        y_true, predictions = toy_predictions
        
        # Compute metrics for all models
        results = {}
        for model_name, preds in predictions.items():
            results[model_name] = benchmark_runner.compute_accuracy_metrics(
                y_true, preds
            )
        
        # Generate summary
        summary = benchmark_runner.generate_benchmark_summary(results)
        
        # Check summary structure
        assert isinstance(summary, dict)
        assert 'best_model' in summary
        assert 'metric_comparison' in summary
        
        # Check that best model is identified
        assert summary['best_model'] in predictions.keys()
    
    def test_save_and_load_results(self, benchmark_runner, toy_predictions, tmp_path):
        """Test saving and loading benchmark results."""
        y_true, predictions = toy_predictions
        
        # Compute metrics
        results = {}
        for model_name, preds in predictions.items():
            results[model_name] = benchmark_runner.compute_accuracy_metrics(
                y_true, preds
            )
        
        # Save results
        results_path = tmp_path / "benchmark_results.json"
        benchmark_runner.save_results(results, str(results_path))
        
        # Check file exists
        assert results_path.exists()
        
        # Load results
        loaded_results = benchmark_runner.load_results(str(results_path))
        
        # Check that loaded results match original
        assert loaded_results.keys() == results.keys()
        
        for model_name in results.keys():
            assert model_name in loaded_results


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
