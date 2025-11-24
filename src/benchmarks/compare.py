"""
Benchmarking and model comparison module.

This module provides comprehensive benchmarking functionality to compare
the capability-region simulation approach against traditional ML baselines.
Computes accuracy, efficiency, and calibration metrics across multiple
evaluation windows.
"""

import os
import time
import tracemalloc
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.integrate import trapezoid
from tqdm import tqdm


def _run_model_worker(args: Tuple) -> Tuple[str, Any]:
    """
    Worker function for parallel model execution.
    
    This function is defined at module level to be picklable for multiprocessing.
    
    Args:
        args: Tuple containing (model_name, model_fn, window_df, cfg)
    
    Returns:
        Tuple of (model_name, predictions)
    """
    model_name, model_fn, window_df, cfg = args
    
    try:
        preds = model_fn(window_df, cfg)
        return (model_name, preds)
    except Exception as e:
        print(f"Error running {model_name}: {e}")
        return (model_name, None)


@dataclass
class EvaluationWindow:
    """Configuration for an evaluation window.
    
    Attributes:
        name: Window identifier (e.g., 'rolling_30_games', 'monthly', 'playoffs_only')
        start_date: Start date for evaluation
        end_date: End date for evaluation
        description: Human-readable description
    """
    name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs.
    
    Attributes:
        evaluation_windows: List of evaluation windows to test
        models_to_compare: List of model names to include
        target_stats: Statistics to evaluate
        n_bootstrap: Number of bootstrap samples for confidence intervals
        confidence_level: Confidence level for intervals (default: 0.95)
    """
    evaluation_windows: List[EvaluationWindow]
    models_to_compare: List[str]
    target_stats: List[str]
    n_bootstrap: int = 1000
    confidence_level: float = 0.95


class BenchmarkRunner:
    """
    Comprehensive benchmarking for NBA prediction models.
    
    This class evaluates models on multiple metrics:
    - Accuracy: MAE, RMSE, CRPS, coverage, ECE, tail recall
    - Efficiency: Training time, inference time, adaptation time, memory
    - Overall: Spearman correlation, decision gain
    
    Supports comparison across multiple evaluation windows and model types.
    """
    
    # Default target statistics
    DEFAULT_TARGET_STATS = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
    
    # Default model names
    DEFAULT_MODELS = [
        'original_global_only',
        'local_only',
        'blended_global_plus_local',
        'baselines_ridge',
        'baselines_xgboost',
        'baselines_mlp'
    ]
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        output_dir: Union[str, Path] = "outputs/benchmarks",
        n_workers: Optional[int] = None,
        enable_progress: bool = True
    ):
        """
        Initialize BenchmarkRunner.
        
        Args:
            config: Benchmark configuration (uses defaults if None)
            output_dir: Directory for saving benchmark results
            n_workers: Number of parallel workers (None = use config/env, 1 = no parallelization)
            enable_progress: Whether to show progress bars (default: True)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.results: Dict[str, Any] = {}
        
        # Load configuration for parallelization settings
        try:
            with open("configs/default.yaml", 'r') as f:
                yaml_config = yaml.safe_load(f)
            parallel_config = yaml_config.get('parallelization', {})
        except Exception:
            parallel_config = {}
        
        # Determine number of workers
        if n_workers is not None:
            self.n_workers = n_workers
        else:
            # Try environment variable first
            env_workers = os.environ.get('NBA_PRED_N_WORKERS')
            if env_workers is not None:
                self.n_workers = int(env_workers)
            else:
                # Use config value
                config_workers = parallel_config.get('n_workers')
                if config_workers is None:
                    # Use all available cores
                    self.n_workers = cpu_count()
                else:
                    self.n_workers = config_workers
        
        # Enable progress bars
        if enable_progress is not None:
            self.enable_progress = enable_progress
        else:
            self.enable_progress = parallel_config.get('enable_progress_bars', True)
    
    def run_eval_window(
        self,
        window_df: pd.DataFrame,
        models: Dict[str, Callable],
        cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate models on a configured evaluation window.
        
        Runs all specified models on the evaluation window and computes
        predictions for comparison.
        
        Args:
            window_df: DataFrame with evaluation data
                Required columns: player_id, game_id, Date, and target stats
            models: Dictionary mapping model names to prediction functions
                Each function should take (window_df, cfg) and return predictions
            cfg: Configuration dictionary with model-specific parameters
        
        Returns:
            Dictionary with structure:
                {
                    'window_info': {...},
                    'predictions': {model_name: predictions_df},
                    'ground_truth': ground_truth_df
                }
        
        Raises:
            ValueError: If window_df is empty or missing required columns
        """
        if window_df.empty:
            raise ValueError("Evaluation window DataFrame is empty")
        
        # Validate required columns
        required_cols = ['player_id', 'game_id']
        missing_cols = [col for col in required_cols if col not in window_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract ground truth
        target_stats = cfg.get('target_stats', self.DEFAULT_TARGET_STATS)
        available_stats = [stat for stat in target_stats if stat in window_df.columns]
        
        ground_truth = window_df[['player_id', 'game_id'] + available_stats].copy()
        
        # Run each model (with optional parallelization)
        predictions = self._run_models_on_window(
            models=models,
            window_df=window_df,
            cfg=cfg
        )
        
        # Compile window info
        window_info = {
            'n_games': len(window_df),
            'n_players': window_df['player_id'].nunique(),
            'date_range': (
                window_df['Date'].min() if 'Date' in window_df.columns else None,
                window_df['Date'].max() if 'Date' in window_df.columns else None
            ),
            'target_stats': available_stats
        }
        
        return {
            'window_info': window_info,
            'predictions': predictions,
            'ground_truth': ground_truth
        }
    
    def _run_models_on_window(
        self,
        models: Dict[str, Callable],
        window_df: pd.DataFrame,
        cfg: Dict[str, Any],
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Run all models on an evaluation window, optionally in parallel.
        
        Args:
            models: Dictionary mapping model names to prediction functions
            window_df: Evaluation window DataFrame
            cfg: Configuration dictionary
            parallel: Whether to use parallel processing (default: True)
        
        Returns:
            Dictionary mapping model names to predictions
        """
        use_parallel = parallel and self.n_workers > 1 and len(models) > 1
        
        if use_parallel:
            # Parallel execution
            predictions = self._run_models_parallel(models, window_df, cfg)
        else:
            # Sequential execution
            predictions = {}
            
            # Create progress bar if enabled
            iterator = models.items()
            if self.enable_progress:
                iterator = tqdm(
                    iterator,
                    total=len(models),
                    desc="Running models",
                    unit="model"
                )
            
            for model_name, model_fn in iterator:
                if self.enable_progress:
                    print(f"Running {model_name} on evaluation window...")
                try:
                    preds = model_fn(window_df, cfg)
                    predictions[model_name] = preds
                except Exception as e:
                    print(f"Error running {model_name}: {e}")
                    predictions[model_name] = None
        
        return predictions
    
    def _run_models_parallel(
        self,
        models: Dict[str, Callable],
        window_df: pd.DataFrame,
        cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run models in parallel using multiprocessing.
        
        Args:
            models: Dictionary of model functions
            window_df: Evaluation window DataFrame
            cfg: Configuration dictionary
        
        Returns:
            Dictionary mapping model names to predictions
        """
        # Prepare arguments for parallel execution
        args_list = [
            (model_name, model_fn, window_df, cfg)
            for model_name, model_fn in models.items()
        ]
        
        # Use multiprocessing pool
        with Pool(processes=self.n_workers) as pool:
            if self.enable_progress:
                # Use imap with tqdm for progress tracking
                results = list(tqdm(
                    pool.imap(_run_model_worker, args_list),
                    total=len(args_list),
                    desc="Running models (parallel)",
                    unit="model"
                ))
            else:
                # Use map without progress tracking
                results = pool.map(_run_model_worker, args_list)
        
        # Convert list of tuples to dictionary
        predictions = {model_name: preds for model_name, preds in results}
        
        return predictions
    
    def compute_accuracy_metrics(
        self,
        y_true: pd.DataFrame,
        preds: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute accuracy metrics for all models and statistics.
        
        Computes:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - CRPS: Continuous Ranked Probability Score
        - coverage_50: Coverage of 50% prediction interval
        - coverage_80: Coverage of 80% prediction interval
        - ECE: Expected Calibration Error
        - tail_recall_p95: Recall for 95th percentile events
        
        Args:
            y_true: DataFrame with ground truth values
                Columns: player_id, game_id, and target statistics
            preds: Dictionary mapping model names to predictions
                Each prediction can be:
                - DataFrame with point predictions (same structure as y_true)
                - Dict with 'samples' key containing distribution samples
                - Dict with 'mean' and 'std' keys for parametric distributions
        
        Returns:
            Dictionary with structure:
                {model_name: {stat: {metric_name: value}}}
        
        Raises:
            ValueError: If y_true and predictions have mismatched indices
        """
        results = {}
        
        # Get target statistics
        stat_cols = [col for col in y_true.columns 
                     if col not in ['player_id', 'game_id', 'Date']]
        
        for model_name, pred_data in preds.items():
            if pred_data is None:
                continue
            
            results[model_name] = {}
            
            for stat in stat_cols:
                if stat not in y_true.columns:
                    continue
                
                # Extract ground truth for this stat
                y_stat = y_true[stat].values
                
                # Extract predictions (handle different formats)
                if isinstance(pred_data, pd.DataFrame):
                    # Point predictions
                    if stat not in pred_data.columns:
                        continue
                    y_pred = pred_data[stat].values
                    pred_samples = None
                elif isinstance(pred_data, dict):
                    # Distribution predictions
                    if 'samples' in pred_data and stat in pred_data['samples']:
                        pred_samples = pred_data['samples'][stat]
                        y_pred = np.mean(pred_samples, axis=1) if pred_samples.ndim > 1 else np.mean(pred_samples)
                    elif 'distributions' in pred_data and stat in pred_data['distributions']:
                        pred_samples = pred_data['distributions'][stat]
                        y_pred = np.mean(pred_samples, axis=1) if pred_samples.ndim > 1 else np.mean(pred_samples)
                    elif 'mean' in pred_data and stat in pred_data['mean']:
                        y_pred = pred_data['mean'][stat]
                        pred_samples = None
                    else:
                        continue
                else:
                    continue
                
                # Ensure arrays have same length
                min_len = min(len(y_stat), len(y_pred) if hasattr(y_pred, '__len__') else len(y_stat))
                y_stat = y_stat[:min_len]
                y_pred = y_pred[:min_len] if hasattr(y_pred, '__len__') else np.full(min_len, y_pred)
                
                # Compute metrics
                metrics = {}
                
                # MAE
                metrics['mae'] = np.mean(np.abs(y_stat - y_pred))
                
                # RMSE
                metrics['rmse'] = np.sqrt(np.mean((y_stat - y_pred) ** 2))
                
                # CRPS (if we have samples)
                if pred_samples is not None:
                    metrics['crps'] = self._compute_crps(y_stat, pred_samples)
                else:
                    metrics['crps'] = np.nan
                
                # Coverage (if we have samples)
                if pred_samples is not None:
                    metrics['coverage_50'] = self._compute_coverage(y_stat, pred_samples, 0.50)
                    metrics['coverage_80'] = self._compute_coverage(y_stat, pred_samples, 0.80)
                else:
                    metrics['coverage_50'] = np.nan
                    metrics['coverage_80'] = np.nan
                
                # ECE (if we have samples)
                if pred_samples is not None:
                    metrics['ece'] = self._compute_ece(y_stat, pred_samples)
                else:
                    metrics['ece'] = np.nan
                
                # Tail recall at 95th percentile
                metrics['tail_recall_p95'] = self._compute_tail_recall(
                    y_stat, y_pred, percentile=95
                )
                
                results[model_name][stat] = metrics
        
        return results
    
    def _compute_crps(
        self,
        y_true: np.ndarray,
        samples: np.ndarray
    ) -> float:
        """
        Compute Continuous Ranked Probability Score.
        
        CRPS measures the quality of probabilistic forecasts by comparing
        the predicted CDF to the empirical CDF of the observation.
        
        Args:
            y_true: Ground truth values (shape: [n])
            samples: Prediction samples (shape: [n, n_samples] or [n_samples])
        
        Returns:
            Mean CRPS across all predictions
        """
        # Handle different sample shapes
        if samples.ndim == 1:
            # Single prediction with multiple samples
            samples = samples.reshape(1, -1)
            y_true = np.array([y_true]) if np.isscalar(y_true) else y_true[:1]
        
        crps_values = []
        
        for i in range(len(y_true)):
            y_i = y_true[i]
            samples_i = samples[i] if samples.ndim > 1 else samples
            
            # Sort samples
            sorted_samples = np.sort(samples_i)
            
            # Compute empirical CDF
            n_samples = len(sorted_samples)
            cdf_values = np.arange(1, n_samples + 1) / n_samples
            
            # Compute CRPS using trapezoidal integration
            # CRPS = integral of (F(x) - 1{y <= x})^2 dx
            # Approximate using sorted samples
            heaviside = (sorted_samples >= y_i).astype(float)
            integrand = (cdf_values - heaviside) ** 2
            
            # Use trapezoidal rule
            crps_i = trapezoid(integrand, sorted_samples)
            crps_values.append(crps_i)
        
        return np.mean(crps_values)
    
    def _compute_coverage(
        self,
        y_true: np.ndarray,
        samples: np.ndarray,
        level: float
    ) -> float:
        """
        Compute coverage of prediction intervals.
        
        Coverage is the fraction of true values that fall within the
        specified prediction interval.
        
        Args:
            y_true: Ground truth values (shape: [n])
            samples: Prediction samples (shape: [n, n_samples] or [n_samples])
            level: Prediction interval level (e.g., 0.80 for 80% interval)
        
        Returns:
            Coverage fraction (between 0 and 1)
        """
        # Handle different sample shapes
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
            y_true = np.array([y_true]) if np.isscalar(y_true) else y_true[:1]
        
        # Compute prediction intervals
        alpha = (1 - level) / 2
        lower = np.percentile(samples, alpha * 100, axis=1)
        upper = np.percentile(samples, (1 - alpha) * 100, axis=1)
        
        # Check coverage
        covered = (y_true >= lower) & (y_true <= upper)
        
        return np.mean(covered)
    
    def _compute_ece(
        self,
        y_true: np.ndarray,
        samples: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE measures how well predicted probabilities match observed frequencies
        by binning predictions and comparing predicted vs. observed rates.
        
        Args:
            y_true: Ground truth values (shape: [n])
            samples: Prediction samples (shape: [n, n_samples] or [n_samples])
            n_bins: Number of bins for calibration curve
        
        Returns:
            Expected Calibration Error
        """
        # Handle different sample shapes
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
            y_true = np.array([y_true]) if np.isscalar(y_true) else y_true[:1]
        
        # Compute predicted quantiles for each observation
        quantiles = []
        for i in range(len(y_true)):
            y_i = y_true[i]
            samples_i = samples[i] if samples.ndim > 1 else samples
            
            # Compute empirical CDF at y_i
            quantile = np.mean(samples_i <= y_i)
            quantiles.append(quantile)
        
        quantiles = np.array(quantiles)
        
        # Bin quantiles
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(quantiles, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute ECE
        ece = 0.0
        for b in range(n_bins):
            mask = bin_indices == b
            if not np.any(mask):
                continue
            
            # Predicted probability (bin center)
            pred_prob = (bins[b] + bins[b + 1]) / 2
            
            # Observed frequency
            obs_freq = np.mean(quantiles[mask])
            
            # Weight by bin size
            weight = np.sum(mask) / len(quantiles)
            
            # Add to ECE
            ece += weight * np.abs(pred_prob - obs_freq)
        
        return ece
    
    def _compute_tail_recall(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        percentile: float = 95
    ) -> float:
        """
        Compute recall for tail events.
        
        Tail recall measures how well the model identifies extreme events
        (e.g., games where a player scores above the 95th percentile).
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            percentile: Percentile threshold for tail events
        
        Returns:
            Recall for tail events (between 0 and 1)
        """
        # Compute threshold
        threshold = np.percentile(y_true, percentile)
        
        # Identify tail events
        is_tail = y_true >= threshold
        
        if not np.any(is_tail):
            return np.nan
        
        # Predict tail events (use same threshold)
        pred_tail = y_pred >= threshold
        
        # Compute recall
        true_positives = np.sum(is_tail & pred_tail)
        recall = true_positives / np.sum(is_tail)
        
        return recall

    
    def compute_efficiency_metrics(
        self,
        model_fn: Callable,
        train_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        n_runs: int = 10
    ) -> Dict[str, float]:
        """
        Compute efficiency metrics for a model.
        
        Measures:
        - train_time_sec: Time to train the model (if train_data provided)
        - infer_time_ms_per_player: Average inference time per player
        - adaptation_time_ms: Time to adapt to scheme changes
        - memory_mb: Peak memory usage during inference
        
        Args:
            model_fn: Model function to benchmark
                Should accept (data, config) and return predictions
            train_data: Optional training data for measuring training time
            test_data: Test data for measuring inference time
            n_runs: Number of runs for averaging timing measurements
        
        Returns:
            Dictionary of efficiency metrics
        
        Raises:
            ValueError: If test_data is None
        """
        if test_data is None:
            raise ValueError("test_data is required for efficiency metrics")
        
        metrics = {}
        
        # Measure training time (if train_data provided)
        if train_data is not None:
            train_times = []
            for _ in range(min(n_runs, 3)):  # Limit training runs
                start_time = time.time()
                try:
                    model_fn(train_data, {'mode': 'train'})
                except Exception as e:
                    print(f"Training error: {e}")
                    break
                train_times.append(time.time() - start_time)
            
            if train_times:
                metrics['train_time_sec'] = np.mean(train_times)
            else:
                metrics['train_time_sec'] = np.nan
        else:
            metrics['train_time_sec'] = np.nan
        
        # Measure inference time
        infer_times = []
        for _ in range(n_runs):
            start_time = time.time()
            try:
                model_fn(test_data, {'mode': 'predict'})
            except Exception as e:
                print(f"Inference error: {e}")
                break
            infer_times.append(time.time() - start_time)
        
        if infer_times:
            # Convert to ms per player
            n_players = len(test_data) if hasattr(test_data, '__len__') else 1
            avg_time_sec = np.mean(infer_times)
            metrics['infer_time_ms_per_player'] = (avg_time_sec * 1000) / n_players
        else:
            metrics['infer_time_ms_per_player'] = np.nan
        
        # Measure adaptation time (scheme toggle)
        # Simulate a small change in input and measure response time
        adaptation_times = []
        for _ in range(n_runs):
            start_time = time.time()
            try:
                # Simulate scheme toggle by modifying test_data slightly
                model_fn(test_data, {'mode': 'predict', 'scheme_toggle': True})
            except Exception as e:
                # If scheme toggle not supported, use regular inference time
                adaptation_times.append(infer_times[0] if infer_times else 0)
                break
            adaptation_times.append(time.time() - start_time)
        
        if adaptation_times:
            metrics['adaptation_time_ms'] = np.mean(adaptation_times) * 1000
        else:
            metrics['adaptation_time_ms'] = np.nan
        
        # Measure memory usage
        tracemalloc.start()
        try:
            model_fn(test_data, {'mode': 'predict'})
            current, peak = tracemalloc.get_traced_memory()
            metrics['memory_mb'] = peak / (1024 * 1024)  # Convert to MB
        except Exception as e:
            print(f"Memory measurement error: {e}")
            metrics['memory_mb'] = np.nan
        finally:
            tracemalloc.stop()
        
        return metrics
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate side-by-side comparison DataFrame for all models.
        
        Creates a comprehensive comparison table with all metrics for all
        models and statistics. Useful for generating benchmark reports.
        
        Args:
            results: Dictionary with structure:
                {
                    model_name: {
                        'accuracy': {stat: {metric: value}},
                        'efficiency': {metric: value}
                    }
                }
        
        Returns:
            DataFrame with multi-index columns (model, metric) and
            index of statistics
        
        Raises:
            ValueError: If results dictionary is empty or malformed
        """
        if not results:
            raise ValueError("Results dictionary is empty")
        
        # Collect all statistics and metrics
        all_stats = set()
        all_metrics = set()
        
        for model_name, model_results in results.items():
            if 'accuracy' in model_results:
                for stat, metrics in model_results['accuracy'].items():
                    all_stats.add(stat)
                    all_metrics.update(metrics.keys())
        
        # Create comparison DataFrame
        comparison_data = []
        
        for stat in sorted(all_stats):
            row = {'stat': stat}
            
            for model_name in results.keys():
                if 'accuracy' not in results[model_name]:
                    continue
                
                if stat not in results[model_name]['accuracy']:
                    continue
                
                metrics = results[model_name]['accuracy'][stat]
                
                for metric_name, value in metrics.items():
                    col_name = f"{model_name}_{metric_name}"
                    row[col_name] = value
            
            comparison_data.append(row)
        
        df_accuracy = pd.DataFrame(comparison_data)
        
        # Add efficiency metrics as a separate section
        efficiency_data = []
        for model_name, model_results in results.items():
            if 'efficiency' not in model_results:
                continue
            
            row = {'model': model_name}
            row.update(model_results['efficiency'])
            efficiency_data.append(row)
        
        df_efficiency = pd.DataFrame(efficiency_data)
        
        # Store both DataFrames
        self.results['accuracy_comparison'] = df_accuracy
        self.results['efficiency_comparison'] = df_efficiency
        
        return df_accuracy
    
    def ablation_study(
        self,
        config_grid: List[Dict[str, Any]],
        model_fn: Callable,
        eval_data: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run ablation study over hyperparameter grid.
        
        Tests different configurations (e.g., blending weights, state amplitudes)
        to understand their impact on model performance.
        
        Args:
            config_grid: List of configuration dictionaries to test
                Each dict should contain hyperparameters to vary
            model_fn: Model function that accepts (data, config)
            eval_data: Evaluation data
            ground_truth: Ground truth values
        
        Returns:
            DataFrame with columns for each config parameter and metrics
        
        Raises:
            ValueError: If config_grid is empty
        """
        if not config_grid:
            raise ValueError("config_grid is empty")
        
        results = []
        
        for i, config in enumerate(config_grid):
            print(f"Running ablation {i+1}/{len(config_grid)}: {config}")
            
            try:
                # Run model with this configuration
                predictions = model_fn(eval_data, config)
                
                # Compute accuracy metrics
                accuracy = self.compute_accuracy_metrics(
                    y_true=ground_truth,
                    preds={'ablation': predictions}
                )
                
                # Compute efficiency metrics
                efficiency = self.compute_efficiency_metrics(
                    model_fn=lambda data, cfg: model_fn(data, config),
                    test_data=eval_data,
                    n_runs=3  # Fewer runs for ablation
                )
                
                # Compile results
                result_row = config.copy()
                
                # Add aggregated accuracy metrics (average across stats)
                if 'ablation' in accuracy:
                    for stat, metrics in accuracy['ablation'].items():
                        for metric_name, value in metrics.items():
                            col_name = f"{stat}_{metric_name}"
                            result_row[col_name] = value
                
                # Add efficiency metrics
                result_row.update(efficiency)
                
                results.append(result_row)
                
            except Exception as e:
                print(f"Error in ablation {i+1}: {e}")
                result_row = config.copy()
                result_row['error'] = str(e)
                results.append(result_row)
        
        df_ablation = pd.DataFrame(results)
        
        # Store results
        self.results['ablation_study'] = df_ablation
        
        return df_ablation
    
    def compute_overall_metrics(
        self,
        y_true: pd.DataFrame,
        preds: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute overall metrics across all statistics.
        
        Computes:
        - Spearman rank correlation: How well does the model rank players?
        - Decision gain: Improvement over baseline for decision-making
        
        Args:
            y_true: Ground truth DataFrame
            preds: Dictionary of predictions by model
        
        Returns:
            Dictionary mapping model names to overall metrics
        """
        overall_metrics = {}
        
        # Get target statistics
        stat_cols = [col for col in y_true.columns 
                     if col not in ['player_id', 'game_id', 'Date']]
        
        for model_name, pred_data in preds.items():
            if pred_data is None:
                continue
            
            metrics = {}
            
            # Compute Spearman correlation for each stat
            spearman_corrs = []
            
            for stat in stat_cols:
                if stat not in y_true.columns:
                    continue
                
                y_stat = y_true[stat].values
                
                # Extract predictions
                if isinstance(pred_data, pd.DataFrame):
                    if stat not in pred_data.columns:
                        continue
                    y_pred = pred_data[stat].values
                elif isinstance(pred_data, dict):
                    if 'samples' in pred_data and stat in pred_data['samples']:
                        pred_samples = pred_data['samples'][stat]
                        y_pred = np.mean(pred_samples, axis=1) if pred_samples.ndim > 1 else np.mean(pred_samples)
                    elif 'distributions' in pred_data and stat in pred_data['distributions']:
                        pred_samples = pred_data['distributions'][stat]
                        y_pred = np.mean(pred_samples, axis=1) if pred_samples.ndim > 1 else np.mean(pred_samples)
                    elif 'mean' in pred_data and stat in pred_data['mean']:
                        y_pred = pred_data['mean'][stat]
                    else:
                        continue
                else:
                    continue
                
                # Ensure same length
                min_len = min(len(y_stat), len(y_pred) if hasattr(y_pred, '__len__') else len(y_stat))
                y_stat = y_stat[:min_len]
                y_pred = y_pred[:min_len] if hasattr(y_pred, '__len__') else np.full(min_len, y_pred)
                
                # Compute Spearman correlation
                if len(y_stat) > 1:
                    corr, _ = stats.spearmanr(y_stat, y_pred)
                    spearman_corrs.append(corr)
            
            # Average Spearman correlation
            if spearman_corrs:
                metrics['spearman_rank_correlation'] = np.mean(spearman_corrs)
            else:
                metrics['spearman_rank_correlation'] = np.nan
            
            # Decision gain (simplified: improvement in top-k precision)
            # For now, use a placeholder
            metrics['decision_gain_sim'] = np.nan
            
            overall_metrics[model_name] = metrics
        
        return overall_metrics
    
    def generate_summary_report(
        self,
        accuracy_results: Dict[str, Dict[str, Dict[str, float]]],
        efficiency_results: Dict[str, Dict[str, float]],
        overall_results: Dict[str, Dict[str, float]],
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate a text summary report of benchmark results.
        
        Args:
            accuracy_results: Accuracy metrics by model and stat
            efficiency_results: Efficiency metrics by model
            overall_results: Overall metrics by model
            output_path: Optional path to save report
        
        Returns:
            Report text as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("BENCHMARK SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Accuracy summary
        lines.append("ACCURACY METRICS")
        lines.append("-" * 80)
        
        for model_name in accuracy_results.keys():
            lines.append(f"\n{model_name}:")
            
            if model_name not in accuracy_results:
                lines.append("  No accuracy results")
                continue
            
            for stat, metrics in accuracy_results[model_name].items():
                lines.append(f"  {stat}:")
                for metric_name, value in metrics.items():
                    lines.append(f"    {metric_name}: {value:.4f}")
        
        # Efficiency summary
        lines.append("\n" + "=" * 80)
        lines.append("EFFICIENCY METRICS")
        lines.append("-" * 80)
        
        for model_name, metrics in efficiency_results.items():
            lines.append(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                if not np.isnan(value):
                    lines.append(f"  {metric_name}: {value:.4f}")
        
        # Overall metrics
        lines.append("\n" + "=" * 80)
        lines.append("OVERALL METRICS")
        lines.append("-" * 80)
        
        for model_name, metrics in overall_results.items():
            lines.append(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                if not np.isnan(value):
                    lines.append(f"  {metric_name}: {value:.4f}")
        
        lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(lines)
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
        
        return report_text
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save benchmark results to disk.
        
        Args:
            results: Dictionary of results to save
            output_path: Path for saving (supports .csv, .json, .pkl)
        
        Raises:
            ValueError: If output format is not supported
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        suffix = output_path.suffix.lower()
        
        if suffix == '.csv':
            # Save as CSV (flatten if needed)
            if isinstance(results, pd.DataFrame):
                results.to_csv(output_path, index=False)
            else:
                # Try to convert to DataFrame
                try:
                    df = pd.DataFrame(results)
                    df.to_csv(output_path, index=False)
                except Exception as e:
                    raise ValueError(f"Cannot convert results to CSV: {e}")
        
        elif suffix == '.json':
            import json
            
            # Convert numpy types to Python types
            def convert_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            results_converted = convert_types(results)
            
            with open(output_path, 'w') as f:
                json.dump(results_converted, f, indent=2)
        
        elif suffix == '.pkl':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
        
        else:
            raise ValueError(f"Unsupported output format: {suffix}")
        
        print(f"Results saved to {output_path}")
    
    def run_full_benchmark(
        self,
        eval_windows: List[Tuple[str, pd.DataFrame]],
        models: Dict[str, Callable],
        ground_truth_dict: Dict[str, pd.DataFrame],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run complete benchmark across all windows and models.
        
        Convenience method that runs the full benchmarking pipeline:
        1. Evaluate models on each window
        2. Compute accuracy metrics
        3. Compute efficiency metrics
        4. Compute overall metrics
        5. Generate comparison tables
        6. Save results
        
        Args:
            eval_windows: List of (window_name, window_df) tuples
            models: Dictionary of model functions
            ground_truth_dict: Dictionary mapping window names to ground truth
            config: Configuration dictionary
        
        Returns:
            Dictionary with all benchmark results
        """
        all_results = {
            'accuracy': {},
            'efficiency': {},
            'overall': {},
            'windows': {}
        }
        
        # Run evaluation on each window
        window_iterator = eval_windows
        if self.enable_progress:
            window_iterator = tqdm(
                eval_windows,
                desc="Evaluating windows",
                unit="window"
            )
        
        for window_name, window_df in window_iterator:
            if not self.enable_progress:
                print(f"\n{'='*80}")
                print(f"Evaluating window: {window_name}")
                print(f"{'='*80}")
            
            # Get ground truth for this window
            ground_truth = ground_truth_dict.get(window_name, window_df)
            
            # Run models on this window
            window_results = self.run_eval_window(
                window_df=window_df,
                models=models,
                cfg=config
            )
            
            # Compute accuracy metrics
            accuracy = self.compute_accuracy_metrics(
                y_true=ground_truth,
                preds=window_results['predictions']
            )
            
            # Compute overall metrics
            overall = self.compute_overall_metrics(
                y_true=ground_truth,
                preds=window_results['predictions']
            )
            
            # Store window results
            all_results['windows'][window_name] = {
                'info': window_results['window_info'],
                'accuracy': accuracy,
                'overall': overall
            }
        
        # Compute efficiency metrics (once, not per window)
        if not self.enable_progress:
            print(f"\n{'='*80}")
            print("Computing efficiency metrics")
            print(f"{'='*80}")
        
        model_iterator = models.items()
        if self.enable_progress:
            model_iterator = tqdm(
                model_iterator,
                total=len(models),
                desc="Computing efficiency metrics",
                unit="model"
            )
        
        for model_name, model_fn in model_iterator:
            if not self.enable_progress:
                print(f"Benchmarking {model_name}...")
            
            # Use first window for efficiency testing
            test_window = eval_windows[0][1] if eval_windows else None
            
            if test_window is not None:
                try:
                    efficiency = self.compute_efficiency_metrics(
                        model_fn=model_fn,
                        test_data=test_window,
                        n_runs=5
                    )
                    all_results['efficiency'][model_name] = efficiency
                except Exception as e:
                    print(f"Error computing efficiency for {model_name}: {e}")
                    all_results['efficiency'][model_name] = {}
        
        # Aggregate results across windows
        if not self.enable_progress:
            print(f"\n{'='*80}")
            print("Aggregating results")
            print(f"{'='*80}")
        
        # Average accuracy metrics across windows
        for model_name in models.keys():
            all_results['accuracy'][model_name] = {}
            
            # Collect metrics from all windows
            stat_metrics = {}
            for window_name, window_results in all_results['windows'].items():
                if model_name not in window_results['accuracy']:
                    continue
                
                for stat, metrics in window_results['accuracy'][model_name].items():
                    if stat not in stat_metrics:
                        stat_metrics[stat] = {k: [] for k in metrics.keys()}
                    
                    for metric_name, value in metrics.items():
                        if not np.isnan(value):
                            stat_metrics[stat][metric_name].append(value)
            
            # Average across windows
            for stat, metrics_dict in stat_metrics.items():
                all_results['accuracy'][model_name][stat] = {
                    metric_name: np.mean(values) if values else np.nan
                    for metric_name, values in metrics_dict.items()
                }
        
        # Average overall metrics across windows
        for model_name in models.keys():
            overall_values = {}
            
            for window_name, window_results in all_results['windows'].items():
                if model_name not in window_results['overall']:
                    continue
                
                for metric_name, value in window_results['overall'][model_name].items():
                    if metric_name not in overall_values:
                        overall_values[metric_name] = []
                    if not np.isnan(value):
                        overall_values[metric_name].append(value)
            
            all_results['overall'][model_name] = {
                metric_name: np.mean(values) if values else np.nan
                for metric_name, values in overall_values.items()
            }
        
        # Generate comparison tables
        comparison_df = self.compare_models({
            model_name: {
                'accuracy': all_results['accuracy'].get(model_name, {}),
                'efficiency': all_results['efficiency'].get(model_name, {})
            }
            for model_name in models.keys()
        })
        
        all_results['comparison_table'] = comparison_df
        
        # Generate summary report
        report = self.generate_summary_report(
            accuracy_results=all_results['accuracy'],
            efficiency_results=all_results['efficiency'],
            overall_results=all_results['overall'],
            output_path=self.output_dir / "benchmark_summary.txt"
        )
        
        all_results['summary_report'] = report
        
        # Save results
        self.save_results(
            results=all_results,
            output_path=self.output_dir / "benchmark_results.json"
        )
        
        if 'comparison_table' in all_results:
            self.save_results(
                results=all_results['comparison_table'],
                output_path=self.output_dir / "comparison_table.csv"
            )
        
        print(f"\n{'='*80}")
        print(f"Benchmark complete! Results saved to {self.output_dir}")
        print(f"{'='*80}")
        
        return all_results
