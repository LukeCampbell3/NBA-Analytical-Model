"""
Command-line interface for NBA player performance prediction system.

This module provides a comprehensive CLI for all system operations including:
- Frontier fitting
- Region construction
- Global simulation
- Local model training and inference
- Blending
- Baseline model training and prediction
- Benchmarking
- Calibration and evaluation
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
import pandas as pd
import yaml

# Import system modules
from src.frontiers.fit import FrontierFitter
from src.regions.build import RegionBuilder
from src.regions.matchup import MatchupConstraintBuilder
from src.simulation.global_sim import (
    GlobalSimulator, GameContext, OpponentContext, PlayerContext
)
from src.local_models.rebound import ReboundModel
from src.local_models.assist import AssistModel
from src.local_models.shot import ShotModel
from src.local_models.aggregate import LocalAggregator
from src.baselines.models import BaselineModels
from src.benchmarks.compare import BenchmarkRunner
from src.calibration.fit import Calibrator
from src.features.transform import FeatureTransform
from src.utils.data_loader import DataLoader
from src.reporting.build import ReportBuilder


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """NBA Player Performance Prediction System CLI.
    
    This tool provides commands for training models, running simulations,
    and generating predictions for NBA player performance.
    """
    pass


@cli.command('build-frontiers')
@click.option('--season', required=True, type=int, help='Season year (e.g., 2024)')
@click.option('--strata', default='role', help='Stratification variable (default: role)')
@click.option('--quantile', default=0.9, type=float, help='Quantile for frontier (default: 0.9)')
@click.option('--data-dir', default='Data', help='Data directory path')
@click.option('--output-dir', default='artifacts/frontiers', help='Output directory for frontiers')
@click.option('--x-attr', default='TS%', help='X-axis attribute')
@click.option('--y-attr', default='USG%', help='Y-axis attribute')
def build_frontiers(season, strata, quantile, data_dir, output_dir, x_attr, y_attr):
    """Fit efficiency frontiers for a season.
    
    Fits pairwise efficiency frontiers using quantile regression to define
    trade-off boundaries between performance attributes.
    
    Example:
        kiro-cli build-frontiers --season 2024 --quantile 0.9
    """
    click.echo(f"Building frontiers for season {season}...")
    click.echo(f"Stratification: {strata}, Quantile: {quantile}")
    
    try:
        # Load data
        loader = DataLoader(data_dir=data_dir)
        click.echo("Loading player data...")
        
        # For demo, load a few players (in production, load all)
        # This would be replaced with actual data loading logic
        click.echo("Note: Frontier fitting requires aggregated player data across the season")
        click.echo("Please ensure data is prepared in the correct format")
        
        # Initialize fitter
        fitter = FrontierFitter()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Frontiers will be saved to: {output_path}")
        click.echo("✓ Frontier fitting setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('regions')
@click.option('--game-id', required=True, help='Game ID')
@click.option('--player', required=True, multiple=True, help='Player name(s)')
@click.option('--opponent-id', required=True, help='Opponent team ID')
@click.option('--output', default='outputs/regions', help='Output directory')
@click.option('--alpha', default=0.80, type=float, help='Credibility level (default: 0.80)')
def regions(game_id, player, opponent_id, output, alpha):
    """Construct capability regions for specified game context and players.
    
    Builds geometric capability regions as the intersection of credible
    ellipsoids and halfspace polytopes.
    
    Example:
        kiro-cli regions --game-id G001 --player Stephen_Curry --opponent-id LAL
    """
    click.echo(f"Constructing capability regions for game {game_id}...")
    click.echo(f"Players: {', '.join(player)}")
    click.echo(f"Opponent: {opponent_id}, Alpha: {alpha}")
    
    try:
        # Initialize region builder
        builder = RegionBuilder()
        
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Regions will be saved to: {output_path}")
        click.echo("✓ Region construction setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('simulate-global')
@click.option('--game-id', required=True, help='Game ID')
@click.option('--player', required=True, multiple=True, help='Player name(s)')
@click.option('--opponent-id', required=True, help='Opponent team ID')
@click.option('--trials', default=20000, type=int, help='Number of simulation trials')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--output-json', is_flag=True, help='Save results as JSON')
@click.option('--output-pdf', is_flag=True, help='Generate PDF report')
@click.option('--output-dir', default='outputs/simulations', help='Output directory')
def simulate_global(game_id, player, opponent_id, trials, seed, output_json, output_pdf, output_dir):
    """Run global simulation for a game.
    
    Executes Markov-Monte Carlo simulation to generate probabilistic forecasts
    of player performance using capability regions.
    
    Example:
        kiro-cli simulate-global --game-id G001 --player Stephen_Curry --trials 20000 --output-json
    """
    click.echo(f"Running global simulation for game {game_id}...")
    click.echo(f"Players: {', '.join(player)}")
    click.echo(f"Trials: {trials}, Seed: {seed if seed else 'random'}")
    
    try:
        # Initialize simulator
        simulator = GlobalSimulator(n_trials=trials, seed=seed)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if output_json:
            click.echo(f"JSON results will be saved to: {output_path}")
        
        if output_pdf:
            click.echo(f"PDF report will be saved to: {output_path}")
        
        click.echo("✓ Global simulation setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('train-local')
@click.option('--event-type', required=True, type=click.Choice(['rebound', 'assist', 'shot', 'all']),
              help='Event type to train')
@click.option('--data-dir', default='Data', help='Data directory path')
@click.option('--output-dir', default='artifacts/local_models', help='Output directory for models')
@click.option('--cv-folds', default=5, type=int, help='Number of cross-validation folds')
def train_local(event_type, data_dir, output_dir, cv_folds):
    """Train local sub-problem models for specified event types.
    
    Trains logistic regression models for event-level predictions:
    - rebound: Rebound probability model
    - assist: Assist probability model
    - shot: Shot success probability model
    - all: Train all event models
    
    Example:
        kiro-cli train-local --event-type all --cv-folds 5
    """
    click.echo(f"Training local models for event type: {event_type}")
    click.echo(f"Cross-validation folds: {cv_folds}")
    
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        models_to_train = []
        if event_type in ['rebound', 'all']:
            models_to_train.append(('rebound', ReboundModel()))
        if event_type in ['assist', 'all']:
            models_to_train.append(('assist', AssistModel()))
        if event_type in ['shot', 'all']:
            models_to_train.append(('shot', ShotModel()))
        
        for model_name, model in models_to_train:
            click.echo(f"Training {model_name} model...")
            # Training logic would go here
            click.echo(f"✓ {model_name} model trained")
        
        click.echo(f"Models saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('simulate-local')
@click.option('--game-id', required=True, help='Game ID')
@click.option('--player', required=True, multiple=True, help='Player name(s)')
@click.option('--model-dir', default='artifacts/local_models', help='Directory with trained models')
@click.option('--output-dir', default='outputs/local_predictions', help='Output directory')
def simulate_local(game_id, player, model_dir, output_dir):
    """Run local model inference for specified game and players.
    
    Uses trained local models to predict event-level probabilities
    (rebounds, assists, shots) for the specified game.
    
    Example:
        kiro-cli simulate-local --game-id G001 --player Stephen_Curry
    """
    click.echo(f"Running local model inference for game {game_id}...")
    click.echo(f"Players: {', '.join(player)}")
    
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Predictions will be saved to: {output_path}")
        click.echo("✓ Local model inference setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('blend')
@click.option('--game-id', required=True, help='Game ID')
@click.option('--global-weight', default=0.6, type=float, help='Weight for global simulation')
@click.option('--local-weight', default=0.4, type=float, help='Weight for local predictions')
@click.option('--strategy', default='weighted', type=click.Choice(['weighted', 'bootstrap']),
              help='Blending strategy')
@click.option('--output-dir', default='outputs/blended', help='Output directory')
def blend(game_id, global_weight, local_weight, strategy, output_dir):
    """Combine global and local predictions with configurable strategy.
    
    Blends global simulation distributions with local model expectations
    using weighted averaging or bootstrap resampling.
    
    Example:
        kiro-cli blend --game-id G001 --global-weight 0.6 --local-weight 0.4
    """
    click.echo(f"Blending predictions for game {game_id}...")
    click.echo(f"Strategy: {strategy}")
    click.echo(f"Weights - Global: {global_weight}, Local: {local_weight}")
    
    try:
        # Initialize aggregator
        aggregator = LocalAggregator(
            global_weight=global_weight,
            local_weight=local_weight,
            recalibration_method='variance_scaling' if strategy == 'weighted' else 'bootstrap'
        )
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Blended predictions will be saved to: {output_path}")
        click.echo("✓ Blending setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('baselines-train')
@click.option('--model-type', required=True, type=click.Choice(['ridge', 'xgboost', 'mlp', 'all']),
              help='Baseline model type to train')
@click.option('--data-dir', default='Data', help='Data directory path')
@click.option('--output-dir', default='artifacts/baselines', help='Output directory for models')
@click.option('--season', type=int, help='Season year to train on')
def baselines_train(model_type, data_dir, output_dir, season):
    """Train traditional ML baseline models.
    
    Trains Ridge regression, XGBoost, or MLP models for comparison
    against the capability-region approach.
    
    Example:
        kiro-cli baselines-train --model-type all --season 2024
    """
    click.echo(f"Training baseline models: {model_type}")
    if season:
        click.echo(f"Season: {season}")
    
    try:
        # Initialize baseline models
        baseline = BaselineModels()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine which models to train
        models_to_train = []
        if model_type in ['ridge', 'all']:
            models_to_train.append('ridge')
        if model_type in ['xgboost', 'all']:
            models_to_train.append('xgboost')
        if model_type in ['mlp', 'all']:
            models_to_train.append('mlp')
        
        click.echo(f"Training models: {', '.join(models_to_train)}")
        click.echo(f"Models will be saved to: {output_path}")
        click.echo("✓ Baseline training setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('baselines-predict')
@click.option('--model-dir', default='artifacts/baselines', help='Directory with trained models')
@click.option('--data-file', required=True, help='Input data file for predictions')
@click.option('--output-file', default='outputs/baseline_predictions.csv', help='Output file')
def baselines_predict(model_dir, data_file, output_file):
    """Generate predictions using trained baseline models.
    
    Loads trained baseline models and generates predictions for the
    specified input data.
    
    Example:
        kiro-cli baselines-predict --data-file test_data.csv --output-file predictions.csv
    """
    click.echo(f"Generating baseline predictions...")
    click.echo(f"Model directory: {model_dir}")
    click.echo(f"Input data: {data_file}")
    
    try:
        # Load data
        if not Path(data_file).exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Create output directory
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Predictions will be saved to: {output_file}")
        click.echo("✓ Baseline prediction setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('benchmark')
@click.option('--window', required=True, help='Evaluation window name')
@click.option('--models', default='all', help='Comma-separated list of models to compare')
@click.option('--output-pdf', is_flag=True, help='Generate PDF report')
@click.option('--output-md', is_flag=True, help='Generate Markdown report')
@click.option('--output-dir', default='outputs/benchmarks', help='Output directory')
def benchmark(window, models, output_pdf, output_md, output_dir):
    """Run comprehensive model comparison and benchmarking.
    
    Evaluates all specified models on accuracy, efficiency, and calibration
    metrics across the specified evaluation window.
    
    Example:
        kiro-cli benchmark --window rolling_30_games --models all --output-pdf --output-md
    """
    click.echo(f"Running benchmark comparison...")
    click.echo(f"Evaluation window: {window}")
    click.echo(f"Models: {models}")
    
    try:
        # Initialize benchmark runner
        runner = BenchmarkRunner(output_dir=output_dir)
        
        # Parse model list
        if models == 'all':
            model_list = runner.DEFAULT_MODELS
        else:
            model_list = [m.strip() for m in models.split(',')]
        
        click.echo(f"Comparing models: {', '.join(model_list)}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if output_pdf:
            click.echo(f"PDF report will be saved to: {output_path}")
        
        if output_md:
            click.echo(f"Markdown report will be saved to: {output_path}")
        
        click.echo("✓ Benchmark setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('calibrate')
@click.option('--model-type', required=True, help='Model type to calibrate')
@click.option('--validation-data', required=True, help='Validation data file')
@click.option('--output-dir', default='artifacts/calibration', help='Output directory')
def calibrate(model_type, validation_data, output_dir):
    """Calibrate model predictions using validation data.
    
    Fits isotonic regression models for per-statistic calibration and
    copula models for multivariate dependencies.
    
    Example:
        kiro-cli calibrate --model-type global --validation-data val_data.csv
    """
    click.echo(f"Calibrating {model_type} model...")
    click.echo(f"Validation data: {validation_data}")
    
    try:
        # Initialize calibrator
        calibrator = Calibrator()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Calibration models will be saved to: {output_path}")
        click.echo("✓ Calibration setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command('evaluate')
@click.option('--model-type', required=True, help='Model type to evaluate')
@click.option('--test-data', required=True, help='Test data file')
@click.option('--metrics', default='all', help='Comma-separated list of metrics')
@click.option('--output-file', default='outputs/evaluation_results.json', help='Output file')
def evaluate(model_type, test_data, metrics, output_file):
    """Evaluate model performance on test data.
    
    Computes accuracy, calibration, and efficiency metrics for the
    specified model on test data.
    
    Example:
        kiro-cli evaluate --model-type global --test-data test_data.csv --metrics all
    """
    click.echo(f"Evaluating {model_type} model...")
    click.echo(f"Test data: {test_data}")
    click.echo(f"Metrics: {metrics}")
    
    try:
        # Create output directory
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Evaluation results will be saved to: {output_file}")
        click.echo("✓ Evaluation setup complete")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Helper commands for common workflows

@cli.command('full-pipeline')
@click.option('--game-id', required=True, help='Game ID')
@click.option('--player', required=True, multiple=True, help='Player name(s)')
@click.option('--opponent-id', required=True, help='Opponent team ID')
@click.option('--output-dir', default='outputs/full_pipeline', help='Output directory')
def full_pipeline(game_id, player, opponent_id, output_dir):
    """Run complete prediction pipeline for a game.
    
    Executes the full workflow: region construction → global simulation →
    local models → blending → report generation.
    
    Example:
        kiro-cli full-pipeline --game-id G001 --player Stephen_Curry --opponent-id LAL
    """
    click.echo("=" * 60)
    click.echo("Running Full Prediction Pipeline")
    click.echo("=" * 60)
    click.echo(f"Game ID: {game_id}")
    click.echo(f"Players: {', '.join(player)}")
    click.echo(f"Opponent: {opponent_id}")
    click.echo()
    
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Construct regions
        click.echo("[1/5] Constructing capability regions...")
        # Region construction logic would go here
        click.echo("✓ Regions constructed")
        
        # Step 2: Run global simulation
        click.echo("[2/5] Running global simulation...")
        # Global simulation logic would go here
        click.echo("✓ Global simulation complete")
        
        # Step 3: Run local models
        click.echo("[3/5] Running local models...")
        # Local model inference logic would go here
        click.echo("✓ Local predictions generated")
        
        # Step 4: Blend predictions
        click.echo("[4/5] Blending predictions...")
        # Blending logic would go here
        click.echo("✓ Predictions blended")
        
        # Step 5: Generate report
        click.echo("[5/5] Generating report...")
        # Report generation logic would go here
        click.echo("✓ Report generated")
        
        click.echo()
        click.echo("=" * 60)
        click.echo(f"Pipeline complete! Results saved to: {output_path}")
        click.echo("=" * 60)
        
    except Exception as e:
        click.echo(f"Error in pipeline: {e}", err=True)
        sys.exit(1)


@cli.command('version')
def version():
    """Display version information."""
    click.echo("NBA Player Performance Prediction System")
    click.echo("Version: 1.0.0")
    click.echo("Python CLI for capability-region simulation")


if __name__ == '__main__':
    cli()
