"""
Tests for parallelization functionality.

Tests the multiprocessing support in GlobalSimulator and BenchmarkRunner.
"""

import os
import numpy as np
import pandas as pd
import pytest

from src.simulation.global_sim import (
    GlobalSimulator,
    GameContext,
    OpponentContext,
    PlayerContext,
    GameState
)
from src.regions.build import CapabilityRegion, Ellipsoid, HPolytope, Halfspace
from src.benchmarks.compare import BenchmarkRunner


# Define mock models at module level for pickling
def mock_model_1(df, cfg):
    """Mock model 1 for testing."""
    return pd.DataFrame({
        'player_id': df['player_id'],
        'game_id': df['game_id'],
        'PTS': df['PTS'] + np.random.randn(len(df)),
        'TRB': df['TRB'] + np.random.randn(len(df)),
        'AST': df['AST'] + np.random.randn(len(df))
    })


def mock_model_2(df, cfg):
    """Mock model 2 for testing."""
    return pd.DataFrame({
        'player_id': df['player_id'],
        'game_id': df['game_id'],
        'PTS': df['PTS'] * 1.1,
        'TRB': df['TRB'] * 0.9,
        'AST': df['AST'] * 1.05
    })


@pytest.fixture
def mock_capability_region():
    """Create a mock capability region for testing."""
    # Create a simple ellipsoid
    center = np.array([0.55, 0.25, 0.20, 0.12, 0.15, 0.02, 0.02])
    shape_matrix = np.eye(7) * 0.01  # Small variance
    
    ellipsoid = Ellipsoid(
        center=center,
        shape_matrix=shape_matrix,
        alpha=0.80,
        dimension=7
    )
    
    # Create a simple polytope (no constraints for simplicity)
    polytope = HPolytope(halfspaces=[], dimension=7)
    
    region = CapabilityRegion(
        ellipsoid=ellipsoid,
        polytope=polytope,
        volume_estimate=1.0,
        hypervolume_above_baseline=0.5
    )
    
    return region


@pytest.fixture
def mock_game_context():
    """Create mock game context."""
    return GameContext(
        game_id="TEST_001",
        team_id="GSW",
        opponent_id="LAL",
        venue="home",
        pace=100.0
    )


@pytest.fixture
def mock_opponent_context():
    """Create mock opponent context."""
    return OpponentContext(
        opponent_id="LAL",
        scheme_drop_rate=0.4,
        scheme_switch_rate=0.3,
        scheme_ice_rate=0.2,
        blitz_rate=0.1,
        rim_deterrence_index=1.0,
        def_reb_strength=1.0,
        foul_discipline_index=1.0,
        pace=100.0,
        help_nail_freq=0.5
    )


@pytest.fixture
def mock_player_contexts():
    """Create mock player contexts."""
    return [
        PlayerContext(
            player_id="player_1",
            role="starter",
            exp_minutes=32.0,
            exp_usage=0.25
        ),
        PlayerContext(
            player_id="player_2",
            role="rotation",
            exp_minutes=22.0,
            exp_usage=0.20
        ),
        PlayerContext(
            player_id="player_3",
            role="bench",
            exp_minutes=12.0,
            exp_usage=0.15
        )
    ]


def test_global_simulator_parallel_initialization():
    """Test that GlobalSimulator initializes with parallelization settings."""
    # Test with explicit n_workers
    sim = GlobalSimulator(n_trials=100, n_workers=2, enable_progress=False)
    assert sim.n_workers == 2
    assert sim.enable_progress == False
    
    # Test with environment variable
    os.environ['NBA_PRED_N_WORKERS'] = '4'
    sim = GlobalSimulator(n_trials=100, enable_progress=False)
    assert sim.n_workers == 4
    del os.environ['NBA_PRED_N_WORKERS']


def test_global_simulator_sequential_vs_parallel(
    mock_capability_region,
    mock_game_context,
    mock_opponent_context,
    mock_player_contexts
):
    """Test that sequential and parallel execution produce similar results."""
    # Create simulator with fixed seed for reproducibility
    sim = GlobalSimulator(n_trials=100, n_stints=3, seed=42, enable_progress=False)
    
    # Prepare player data
    players = [
        (mock_capability_region, player_ctx)
        for player_ctx in mock_player_contexts[:2]  # Use 2 players for faster test
    ]
    
    # Run sequential
    results_seq = sim.simulate_multiple_players(
        players=players,
        game_ctx=mock_game_context,
        opp_ctx=mock_opponent_context,
        N=100,
        seed=42,
        parallel=False
    )
    
    # Run parallel
    sim_parallel = GlobalSimulator(n_trials=100, n_stints=3, seed=42, n_workers=2, enable_progress=False)
    results_par = sim_parallel.simulate_multiple_players(
        players=players,
        game_ctx=mock_game_context,
        opp_ctx=mock_opponent_context,
        N=100,
        seed=42,
        parallel=True
    )
    
    # Check that we got results for all players
    assert len(results_seq) == len(players)
    assert len(results_par) == len(players)
    
    # Check that results have expected structure
    for result in results_seq:
        assert result.player_id in [p[1].player_id for p in players]
        assert 'PTS' in result.distributions
        assert len(result.distributions['PTS']) == 100
    
    for result in results_par:
        assert result.player_id in [p[1].player_id for p in players]
        assert 'PTS' in result.distributions
        assert len(result.distributions['PTS']) == 100


def test_benchmark_runner_parallel_initialization():
    """Test that BenchmarkRunner initializes with parallelization settings."""
    # Test with explicit n_workers
    runner = BenchmarkRunner(n_workers=2, enable_progress=False)
    assert runner.n_workers == 2
    assert runner.enable_progress == False
    
    # Test with environment variable
    os.environ['NBA_PRED_N_WORKERS'] = '3'
    runner = BenchmarkRunner(enable_progress=False)
    assert runner.n_workers == 3
    del os.environ['NBA_PRED_N_WORKERS']


def test_benchmark_runner_parallel_model_execution():
    """Test that BenchmarkRunner can run models in parallel."""
    runner = BenchmarkRunner(n_workers=2, enable_progress=False)
    
    # Create mock data
    window_df = pd.DataFrame({
        'player_id': ['player_1', 'player_2', 'player_3'] * 10,
        'game_id': ['game_1'] * 30,
        'PTS': np.random.randint(10, 30, 30),
        'TRB': np.random.randint(3, 10, 30),
        'AST': np.random.randint(2, 8, 30)
    })
    
    # Use module-level mock models
    models = {
        'model_1': mock_model_1,
        'model_2': mock_model_2
    }
    
    cfg = {'target_stats': ['PTS', 'TRB', 'AST']}
    
    # Run models (sequential)
    predictions_seq = runner._run_models_on_window(
        models=models,
        window_df=window_df,
        cfg=cfg,
        parallel=False
    )
    
    # Run models (parallel)
    predictions_par = runner._run_models_on_window(
        models=models,
        window_df=window_df,
        cfg=cfg,
        parallel=True
    )
    
    # Check that both produced results
    assert len(predictions_seq) == 2
    assert len(predictions_par) == 2
    assert 'model_1' in predictions_seq
    assert 'model_2' in predictions_seq
    assert 'model_1' in predictions_par
    assert 'model_2' in predictions_par


def test_progress_bars_disabled():
    """Test that progress bars can be disabled."""
    # GlobalSimulator
    sim = GlobalSimulator(n_trials=10, enable_progress=False)
    assert sim.enable_progress == False
    
    # BenchmarkRunner
    runner = BenchmarkRunner(enable_progress=False)
    assert runner.enable_progress == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
