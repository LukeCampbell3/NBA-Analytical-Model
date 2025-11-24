"""
Demonstration of parallelization features in the NBA prediction system.

This script shows how to use the parallelization capabilities of:
1. GlobalSimulator for parallel player simulation
2. BenchmarkRunner for parallel model evaluation

The parallelization can be configured via:
- Constructor arguments (n_workers, enable_progress)
- Environment variable (NBA_PRED_N_WORKERS)
- Configuration file (configs/default.yaml)
"""

import os
import time
import numpy as np
import pandas as pd

from src.simulation.global_sim import (
    GlobalSimulator,
    GameContext,
    OpponentContext,
    PlayerContext
)
from src.regions.build import (
    CapabilityRegion,
    Ellipsoid,
    HPolytope,
    RegionBuilder
)
from src.benchmarks.compare import BenchmarkRunner


def create_mock_region():
    """Create a mock capability region for demonstration."""
    center = np.array([0.55, 0.25, 0.20, 0.12, 0.15, 0.02, 0.02])
    shape_matrix = np.eye(7) * 0.01
    
    ellipsoid = Ellipsoid(
        center=center,
        shape_matrix=shape_matrix,
        alpha=0.80,
        dimension=7
    )
    
    polytope = HPolytope(halfspaces=[], dimension=7)
    
    return CapabilityRegion(
        ellipsoid=ellipsoid,
        polytope=polytope,
        volume_estimate=1.0,
        hypervolume_above_baseline=0.5
    )


def demo_global_simulator_parallelization():
    """Demonstrate parallel player simulation."""
    print("=" * 80)
    print("GLOBAL SIMULATOR PARALLELIZATION DEMO")
    print("=" * 80)
    
    # Create game context
    game_ctx = GameContext(
        game_id="DEMO_001",
        team_id="GSW",
        opponent_id="LAL",
        venue="home",
        pace=100.0
    )
    
    # Create opponent context
    opp_ctx = OpponentContext(
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
    
    # Create player contexts
    players = [
        (
            create_mock_region(),
            PlayerContext(
                player_id=f"player_{i}",
                role="starter" if i < 5 else "rotation",
                exp_minutes=32.0 if i < 5 else 22.0,
                exp_usage=0.25 if i < 5 else 0.20
            )
        )
        for i in range(10)
    ]
    
    print(f"\nSimulating {len(players)} players with 1000 trials each...")
    
    # Sequential execution
    print("\n1. Sequential Execution:")
    sim_seq = GlobalSimulator(
        n_trials=1000,
        n_workers=1,
        enable_progress=True
    )
    
    start_time = time.time()
    results_seq = sim_seq.simulate_multiple_players(
        players=players,
        game_ctx=game_ctx,
        opp_ctx=opp_ctx,
        parallel=False
    )
    seq_time = time.time() - start_time
    print(f"   Time: {seq_time:.2f} seconds")
    
    # Parallel execution
    print("\n2. Parallel Execution (4 workers):")
    sim_par = GlobalSimulator(
        n_trials=1000,
        n_workers=4,
        enable_progress=True
    )
    
    start_time = time.time()
    results_par = sim_par.simulate_multiple_players(
        players=players,
        game_ctx=game_ctx,
        opp_ctx=opp_ctx,
        parallel=True
    )
    par_time = time.time() - start_time
    print(f"   Time: {par_time:.2f} seconds")
    
    # Show speedup
    speedup = seq_time / par_time
    print(f"\n   Speedup: {speedup:.2f}x")
    
    # Show sample results
    print("\n3. Sample Results:")
    for i, result in enumerate(results_par[:3]):
        pts_mean = np.mean(result.distributions['PTS'])
        pts_std = np.std(result.distributions['PTS'])
        print(f"   {result.player_id}: PTS = {pts_mean:.1f} ± {pts_std:.1f}")


def demo_benchmark_runner_parallelization():
    """Demonstrate parallel model evaluation."""
    print("\n" + "=" * 80)
    print("BENCHMARK RUNNER PARALLELIZATION DEMO")
    print("=" * 80)
    
    # Create mock evaluation data
    n_games = 100
    window_df = pd.DataFrame({
        'player_id': [f'player_{i % 10}' for i in range(n_games)],
        'game_id': [f'game_{i}' for i in range(n_games)],
        'PTS': np.random.randint(10, 35, n_games),
        'TRB': np.random.randint(3, 12, n_games),
        'AST': np.random.randint(2, 10, n_games)
    })
    
    # Create mock models
    def model_baseline(df, cfg):
        """Simple baseline model."""
        time.sleep(0.1)  # Simulate computation
        return pd.DataFrame({
            'player_id': df['player_id'],
            'game_id': df['game_id'],
            'PTS': df['PTS'] * 0.95,
            'TRB': df['TRB'] * 0.98,
            'AST': df['AST'] * 0.97
        })
    
    def model_advanced(df, cfg):
        """Advanced model."""
        time.sleep(0.15)  # Simulate more computation
        return pd.DataFrame({
            'player_id': df['player_id'],
            'game_id': df['game_id'],
            'PTS': df['PTS'] * 1.02,
            'TRB': df['TRB'] * 1.01,
            'AST': df['AST'] * 1.03
        })
    
    def model_ensemble(df, cfg):
        """Ensemble model."""
        time.sleep(0.2)  # Simulate even more computation
        return pd.DataFrame({
            'player_id': df['player_id'],
            'game_id': df['game_id'],
            'PTS': df['PTS'] * 1.0,
            'TRB': df['TRB'] * 1.0,
            'AST': df['AST'] * 1.0
        })
    
    models = {
        'baseline': model_baseline,
        'advanced': model_advanced,
        'ensemble': model_ensemble
    }
    
    cfg = {'target_stats': ['PTS', 'TRB', 'AST']}
    
    print(f"\nEvaluating {len(models)} models on {n_games} games...")
    
    # Sequential execution
    print("\n1. Sequential Execution:")
    runner_seq = BenchmarkRunner(n_workers=1, enable_progress=True)
    
    start_time = time.time()
    predictions_seq = runner_seq._run_models_on_window(
        models=models,
        window_df=window_df,
        cfg=cfg,
        parallel=False
    )
    seq_time = time.time() - start_time
    print(f"   Time: {seq_time:.2f} seconds")
    
    # Parallel execution
    print("\n2. Parallel Execution (3 workers):")
    runner_par = BenchmarkRunner(n_workers=3, enable_progress=True)
    
    start_time = time.time()
    predictions_par = runner_par._run_models_on_window(
        models=models,
        window_df=window_df,
        cfg=cfg,
        parallel=True
    )
    par_time = time.time() - start_time
    print(f"   Time: {par_time:.2f} seconds")
    
    # Show speedup
    speedup = seq_time / par_time
    print(f"\n   Speedup: {speedup:.2f}x")
    
    # Show that results are consistent
    print("\n3. Results Consistency:")
    for model_name in models.keys():
        seq_pts = predictions_seq[model_name]['PTS'].mean()
        par_pts = predictions_par[model_name]['PTS'].mean()
        print(f"   {model_name}: Sequential={seq_pts:.2f}, Parallel={par_pts:.2f}")


def demo_configuration_options():
    """Demonstrate different ways to configure parallelization."""
    print("\n" + "=" * 80)
    print("CONFIGURATION OPTIONS DEMO")
    print("=" * 80)
    
    print("\n1. Via Constructor Arguments:")
    sim = GlobalSimulator(n_trials=100, n_workers=4, enable_progress=False)
    print(f"   GlobalSimulator: n_workers={sim.n_workers}, enable_progress={sim.enable_progress}")
    
    runner = BenchmarkRunner(n_workers=2, enable_progress=True)
    print(f"   BenchmarkRunner: n_workers={runner.n_workers}, enable_progress={runner.enable_progress}")
    
    print("\n2. Via Environment Variable:")
    os.environ['NBA_PRED_N_WORKERS'] = '8'
    sim = GlobalSimulator(n_trials=100, enable_progress=False)
    print(f"   GlobalSimulator: n_workers={sim.n_workers} (from env)")
    del os.environ['NBA_PRED_N_WORKERS']
    
    print("\n3. Via Configuration File:")
    print("   Set 'parallelization.n_workers' in configs/default.yaml")
    print("   Set 'parallelization.enable_progress_bars' in configs/default.yaml")
    
    print("\n4. Priority Order:")
    print("   1. Constructor arguments (highest priority)")
    print("   2. Environment variable (NBA_PRED_N_WORKERS)")
    print("   3. Configuration file")
    print("   4. Default (use all available CPU cores)")


if __name__ == '__main__':
    print("\nNBA Prediction System - Parallelization Demo")
    print("=" * 80)
    
    # Run demos
    demo_global_simulator_parallelization()
    demo_benchmark_runner_parallelization()
    demo_configuration_options()
    
    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
