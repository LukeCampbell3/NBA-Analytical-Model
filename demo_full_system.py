"""
Full System Demo: NBA Player Performance Prediction

Runs complete prediction pipeline on real players and generates all outputs.
Excludes positional tracking (data not available yet).
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import DataLoader
from src.features.transform import FeatureTransform
from src.regions.build import RegionBuilder
from src.simulation.global_sim import GlobalSimulator, GameContext, OpponentContext, PlayerContext
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configuration
DEMO_PLAYERS = [
    "Stephen_Curry",
    "LeBron_James", 
    "Nikola_Jokic",
    "Giannis_Antetokounmpo",
    "Luka_Doncic"
]
SEASON = 2024
OUTPUT_DIR = Path("results")
N_TRIALS = 10000  # Reduced for demo speed


def setup_output_dir():
    """Create output directory structure."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "posteriors").mkdir(exist_ok=True)
    (OUTPUT_DIR / "regions").mkdir(exist_ok=True)
    (OUTPUT_DIR / "simulations").mkdir(exist_ok=True)
    (OUTPUT_DIR / "summaries").mkdir(exist_ok=True)
    logger.info(f"Created output directory: {OUTPUT_DIR}")


def load_player_data(loader, player_name, season):
    """Load and validate player data."""
    logger.info(f"Loading data for {player_name} (season {season})")
    
    try:
        df = loader.load_player_data(player_name, season, use_processed=True)
        logger.info(f"Loaded {len(df)} games for {player_name}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {player_name}: {e}")
        return None


def compute_posterior(transform, df, player_name):
    """Compute player posterior with fallback."""
    logger.info(f"Computing posterior for {player_name}")
    
    try:
        posterior = transform.compute_player_posteriors_with_fallback(
            df=df,
            player_id=player_name,
            as_of_date=datetime.now()
        )
        
        # Save posterior
        posterior_data = {
            'player_id': posterior.player_id,
            'as_of_date': posterior.as_of_date.isoformat(),
            'mu': posterior.mu.tolist(),
            'Sigma': posterior.Sigma.tolist(),
            'feature_names': posterior.feature_names or []
        }
        
        output_file = OUTPUT_DIR / "posteriors" / f"{player_name}_posterior.json"
        with open(output_file, 'w') as f:
            json.dump(posterior_data, f, indent=2)
        
        logger.info(f"Saved posterior to {output_file}")
        return posterior
        
    except Exception as e:
        logger.error(f"Failed to compute posterior for {player_name}: {e}")
        return None


def build_region(builder, posterior, player_name):
    """Build capability region."""
    logger.info(f"Building capability region for {player_name}")
    
    try:
        ellipsoid = builder.credible_ellipsoid(
            mu=posterior.mu,
            Sigma=posterior.Sigma,
            alpha=0.80
        )
        
        # Save region info
        region_data = {
            'player_id': player_name,
            'center': ellipsoid.center.tolist(),
            'alpha': ellipsoid.alpha,
            'dimension': ellipsoid.dimension
        }
        
        output_file = OUTPUT_DIR / "regions" / f"{player_name}_region.json"
        with open(output_file, 'w') as f:
            json.dump(region_data, f, indent=2)
        
        logger.info(f"Saved region to {output_file}")
        return ellipsoid
        
    except Exception as e:
        logger.error(f"Failed to build region for {player_name}: {e}")
        return None


def run_simulation(simulator, ellipsoid, player_name):
    """Run global simulation."""
    logger.info(f"Running simulation for {player_name} ({N_TRIALS} trials)")
    
    try:
        # Create game context (example opponent: Lakers)
        game_ctx = GameContext(
            game_id=f"DEMO_{player_name}",
            team_id="DEMO_TEAM",
            opponent_id="LAL",
            venue="home",
            pace=100.0
        )
        
        # Create opponent context (league average)
        opp_ctx = OpponentContext(
            opponent_id="LAL",
            scheme_drop_rate=0.40,
            scheme_switch_rate=0.30,
            scheme_ice_rate=0.20,
            blitz_rate=0.15,
            rim_deterrence_index=1.0,
            def_reb_strength=1.0,
            foul_discipline_index=1.0,
            pace=100.0,
            help_nail_freq=0.35
        )
        
        # Create player context
        player_ctx = PlayerContext(
            player_id=player_name,
            role="starter",
            exp_minutes=34.0,
            exp_usage=0.28
        )
        
        # Run simulation
        from src.regions.build import CapabilityRegion
        region = CapabilityRegion(
            ellipsoid=ellipsoid,
            polytope=None,
            volume_estimate=0.0,
            hypervolume_above_baseline=0.0
        )
        
        result = simulator.simulate_player_game(
            region=region,
            player_ctx=player_ctx,
            game_ctx=game_ctx,
            opp_ctx=opp_ctx,
            N=N_TRIALS,
            seed=42
        )
        
        # Get summary statistics
        summary = simulator.get_summary_statistics(result)
        
        # Save simulation results
        sim_data = {
            'player_id': player_name,
            'n_trials': N_TRIALS,
            'distributions': summary,
            'risk_metrics': result.risk_metrics,
            'hypervolume_index': result.hypervolume_index,
            'metadata': result.metadata
        }
        
        output_file = OUTPUT_DIR / "simulations" / f"{player_name}_simulation.json"
        with open(output_file, 'w') as f:
            json.dump(sim_data, f, indent=2)
        
        logger.info(f"Saved simulation to {output_file}")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to run simulation for {player_name}: {e}")
        return None


def create_summary_report(results):
    """Create summary report of all results."""
    logger.info("Creating summary report")
    
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'season': SEASON,
        'n_trials': N_TRIALS,
        'players': []
    }
    
    for player_name, player_results in results.items():
        if player_results['simulation']:
            player_summary = {
                'name': player_name,
                'games_loaded': player_results['n_games'],
                'posterior_computed': player_results['posterior'] is not None,
                'region_built': player_results['region'] is not None,
                'simulation_run': player_results['simulation'] is not None,
                'predicted_stats': {}
            }
            
            # Extract key stats
            if player_results['simulation']:
                for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
                    if stat in player_results['simulation']:
                        player_summary['predicted_stats'][stat] = {
                            'mean': player_results['simulation'][stat]['mean'],
                            'p10': player_results['simulation'][stat]['p10'],
                            'p90': player_results['simulation'][stat]['p90']
                        }
            
            summary_data['players'].append(player_summary)
    
    # Save summary
    output_file = OUTPUT_DIR / "summaries" / "full_system_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"Saved summary report to {output_file}")
    
    # Create readable text summary
    text_summary = []
    text_summary.append("=" * 80)
    text_summary.append("NBA PLAYER PERFORMANCE PREDICTION - FULL SYSTEM DEMO")
    text_summary.append("=" * 80)
    text_summary.append(f"\nTimestamp: {summary_data['timestamp']}")
    text_summary.append(f"Season: {SEASON}")
    text_summary.append(f"Simulation Trials: {N_TRIALS:,}")
    text_summary.append(f"\nPlayers Analyzed: {len(summary_data['players'])}")
    text_summary.append("\n" + "-" * 80)
    
    for player in summary_data['players']:
        text_summary.append(f"\n{player['name']}")
        text_summary.append(f"  Games Loaded: {player['games_loaded']}")
        text_summary.append(f"  Pipeline Status:")
        text_summary.append(f"    ✓ Posterior Computed" if player['posterior_computed'] else "    ✗ Posterior Failed")
        text_summary.append(f"    ✓ Region Built" if player['region_built'] else "    ✗ Region Failed")
        text_summary.append(f"    ✓ Simulation Run" if player['simulation_run'] else "    ✗ Simulation Failed")
        
        if player['predicted_stats']:
            text_summary.append(f"  Predicted Stats (per game):")
            for stat, values in player['predicted_stats'].items():
                text_summary.append(
                    f"    {stat}: {values['mean']:.1f} "
                    f"(80% interval: [{values['p10']:.1f}, {values['p90']:.1f}])"
                )
    
    text_summary.append("\n" + "=" * 80)
    text_summary.append("Results saved to: " + str(OUTPUT_DIR.absolute()))
    text_summary.append("=" * 80)
    
    text_file = OUTPUT_DIR / "summaries" / "SUMMARY.txt"
    with open(text_file, 'w') as f:
        f.write('\n'.join(text_summary))
    
    # Print to console
    print('\n'.join(text_summary))


def main():
    """Run full system demo."""
    print("\n" + "=" * 80)
    print("NBA PLAYER PERFORMANCE PREDICTION - FULL SYSTEM DEMO")
    print("=" * 80 + "\n")
    
    # Setup
    setup_output_dir()
    
    # Initialize components (season auto-detected)
    print("Initializing system components...")
    loader = DataLoader(use_contracts=True)
    transform = FeatureTransform(use_cold_start=True)
    builder = RegionBuilder()
    simulator = GlobalSimulator(n_trials=N_TRIALS, seed=42)
    print("✓ System initialized\n")
    
    # Process each player
    results = {}
    
    for i, player_name in enumerate(DEMO_PLAYERS, 1):
        print(f"\n[{i}/{len(DEMO_PLAYERS)}] Processing {player_name}...")
        print("-" * 80)
        
        player_results = {
            'n_games': 0,
            'posterior': None,
            'region': None,
            'simulation': None
        }
        
        # 1. Load data
        df = load_player_data(loader, player_name, SEASON)
        if df is None:
            print(f"✗ Failed to load data for {player_name}")
            results[player_name] = player_results
            continue
        
        player_results['n_games'] = len(df)
        print(f"✓ Loaded {len(df)} games")
        
        # 2. Compute posterior
        posterior = compute_posterior(transform, df, player_name)
        if posterior is None:
            print(f"✗ Failed to compute posterior for {player_name}")
            results[player_name] = player_results
            continue
        
        player_results['posterior'] = posterior
        print(f"✓ Computed posterior (dimension: {len(posterior.mu)})")
        
        # 3. Build region
        region = build_region(builder, posterior, player_name)
        if region is None:
            print(f"✗ Failed to build region for {player_name}")
            results[player_name] = player_results
            continue
        
        player_results['region'] = region
        print(f"✓ Built capability region (α={region.alpha})")
        
        # 4. Run simulation
        simulation = run_simulation(simulator, region, player_name)
        if simulation is None:
            print(f"✗ Failed to run simulation for {player_name}")
            results[player_name] = player_results
            continue
        
        player_results['simulation'] = simulation
        print(f"✓ Completed simulation ({N_TRIALS:,} trials)")
        
        # Show quick preview
        if 'PTS' in simulation:
            pts = simulation['PTS']
            print(f"\n  Preview - Points per game:")
            print(f"    Mean: {pts['mean']:.1f}")
            print(f"    80% Interval: [{pts['p10']:.1f}, {pts['p90']:.1f}]")
        
        results[player_name] = player_results
    
    # Create summary report
    print("\n" + "=" * 80)
    print("Creating summary report...")
    create_summary_report(results)
    
    print("\n✓ Demo complete!")
    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print(f"  - {len(list((OUTPUT_DIR / 'posteriors').glob('*.json')))} posterior files")
    print(f"  - {len(list((OUTPUT_DIR / 'regions').glob('*.json')))} region files")
    print(f"  - {len(list((OUTPUT_DIR / 'simulations').glob('*.json')))} simulation files")
    print(f"  - Summary report: {OUTPUT_DIR / 'summaries' / 'SUMMARY.txt'}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
