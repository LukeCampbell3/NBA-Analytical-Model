"""
Simple System Demo: Shows what works with current data

Demonstrates:
1. Data loading with season auto-detection
2. Posterior computation with fallbacks
3. Region construction
4. Results export
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import DataLoader
from src.features.transform import FeatureTransform
from src.regions.build import RegionBuilder
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


def main():
    print("\n" + "=" * 80)
    print("NBA PREDICTION SYSTEM - SIMPLE DEMO")
    print("=" * 80 + "\n")
    
    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "data_samples").mkdir(exist_ok=True)
    (OUTPUT_DIR / "posteriors").mkdir(exist_ok=True)
    (OUTPUT_DIR / "regions").mkdir(exist_ok=True)
    
    # Initialize (season auto-detected!)
    print("Initializing system...")
    loader = DataLoader(use_contracts=False)  # Skip contracts for now
    transform = FeatureTransform(use_cold_start=True)
    builder = RegionBuilder()
    print("✓ System initialized\n")
    
    results_summary = []
    
    for i, player_name in enumerate(DEMO_PLAYERS, 1):
        print(f"[{i}/{len(DEMO_PLAYERS)}] {player_name}")
        print("-" * 80)
        
        try:
            # 1. Load data
            df = loader.load_player_data(player_name, SEASON, use_processed=True)
            print(f"✓ Loaded {len(df)} games")
            
            # Save sample
            sample = df.head(10)[['Player', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK']].to_dict('records')
            with open(OUTPUT_DIR / "data_samples" / f"{player_name}_sample.json", 'w') as f:
                json.dump(sample, f, indent=2)
            
            # 2. Compute posterior (will use fallback due to missing columns)
            posterior = transform.compute_player_posteriors_with_fallback(
                df=df,
                player_id=player_name
            )
            print(f"✓ Computed posterior (using cold-start prior)")
            
            # Save posterior
            posterior_data = {
                'player_id': posterior.player_id,
                'mu': posterior.mu.tolist(),
                'Sigma': posterior.Sigma.tolist(),
                'note': 'Using cold-start prior due to missing percentage columns in data'
            }
            with open(OUTPUT_DIR / "posteriors" / f"{player_name}.json", 'w') as f:
                json.dump(posterior_data, f, indent=2)
            
            # 3. Build region
            ellipsoid = builder.credible_ellipsoid(
                mu=posterior.mu,
                Sigma=posterior.Sigma,
                alpha=0.80
            )
            print(f"✓ Built capability region")
            
            # Save region
            region_data = {
                'player_id': player_name,
                'center': ellipsoid.center.tolist(),
                'alpha': ellipsoid.alpha,
                'dimension': ellipsoid.dimension
            }
            with open(OUTPUT_DIR / "regions" / f"{player_name}.json", 'w') as f:
                json.dump(region_data, f, indent=2)
            
            # Summary
            results_summary.append({
                'player': player_name,
                'games': len(df),
                'posterior_dim': len(posterior.mu),
                'region_alpha': ellipsoid.alpha,
                'status': 'success'
            })
            
            print(f"✓ Complete\n")
            
        except Exception as e:
            print(f"✗ Error: {e}\n")
            results_summary.append({
                'player': player_name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Create summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'season': SEASON,
        'players_processed': len([r for r in results_summary if r['status'] == 'success']),
        'total_players': len(DEMO_PLAYERS),
        'results': results_summary,
        'note': 'Full simulation requires percentage columns (AST%, TRB%, etc.) in data'
    }
    
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    for result in results_summary:
        if result['status'] == 'success':
            print(f"✓ {result['player']}: {result['games']} games, "
                  f"{result['posterior_dim']}D posterior, "
                  f"α={result['region_alpha']}")
        else:
            print(f"✗ {result['player']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")
    print(f"{'=' * 80}\n")
    
    # Show what was generated
    print("Generated files:")
    print(f"  - {len(list((OUTPUT_DIR / 'data_samples').glob('*.json')))} data samples")
    print(f"  - {len(list((OUTPUT_DIR / 'posteriors').glob('*.json')))} posteriors")
    print(f"  - {len(list((OUTPUT_DIR / 'regions').glob('*.json')))} regions")
    print(f"  - 1 summary report\n")
    
    print("Note: Full simulation requires percentage columns in data.")
    print("Current data has: PTS, TRB, AST, STL, BLK (counting stats)")
    print("Needed for simulation: AST%, TRB%, TOV%, STL%, BLK% (percentage stats)\n")


if __name__ == "__main__":
    main()
