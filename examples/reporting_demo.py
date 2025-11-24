"""
Demo script for the reporting module.

This script demonstrates how to use the ReportBuilder to generate:
- Coach one-pager reports
- Analyst detail reports
- Benchmark comparison reports
- JSON and CSV exports
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.reporting.build import ReportBuilder, CalibrationResult, create_benchmark_charts
from src.simulation.global_sim import SimulationResult, GameContext


def create_sample_data():
    """Create sample simulation results for demonstration."""
    n_trials = 1000
    
    # Create game context
    game_ctx = GameContext(
        game_id='2024_01_15_GSW_LAL',
        team_id='GSW',
        opponent_id='LAL',
        venue='home',
        pace=102.5
    )
    
    # Create sample player results
    players = []
    player_names = ['Stephen_Curry', 'Klay_Thompson', 'Draymond_Green', 'Andrew_Wiggins']
    base_pts = [28, 22, 8, 18]
    
    for i, (player_id, pts) in enumerate(zip(player_names, base_pts)):
        distributions = {
            'PTS': np.random.normal(pts, 5, n_trials),
            'TRB': np.random.normal(5 + i, 2, n_trials),
            'AST': np.random.normal(6 - i*0.5, 2, n_trials),
            'STL': np.random.normal(1.2, 0.5, n_trials),
            'BLK': np.random.normal(0.5, 0.3, n_trials),
            'TOV': np.random.normal(2.5, 1, n_trials),
            'FGA': np.random.normal(18 - i*2, 3, n_trials),
            '3PA': np.random.normal(8 - i, 2, n_trials),
            'FTA': np.random.normal(4, 2, n_trials),
            'PF': np.random.normal(2.5, 1, n_trials)
        }
        
        risk_metrics = {
            'var_95': np.percentile(distributions['PTS'], 5),
            'cvar_95': np.mean(distributions['PTS'][distributions['PTS'] <= np.percentile(distributions['PTS'], 5)])
        }
        
        result = SimulationResult(
            player_id=player_id,
            distributions=distributions,
            risk_metrics=risk_metrics,
            hypervolume_index=1.2 + i*0.1,
            metadata={'n_trials': n_trials, 'seed': 42}
        )
        
        players.append({'player_id': player_id, 'result': result})
    
    return game_ctx, players


def demo_coach_report():
    """Demonstrate coach one-pager generation."""
    print("=" * 60)
    print("DEMO: Coach One-Pager Report")
    print("=" * 60)
    
    game_ctx, players = create_sample_data()
    
    # Create report builder
    builder = ReportBuilder(output_dir='outputs/reports/demo')
    
    # Generate coach one-pager
    output_path = 'outputs/reports/demo/coach_one_pager.html'
    pdf_bytes = builder.build_coach_one_pager(
        game_ctx=game_ctx,
        players=players,
        output_path=output_path
    )
    
    print(f"✓ Coach one-pager generated: {output_path}")
    print(f"  Size: {len(pdf_bytes):,} bytes")
    print()


def demo_analyst_report():
    """Demonstrate analyst detail report generation."""
    print("=" * 60)
    print("DEMO: Analyst Detail Report")
    print("=" * 60)
    
    game_ctx, players = create_sample_data()
    
    # Create report builder
    builder = ReportBuilder(output_dir='outputs/reports/demo')
    
    # Generate analyst detail report
    output_path = 'outputs/reports/demo/analyst_detail.html'
    pdf_bytes = builder.build_analyst_detail(
        game_ctx=game_ctx,
        players=players,
        calibration=None,
        output_path=output_path
    )
    
    print(f"✓ Analyst detail report generated: {output_path}")
    print(f"  Size: {len(pdf_bytes):,} bytes")
    print()


def demo_benchmark_report():
    """Demonstrate benchmark report generation."""
    print("=" * 60)
    print("DEMO: Benchmark Comparison Report")
    print("=" * 60)
    
    # Create sample benchmark data
    tables = {
        'accuracy_metrics': pd.DataFrame({
            'model': ['global_only', 'blended', 'ridge', 'xgboost', 'mlp'] * 2,
            'stat': ['PTS'] * 5 + ['AST'] * 5,
            'mae': [4.8, 4.5, 5.2, 4.9, 5.1, 1.8, 1.6, 2.0, 1.7, 1.9],
            'rmse': [6.2, 5.9, 6.8, 6.4, 6.6, 2.3, 2.1, 2.5, 2.2, 2.4],
            'crps': [2.1, 1.9, 2.3, 2.0, 2.2, 0.9, 0.8, 1.0, 0.85, 0.95]
        }),
        'coverage_metrics': pd.DataFrame({
            'model': ['global_only', 'blended', 'ridge', 'xgboost', 'mlp'],
            'coverage_50': [0.52, 0.51, 0.48, 0.49, 0.50],
            'coverage_80': [0.81, 0.83, 0.79, 0.80, 0.78],
            'ece': [0.05, 0.04, 0.06, 0.055, 0.065]
        }),
        'efficiency_metrics': pd.DataFrame({
            'model': ['global_only', 'blended', 'ridge', 'xgboost', 'mlp'],
            'train_time_sec': [120, 180, 5, 45, 90],
            'infer_time_ms': [1800, 2300, 15, 18, 20],
            'memory_mb': [512, 768, 128, 256, 384]
        })
    }
    
    text = {
        'summary': 'The blended model (global + local) shows the best overall performance with MAE of 4.5 for PTS and 1.6 for AST, while maintaining excellent calibration (coverage_80 = 0.83).',
        'conclusions': 'Recommend using the blended approach for production. It provides the best accuracy-efficiency trade-off and maintains proper uncertainty quantification.'
    }
    
    # Create report builder
    builder = ReportBuilder(output_dir='outputs/reports/demo')
    
    # Generate PDF report
    output_path_pdf = 'outputs/reports/demo/benchmark_report.html'
    pdf_bytes = builder.build_benchmark_report(
        tables=tables,
        text=text,
        output_path=output_path_pdf,
        format='pdf'
    )
    
    print(f"✓ Benchmark PDF report generated: {output_path_pdf}")
    print(f"  Size: {len(pdf_bytes):,} bytes")
    
    # Generate Markdown report
    output_path_md = 'outputs/reports/demo/benchmark_report.md'
    md_bytes = builder.build_benchmark_report(
        tables=tables,
        text=text,
        output_path=output_path_md,
        format='markdown'
    )
    
    print(f"✓ Benchmark Markdown report generated: {output_path_md}")
    print(f"  Size: {len(md_bytes):,} bytes")
    
    # Generate charts
    results_df = tables['accuracy_metrics']
    charts = create_benchmark_charts(results_df, output_dir='outputs/reports/demo/charts')
    
    print(f"✓ Benchmark charts generated: {len(charts)} charts")
    for chart_name in charts.keys():
        print(f"  - {chart_name}")
    print()


def demo_json_csv_export():
    """Demonstrate JSON and CSV export."""
    print("=" * 60)
    print("DEMO: JSON and CSV Export")
    print("=" * 60)
    
    game_ctx, players = create_sample_data()
    
    # Create report builder
    builder = ReportBuilder(output_dir='outputs/reports/demo')
    
    # Create summary DataFrame
    summary_df = builder.create_players_summary_dataframe(players)
    
    # Export to CSV
    csv_path = 'outputs/reports/demo/player_projections.csv'
    builder.write_csv_summary(summary_df, csv_path)
    print(f"✓ CSV summary exported: {csv_path}")
    print(f"  Rows: {len(summary_df)}, Columns: {len(summary_df.columns)}")
    
    # Export to JSON
    json_path = 'outputs/reports/demo/game_projections.json'
    payload = {
        'players': [
            {
                'player_id': p['result'].player_id,
                'pts_mean': float(np.mean(p['result'].distributions['PTS'])),
                'pts_std': float(np.std(p['result'].distributions['PTS'])),
                'hypervolume_index': float(p['result'].hypervolume_index)
            }
            for p in players
        ]
    }
    builder.write_json_report(game_ctx, payload, json_path)
    print(f"✓ JSON report exported: {json_path}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("NBA PREDICTION SYSTEM - REPORTING MODULE DEMO")
    print("=" * 60 + "\n")
    
    # Run demos
    demo_coach_report()
    demo_analyst_report()
    demo_benchmark_report()
    demo_json_csv_export()
    
    print("=" * 60)
    print("All demos completed successfully!")
    print("Check outputs/reports/demo/ for generated files")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
