"""
Unit tests for reporting module.

Tests cover:
- ReportBuilder initialization
- Coach one-pager generation
- Analyst detail generation
- Benchmark report generation (PDF and Markdown)
- JSON and CSV export
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting.build import ReportBuilder, CalibrationResult, create_benchmark_charts, WEASYPRINT_AVAILABLE
from src.simulation.global_sim import SimulationResult, GameContext


class TestReportBuilder:
    """Test suite for ReportBuilder class."""
    
    @pytest.fixture
    def report_builder(self):
        """Create a ReportBuilder instance for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ReportBuilder(output_dir=tmpdir)
    
    @pytest.fixture
    def sample_game_context(self):
        """Create sample game context."""
        return GameContext(
            game_id='TEST_001',
            team_id='GSW',
            opponent_id='LAL',
            venue='home',
            pace=100.5
        )
    
    @pytest.fixture
    def sample_simulation_results(self):
        """Create sample simulation results."""
        n_trials = 1000
        
        results = []
        for i, player_id in enumerate(['curry_stephen', 'thompson_klay', 'green_draymond']):
            distributions = {
                'PTS': np.random.normal(25 - i*5, 5, n_trials),
                'TRB': np.random.normal(5 + i*2, 2, n_trials),
                'AST': np.random.normal(6 - i, 2, n_trials),
                'STL': np.random.normal(1.5, 0.5, n_trials),
                'BLK': np.random.normal(0.5 + i*0.5, 0.3, n_trials),
                'TOV': np.random.normal(2, 1, n_trials),
                'FGA': np.random.normal(18 - i*3, 3, n_trials),
                '3PA': np.random.normal(8 - i*2, 2, n_trials),
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
            
            results.append({'player_id': player_id, 'result': result})
        
        return results
    
    def test_initialization(self, report_builder):
        """Test ReportBuilder initialization."""
        assert report_builder is not None
        assert report_builder.output_dir.exists()
    
    def test_coach_one_pager_generation(self, report_builder, sample_game_context, sample_simulation_results):
        """Test coach one-pager PDF generation."""
        pdf_bytes = report_builder.build_coach_one_pager(
            game_ctx=sample_game_context,
            players=sample_simulation_results
        )
        
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        
        if WEASYPRINT_AVAILABLE:
            assert pdf_bytes[:4] == b'%PDF'  # PDF magic number
        else:
            # Should be HTML
            assert b'<!DOCTYPE html>' in pdf_bytes or b'<html>' in pdf_bytes
    
    def test_analyst_detail_generation(self, report_builder, sample_game_context, sample_simulation_results):
        """Test analyst detail PDF generation."""
        pdf_bytes = report_builder.build_analyst_detail(
            game_ctx=sample_game_context,
            players=sample_simulation_results,
            calibration=None
        )
        
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        
        if WEASYPRINT_AVAILABLE:
            assert pdf_bytes[:4] == b'%PDF'
        else:
            # Should be HTML
            assert b'<!DOCTYPE html>' in pdf_bytes or b'<html>' in pdf_bytes
    
    def test_benchmark_report_pdf(self, report_builder):
        """Test benchmark report PDF generation."""
        # Create sample benchmark tables
        tables = {
            'accuracy_metrics': pd.DataFrame({
                'model': ['global_only', 'blended', 'ridge'],
                'stat': ['PTS', 'PTS', 'PTS'],
                'mae': [4.8, 4.5, 5.2],
                'rmse': [6.2, 5.9, 6.8],
                'crps': [2.1, 1.9, 2.3]
            }),
            'coverage_metrics': pd.DataFrame({
                'model': ['global_only', 'blended', 'ridge'],
                'coverage_80': [0.81, 0.83, 0.79],
                'ece': [0.05, 0.04, 0.06]
            })
        }
        
        text = {
            'summary': 'The blended model shows the best overall performance.',
            'conclusions': 'Recommend using blended approach for production.'
        }
        
        pdf_bytes = report_builder.build_benchmark_report(
            tables=tables,
            text=text,
            format='pdf'
        )
        
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        
        if WEASYPRINT_AVAILABLE:
            assert pdf_bytes[:4] == b'%PDF'
        else:
            # Should be HTML
            assert b'<!DOCTYPE html>' in pdf_bytes or b'<html>' in pdf_bytes
    
    def test_benchmark_report_markdown(self, report_builder):
        """Test benchmark report Markdown generation."""
        tables = {
            'accuracy_metrics': pd.DataFrame({
                'model': ['global_only', 'blended'],
                'mae': [4.8, 4.5]
            })
        }
        
        md_bytes = report_builder.build_benchmark_report(
            tables=tables,
            format='markdown'
        )
        
        assert md_bytes is not None
        assert len(md_bytes) > 0
        
        md_str = md_bytes.decode('utf-8')
        assert '# Model Benchmark Comparison Report' in md_str
        assert 'global_only' in md_str
        assert 'blended' in md_str
    
    def test_json_report_export(self, report_builder, sample_game_context):
        """Test JSON report export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_report.json'
            
            payload = {
                'players': [
                    {'player_id': 'curry_stephen', 'pts_mean': 28.5, 'pts_std': 5.2}
                ]
            }
            
            report_builder.write_json_report(
                game_ctx=sample_game_context,
                payload=payload,
                output_path=output_path
            )
            
            assert output_path.exists()
            
            import json
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'data' in data
            assert data['metadata']['game_id'] == 'TEST_001'
    
    def test_csv_summary_export(self, report_builder, sample_simulation_results):
        """Test CSV summary export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_summary.csv'
            
            # Create summary DataFrame
            summary_df = report_builder.create_players_summary_dataframe(sample_simulation_results)
            
            report_builder.write_csv_summary(
                players_summary=summary_df,
                output_path=output_path
            )
            
            assert output_path.exists()
            
            # Read back and verify
            df = pd.read_csv(output_path)
            assert 'player_id' in df.columns
            assert 'stat' in df.columns
            assert 'mean' in df.columns
            assert len(df) > 0
    
    def test_create_players_summary_dataframe(self, report_builder, sample_simulation_results):
        """Test creating summary DataFrame from simulation results."""
        summary_df = report_builder.create_players_summary_dataframe(sample_simulation_results)
        
        assert isinstance(summary_df, pd.DataFrame)
        assert 'player_id' in summary_df.columns
        assert 'stat' in summary_df.columns
        assert 'mean' in summary_df.columns
        assert 'std' in summary_df.columns
        assert 'p10' in summary_df.columns
        assert 'p50' in summary_df.columns
        assert 'p90' in summary_df.columns
        
        # Check we have data for all players
        assert len(summary_df['player_id'].unique()) == 3
        
        # Check we have data for key stats
        assert 'PTS' in summary_df['stat'].values
        assert 'TRB' in summary_df['stat'].values
        assert 'AST' in summary_df['stat'].values


class TestBenchmarkCharts:
    """Test suite for benchmark chart creation."""
    
    def test_create_benchmark_charts(self):
        """Test benchmark chart creation."""
        results_df = pd.DataFrame({
            'model': ['global_only', 'blended', 'ridge'] * 2,
            'stat': ['PTS'] * 3 + ['AST'] * 3,
            'mae': [4.8, 4.5, 5.2, 1.8, 1.6, 2.0],
            'coverage_80': [0.81, 0.83, 0.79, 0.82, 0.84, 0.78]
        })
        
        charts = create_benchmark_charts(results_df)
        
        assert isinstance(charts, dict)
        assert 'mae_comparison' in charts
        assert 'coverage_comparison' in charts
        
        # Verify charts are PNG images
        assert charts['mae_comparison'][:8] == b'\x89PNG\r\n\x1a\n'
        assert charts['coverage_comparison'][:8] == b'\x89PNG\r\n\x1a\n'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
