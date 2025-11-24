"""
Reporting module for NBA player performance predictions.

This module provides comprehensive reporting functionality including:
- Coach one-pager PDFs with key projections
- Analyst detail PDFs with full distributions and diagnostics
- Benchmark comparison reports (PDF and Markdown)
- JSON and CSV exports for structured data
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template

# Try to import weasyprint, but make it optional
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False
    import warnings
    warnings.warn(
        "WeasyPrint not available. PDF generation will be disabled. "
        "HTML reports will be generated instead.",
        ImportWarning
    )

from src.simulation.global_sim import SimulationResult, GameContext, PlayerContext
from src.calibration.fit import Calibrator


# Set style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150


@dataclass
class CalibrationResult:
    """Results from calibration diagnostics."""
    pit_histograms: Dict[str, np.ndarray]
    reliability_diagrams: Dict[str, Dict[str, np.ndarray]]
    ece_scores: Dict[str, float]
    metadata: Dict[str, Any]


class ReportBuilder:
    """
    Comprehensive report builder for NBA predictions.
    
    Generates multiple report types:
    - Coach one-pager: Single-page PDF with key projections
    - Analyst detail: Multi-page PDF with full distributions and diagnostics
    - Benchmark report: Model comparison with tables and charts
    - JSON/CSV exports: Structured data outputs
    """
    
    # Key statistics for coach reports
    COACH_KEY_STATS = ['PTS', 'TRB', 'AST']
    
    # All statistics for analyst reports
    ANALYST_ALL_STATS = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', '3PA', 'FTA', 'PF']
    
    # Risk thresholds
    HIGH_VARIANCE_THRESHOLD = 1.5  # Coefficient of variation
    FOUL_RISK_THRESHOLD = 4.5  # Expected fouls
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/reports",
        template_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize ReportBuilder.
        
        Args:
            output_dir: Directory for saving reports
            template_dir: Directory containing Jinja2 templates (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja2 environment if templates provided
        if template_dir:
            self.template_dir = Path(template_dir)
            self.jinja_env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        else:
            self.jinja_env = None
    
    def build_coach_one_pager(
        self,
        game_ctx: GameContext,
        players: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None
    ) -> bytes:
        """
        Generate a single-page PDF report for coaches.
        
        The one-pager includes:
        - Game header (date, opponent, venue)
        - Player grid with key projections (PTS, REB, AST)
        - 80% confidence intervals
        - Risk flags (high variance, foul risk, matchup disadvantage)
        - Model version and confidence badge
        
        Args:
            game_ctx: Game context information
            players: List of player results with SimulationResult data
            output_path: Optional path to save PDF (if None, returns bytes only)
        
        Returns:
            PDF as bytes (or HTML as bytes if WeasyPrint not available)
        """
        # Generate HTML content
        html_content = self._generate_coach_html(game_ctx, players)
        
        # Convert to PDF if WeasyPrint is available
        if WEASYPRINT_AVAILABLE:
            pdf_bytes = HTML(string=html_content).write_pdf()
        else:
            # Return HTML as bytes if PDF generation not available
            pdf_bytes = html_content.encode('utf-8')
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Adjust extension if needed
            if not WEASYPRINT_AVAILABLE and output_path.suffix == '.pdf':
                output_path = output_path.with_suffix('.html')
            
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
        
        return pdf_bytes
    
    def _generate_coach_html(
        self,
        game_ctx: GameContext,
        players: List[Dict[str, Any]]
    ) -> str:
        """Generate HTML for coach one-pager."""
        # Build player summaries
        player_summaries = []
        for player_data in players:
            summary = self._summarize_player_for_coach(player_data)
            player_summaries.append(summary)
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {{
                    size: letter;
                    margin: 0.5in;
                }}
                body {{
                    font-family: Arial, sans-serif;
                    font-size: 10pt;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 20px;
                    border-bottom: 2px solid #333;
                    padding-bottom: 10px;
                }}
                .header h1 {{
                    margin: 5px 0;
                    font-size: 18pt;
                }}
                .header .game-info {{
                    font-size: 11pt;
                    color: #666;
                }}
                .player-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .player-card {{
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 5px;
                    background: #f9f9f9;
                }}
                .player-name {{
                    font-weight: bold;
                    font-size: 11pt;
                    margin-bottom: 8px;
                    color: #333;
                }}
                .stat-row {{
                    display: flex;
                    justify-content: space-between;
                    margin: 4px 0;
                    font-size: 9pt;
                }}
                .stat-label {{
                    font-weight: bold;
                }}
                .stat-value {{
                    color: #0066cc;
                }}
                .risk-flags {{
                    margin-top: 8px;
                    padding-top: 8px;
                    border-top: 1px solid #ddd;
                    font-size: 8pt;
                }}
                .risk-flag {{
                    display: inline-block;
                    padding: 2px 6px;
                    margin: 2px;
                    border-radius: 3px;
                    background: #ff9800;
                    color: white;
                }}
                .footer {{
                    text-align: center;
                    font-size: 8pt;
                    color: #666;
                    margin-top: 20px;
                    padding-top: 10px;
                    border-top: 1px solid #ddd;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Game Day Projections</h1>
                <div class="game-info">
                    {game_ctx.team_id} vs {game_ctx.opponent_id} | {game_ctx.venue.upper()} | Pace: {game_ctx.pace:.1f}
                </div>
            </div>
            
            <div class="player-grid">
        """
        
        for summary in player_summaries:
            html += f"""
                <div class="player-card">
                    <div class="player-name">{summary['player_id'].replace('_', ' ').title()}</div>
            """
            
            for stat in self.COACH_KEY_STATS:
                if stat in summary['stats']:
                    stat_data = summary['stats'][stat]
                    html += f"""
                    <div class="stat-row">
                        <span class="stat-label">{stat}:</span>
                        <span class="stat-value">{stat_data['mean']:.1f} ({stat_data['ci_low']:.1f}-{stat_data['ci_high']:.1f})</span>
                    </div>
                    """
            
            if summary['risk_flags']:
                html += '<div class="risk-flags">'
                for flag in summary['risk_flags']:
                    html += f'<span class="risk-flag">{flag}</span>'
                html += '</div>'
            
            html += """
                </div>
            """
        
        html += f"""
            </div>
            
            <div class="footer">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 
                Model: Capability Region Simulation v1.0 | 
                Confidence: 80% Intervals
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _summarize_player_for_coach(self, player_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize player data for coach report."""
        result = player_data.get('result')
        if not result:
            return {'player_id': player_data.get('player_id', 'Unknown'), 'stats': {}, 'risk_flags': []}
        
        # Extract distributions
        distributions = result.distributions if hasattr(result, 'distributions') else result.get('distributions', {})
        player_id = result.player_id if hasattr(result, 'player_id') else result.get('player_id', 'Unknown')
        
        # Compute summary statistics
        stats = {}
        for stat in self.COACH_KEY_STATS:
            if stat in distributions:
                samples = distributions[stat]
                stats[stat] = {
                    'mean': np.mean(samples),
                    'ci_low': np.percentile(samples, 10),
                    'ci_high': np.percentile(samples, 90)
                }
        
        # Identify risk flags
        risk_flags = []
        
        # High variance check
        for stat in self.COACH_KEY_STATS:
            if stat in distributions:
                samples = distributions[stat]
                cv = np.std(samples) / (np.mean(samples) + 1e-6)
                if cv > self.HIGH_VARIANCE_THRESHOLD:
                    risk_flags.append(f"High {stat} Variance")
                    break  # Only flag once
        
        # Foul risk check
        if 'PF' in distributions:
            exp_fouls = np.mean(distributions['PF'])
            if exp_fouls > self.FOUL_RISK_THRESHOLD:
                risk_flags.append("Foul Risk")
        
        return {
            'player_id': player_id,
            'stats': stats,
            'risk_flags': risk_flags
        }

    def build_analyst_detail(
        self,
        game_ctx: GameContext,
        players: List[Dict[str, Any]],
        calibration: Optional[CalibrationResult] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> bytes:
        """
        Generate a detailed multi-page PDF report for analysts.
        
        The analyst report includes:
        - Page 1: Full distribution plots for all statistics (violin plots)
        - Page 2: Capability region visualizations (2D projections)
        - Page 3: Calibration diagnostics (PIT histograms, reliability diagrams)
        - Page 4: Hypervolume metrics and state transition probabilities
        - Page 5: Model comparison table (if multiple models run)
        
        Args:
            game_ctx: Game context information
            players: List of player results with SimulationResult data
            calibration: Optional calibration diagnostics
            output_path: Optional path to save PDF
        
        Returns:
            PDF as bytes (or HTML as bytes if WeasyPrint not available)
        """
        # Generate figures for each page
        figures = []
        
        # Page 1: Distribution plots
        fig1 = self._create_distribution_plots(players)
        figures.append(fig1)
        
        # Page 2: Capability region visualizations
        fig2 = self._create_region_visualizations(players)
        figures.append(fig2)
        
        # Page 3: Calibration diagnostics (if available)
        if calibration:
            fig3 = self._create_calibration_diagnostics(calibration)
            figures.append(fig3)
        
        # Page 4: Hypervolume and metadata
        fig4 = self._create_hypervolume_plots(players)
        figures.append(fig4)
        
        # Generate HTML with embedded figures
        html_content = self._generate_analyst_html(game_ctx, players, figures, calibration)
        
        # Convert to PDF if WeasyPrint is available
        if WEASYPRINT_AVAILABLE:
            pdf_bytes = HTML(string=html_content).write_pdf()
        else:
            # Return HTML as bytes if PDF generation not available
            pdf_bytes = html_content.encode('utf-8')
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Adjust extension if needed
            if not WEASYPRINT_AVAILABLE and output_path.suffix == '.pdf':
                output_path = output_path.with_suffix('.html')
            
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
        
        # Clean up figures
        for fig in figures:
            plt.close(fig)
        
        return pdf_bytes
    
    def _create_distribution_plots(self, players: List[Dict[str, Any]]) -> plt.Figure:
        """Create violin plots for all statistics."""
        n_stats = len(self.ANALYST_ALL_STATS)
        n_cols = 3
        n_rows = (n_stats + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, stat in enumerate(self.ANALYST_ALL_STATS):
            ax = axes[idx]
            
            # Collect data for this stat across all players
            data_for_violin = []
            labels = []
            
            for player_data in players:
                result = player_data.get('result')
                if result:
                    distributions = result.distributions if hasattr(result, 'distributions') else result.get('distributions', {})
                    if stat in distributions:
                        data_for_violin.append(distributions[stat])
                        player_id = result.player_id if hasattr(result, 'player_id') else result.get('player_id', 'Unknown')
                        labels.append(player_id.split('_')[0][:8])  # Shortened name
            
            if data_for_violin:
                parts = ax.violinplot(data_for_violin, positions=range(len(data_for_violin)), 
                                     showmeans=True, showmedians=True)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel(stat, fontsize=10)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {stat}', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Hide unused subplots
        for idx in range(len(self.ANALYST_ALL_STATS), len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Player Performance Distributions', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        return fig
    
    def _create_region_visualizations(self, players: List[Dict[str, Any]]) -> plt.Figure:
        """Create 2D projections of capability regions."""
        # For now, create placeholder visualizations
        # In a full implementation, this would show actual region boundaries
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Common 2D projections
        projections = [
            ('PTS', 'AST'),
            ('PTS', 'TRB'),
            ('AST', 'TRB'),
            ('PTS', 'TOV')
        ]
        
        for idx, (stat_x, stat_y) in enumerate(projections):
            ax = axes[idx]
            
            for player_data in players:
                result = player_data.get('result')
                if result:
                    distributions = result.distributions if hasattr(result, 'distributions') else result.get('distributions', {})
                    player_id = result.player_id if hasattr(result, 'player_id') else result.get('player_id', 'Unknown')
                    
                    if stat_x in distributions and stat_y in distributions:
                        x_samples = distributions[stat_x]
                        y_samples = distributions[stat_y]
                        
                        # Plot scatter with alpha
                        ax.scatter(x_samples, y_samples, alpha=0.1, s=1, label=player_id.split('_')[0][:8])
                        
                        # Plot mean
                        ax.scatter(np.mean(x_samples), np.mean(y_samples), 
                                 marker='x', s=100, linewidths=2)
            
            ax.set_xlabel(stat_x, fontsize=10)
            ax.set_ylabel(stat_y, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
        
        fig.suptitle('Capability Region Projections', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        return fig
    
    def _create_calibration_diagnostics(self, calibration: CalibrationResult) -> plt.Figure:
        """Create calibration diagnostic plots."""
        n_stats = len(calibration.pit_histograms)
        n_cols = 3
        n_rows = (n_stats + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, (stat, pit_values) in enumerate(calibration.pit_histograms.items()):
            ax = axes[idx]
            
            # Plot PIT histogram
            ax.hist(pit_values, bins=20, density=True, alpha=0.7, edgecolor='black')
            ax.axhline(y=1.0, color='r', linestyle='--', label='Uniform (ideal)')
            ax.set_xlabel('PIT Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'{stat} - ECE: {calibration.ece_scores.get(stat, 0):.3f}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_stats, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Calibration Diagnostics (PIT Histograms)', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        return fig
    
    def _create_hypervolume_plots(self, players: List[Dict[str, Any]]) -> plt.Figure:
        """Create hypervolume and risk metric visualizations."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract hypervolume indices
        player_names = []
        hypervolumes = []
        
        for player_data in players:
            result = player_data.get('result')
            if result:
                player_id = result.player_id if hasattr(result, 'player_id') else result.get('player_id', 'Unknown')
                hv = result.hypervolume_index if hasattr(result, 'hypervolume_index') else result.get('hypervolume_index', 0)
                
                player_names.append(player_id.split('_')[0][:10])
                hypervolumes.append(hv)
        
        # Plot 1: Hypervolume indices
        if hypervolumes:
            axes[0].barh(player_names, hypervolumes, color='steelblue')
            axes[0].set_xlabel('Hypervolume Index', fontsize=10)
            axes[0].set_title('Capability Region Size', fontsize=11, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Risk metrics summary
        risk_data = []
        for player_data in players:
            result = player_data.get('result')
            if result:
                risk_metrics = result.risk_metrics if hasattr(result, 'risk_metrics') else result.get('risk_metrics', {})
                player_id = result.player_id if hasattr(result, 'player_id') else result.get('player_id', 'Unknown')
                
                risk_data.append({
                    'Player': player_id.split('_')[0][:10],
                    'VaR_95': risk_metrics.get('var_95', 0),
                    'CVaR_95': risk_metrics.get('cvar_95', 0)
                })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            x = np.arange(len(risk_df))
            width = 0.35
            
            axes[1].bar(x - width/2, risk_df['VaR_95'], width, label='VaR 95%', color='orange')
            axes[1].bar(x + width/2, risk_df['CVaR_95'], width, label='CVaR 95%', color='red')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(risk_df['Player'], rotation=45, ha='right', fontsize=8)
            axes[1].set_ylabel('Risk Metric Value', fontsize=10)
            axes[1].set_title('Risk Metrics', fontsize=11, fontweight='bold')
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        return fig
    
    def _generate_analyst_html(
        self,
        game_ctx: GameContext,
        players: List[Dict[str, Any]],
        figures: List[plt.Figure],
        calibration: Optional[CalibrationResult]
    ) -> str:
        """Generate HTML for analyst detail report."""
        # Convert figures to base64 images
        import base64
        
        figure_images = []
        for fig in figures:
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            figure_images.append(img_base64)
            buf.close()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {{
                    size: letter;
                    margin: 0.5in;
                }}
                body {{
                    font-family: Arial, sans-serif;
                    font-size: 10pt;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 20px;
                    border-bottom: 2px solid #333;
                    padding-bottom: 10px;
                }}
                .page-break {{
                    page-break-after: always;
                }}
                .figure {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .figure img {{
                    max-width: 100%;
                    height: auto;
                }}
                h2 {{
                    color: #333;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Analyst Detail Report</h1>
                <p>{game_ctx.team_id} vs {game_ctx.opponent_id} | {game_ctx.venue.upper()}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
        """
        
        # Add each figure as a page
        page_titles = [
            'Performance Distributions',
            'Capability Region Projections',
            'Calibration Diagnostics',
            'Hypervolume and Risk Metrics'
        ]
        
        for idx, img_base64 in enumerate(figure_images):
            if idx < len(page_titles):
                html += f"""
                <div class="page-break">
                    <h2>{page_titles[idx]}</h2>
                    <div class="figure">
                        <img src="data:image/png;base64,{img_base64}" />
                    </div>
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        return html

    def build_benchmark_report(
        self,
        tables: Dict[str, pd.DataFrame],
        charts: Optional[Dict[str, bytes]] = None,
        text: Optional[Dict[str, str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        format: str = 'pdf'
    ) -> bytes:
        """
        Generate a benchmark comparison report.
        
        The benchmark report includes:
        - Executive summary with key findings
        - Accuracy metrics table (MAE, RMSE, CRPS) by model and stat
        - Coverage and calibration table (coverage_50, coverage_80, ECE)
        - Efficiency metrics table (runtime, memory, adaptation time)
        - Charts: Box plots, scatter plots of predicted vs actual
        - Statistical significance tests
        
        Args:
            tables: Dictionary of DataFrames with benchmark results
            charts: Optional dictionary of chart images (as bytes)
            text: Optional dictionary of text sections (e.g., 'summary', 'conclusions')
            output_path: Optional path to save report
            format: Output format ('pdf' or 'markdown')
        
        Returns:
            Report as bytes (PDF) or string (Markdown)
        """
        if format == 'markdown':
            return self._build_benchmark_markdown(tables, text, output_path)
        else:
            return self._build_benchmark_pdf(tables, charts, text, output_path)
    
    def _build_benchmark_pdf(
        self,
        tables: Dict[str, pd.DataFrame],
        charts: Optional[Dict[str, bytes]],
        text: Optional[Dict[str, str]],
        output_path: Optional[Union[str, Path]]
    ) -> bytes:
        """Build benchmark report as PDF."""
        html_content = self._generate_benchmark_html(tables, charts, text)
        
        # Convert to PDF if WeasyPrint is available
        if WEASYPRINT_AVAILABLE:
            pdf_bytes = HTML(string=html_content).write_pdf()
        else:
            # Return HTML as bytes if PDF generation not available
            pdf_bytes = html_content.encode('utf-8')
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Adjust extension if needed
            if not WEASYPRINT_AVAILABLE and output_path.suffix == '.pdf':
                output_path = output_path.with_suffix('.html')
            
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
        
        return pdf_bytes
    
    def _generate_benchmark_html(
        self,
        tables: Dict[str, pd.DataFrame],
        charts: Optional[Dict[str, bytes]],
        text: Optional[Dict[str, str]]
    ) -> str:
        """Generate HTML for benchmark report."""
        import base64
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {
                    size: letter landscape;
                    margin: 0.5in;
                }
                body {
                    font-family: Arial, sans-serif;
                    font-size: 9pt;
                }
                .header {
                    text-align: center;
                    margin-bottom: 20px;
                    border-bottom: 2px solid #333;
                    padding-bottom: 10px;
                }
                .summary {
                    background: #f0f0f0;
                    padding: 15px;
                    margin: 20px 0;
                    border-left: 4px solid #0066cc;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 8pt;
                }
                th {
                    background: #333;
                    color: white;
                    padding: 8px;
                    text-align: left;
                }
                td {
                    padding: 6px;
                    border-bottom: 1px solid #ddd;
                }
                tr:hover {
                    background: #f5f5f5;
                }
                .best-value {
                    font-weight: bold;
                    color: #00aa00;
                }
                .page-break {
                    page-break-after: always;
                }
                h2 {
                    color: #333;
                    border-bottom: 2px solid #0066cc;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }
                .chart {
                    text-align: center;
                    margin: 20px 0;
                }
                .chart img {
                    max-width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Benchmark Comparison Report</h1>
                <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M') + """</p>
            </div>
        """
        
        # Add executive summary if provided
        if text and 'summary' in text:
            html += f"""
            <div class="summary">
                <h3>Executive Summary</h3>
                <p>{text['summary']}</p>
            </div>
            """
        
        # Add tables
        for table_name, df in tables.items():
            html += f"""
            <h2>{table_name.replace('_', ' ').title()}</h2>
            {self._dataframe_to_html_table(df)}
            """
        
        # Add charts if provided
        if charts:
            html += '<div class="page-break"></div>'
            html += '<h2>Visualizations</h2>'
            
            for chart_name, chart_bytes in charts.items():
                img_base64 = base64.b64encode(chart_bytes).decode('utf-8')
                html += f"""
                <div class="chart">
                    <h3>{chart_name.replace('_', ' ').title()}</h3>
                    <img src="data:image/png;base64,{img_base64}" />
                </div>
                """
        
        # Add conclusions if provided
        if text and 'conclusions' in text:
            html += f"""
            <div class="page-break"></div>
            <h2>Conclusions</h2>
            <p>{text['conclusions']}</p>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _dataframe_to_html_table(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to HTML table with highlighting."""
        html = '<table>\n<thead>\n<tr>\n'
        
        # Header
        for col in df.columns:
            html += f'<th>{col}</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Rows
        for _, row in df.iterrows():
            html += '<tr>\n'
            for col in df.columns:
                value = row[col]
                
                # Format value
                if isinstance(value, float):
                    formatted = f'{value:.4f}'
                else:
                    formatted = str(value)
                
                html += f'<td>{formatted}</td>\n'
            html += '</tr>\n'
        
        html += '</tbody>\n</table>\n'
        return html
    
    def _build_benchmark_markdown(
        self,
        tables: Dict[str, pd.DataFrame],
        text: Optional[Dict[str, str]],
        output_path: Optional[Union[str, Path]]
    ) -> str:
        """Build benchmark report as Markdown."""
        md = f"# Model Benchmark Comparison Report\n\n"
        md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Add executive summary if provided
        if text and 'summary' in text:
            md += "## Executive Summary\n\n"
            md += f"{text['summary']}\n\n"
        
        # Add tables
        for table_name, df in tables.items():
            md += f"## {table_name.replace('_', ' ').title()}\n\n"
            
            # Try to use pandas to_markdown if tabulate is available
            try:
                md += df.to_markdown(index=False)
            except ImportError:
                # Fallback to simple markdown table
                md += self._dataframe_to_markdown_simple(df)
            
            md += "\n\n"
        
        # Add conclusions if provided
        if text and 'conclusions' in text:
            md += "## Conclusions\n\n"
            md += f"{text['conclusions']}\n\n"
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md)
        
        return md.encode('utf-8')
    
    def _dataframe_to_markdown_simple(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to simple markdown table without tabulate."""
        # Header
        md = "| " + " | ".join(str(col) for col in df.columns) + " |\n"
        md += "| " + " | ".join("---" for _ in df.columns) + " |\n"
        
        # Rows
        for _, row in df.iterrows():
            values = []
            for val in row:
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            md += "| " + " | ".join(values) + " |\n"
        
        return md
    
    def write_json_report(
        self,
        game_ctx: GameContext,
        payload: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Write structured JSON report.
        
        Args:
            game_ctx: Game context information
            payload: Dictionary containing report data
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build report structure
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'game_id': game_ctx.game_id,
                'team_id': game_ctx.team_id,
                'opponent_id': game_ctx.opponent_id,
                'venue': game_ctx.venue,
                'pace': game_ctx.pace
            },
            'data': payload
        }
        
        # Write JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
    
    def write_csv_summary(
        self,
        players_summary: pd.DataFrame,
        output_path: Union[str, Path]
    ) -> None:
        """
        Write CSV summary of player projections.
        
        Args:
            players_summary: DataFrame with player projection summaries
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV
        players_summary.to_csv(output_path, index=False)
    
    def create_players_summary_dataframe(
        self,
        players: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create a summary DataFrame from player results.
        
        Args:
            players: List of player results with SimulationResult data
        
        Returns:
            DataFrame with columns: player_id, stat, mean, std, p10, p50, p90
        """
        rows = []
        
        for player_data in players:
            result = player_data.get('result')
            if not result:
                continue
            
            distributions = result.distributions if hasattr(result, 'distributions') else result.get('distributions', {})
            player_id = result.player_id if hasattr(result, 'player_id') else result.get('player_id', 'Unknown')
            
            for stat, samples in distributions.items():
                rows.append({
                    'player_id': player_id,
                    'stat': stat,
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    'p10': np.percentile(samples, 10),
                    'p50': np.percentile(samples, 50),
                    'p90': np.percentile(samples, 90)
                })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


def create_benchmark_charts(
    results_df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, bytes]:
    """
    Create visualization charts for benchmark results.
    
    Args:
        results_df: DataFrame with benchmark results
        output_dir: Optional directory to save chart files
    
    Returns:
        Dictionary mapping chart names to image bytes
    """
    charts = {}
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chart 1: MAE comparison across models
    if 'model' in results_df.columns and 'mae' in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = results_df['model'].unique()
        stats = results_df['stat'].unique() if 'stat' in results_df.columns else ['Overall']
        
        x = np.arange(len(models))
        width = 0.8 / len(stats)
        
        for idx, stat in enumerate(stats):
            stat_data = results_df[results_df['stat'] == stat] if 'stat' in results_df.columns else results_df
            mae_values = [stat_data[stat_data['model'] == m]['mae'].values[0] if len(stat_data[stat_data['model'] == m]) > 0 else 0 for m in models]
            ax.bar(x + idx * width, mae_values, width, label=stat)
        
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title('Mean Absolute Error by Model', fontsize=13, fontweight='bold')
        tick_positions = x + width * (len(stats) - 1) / 2
        ax.set_xticks(tick_positions, models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        
        # Save to bytes
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        charts['mae_comparison'] = buf.read()
        buf.close()
        
        if output_dir:
            fig.savefig(output_dir / 'mae_comparison.png', bbox_inches='tight')
        
        plt.close(fig)
    
    # Chart 2: Coverage comparison
    if 'model' in results_df.columns and 'coverage_80' in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = results_df['model'].unique()
        coverage_values = [results_df[results_df['model'] == m]['coverage_80'].mean() for m in models]
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, coverage_values, color='steelblue')
        ax.axhline(y=0.80, color='r', linestyle='--', label='Target (80%)')
        ax.axhline(y=0.78, color='orange', linestyle=':', label='Lower bound')
        ax.axhline(y=0.84, color='orange', linestyle=':', label='Upper bound')
        
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Coverage (80% CI)', fontsize=11)
        ax.set_title('Prediction Interval Coverage', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos, models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.7, 0.9)
        fig.tight_layout()
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        charts['coverage_comparison'] = buf.read()
        buf.close()
        
        if output_dir:
            fig.savefig(output_dir / 'coverage_comparison.png', bbox_inches='tight')
        
        plt.close(fig)
    
    return charts
