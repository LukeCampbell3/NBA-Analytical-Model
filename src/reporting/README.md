# Reporting Module

The reporting module provides comprehensive report generation for NBA player performance predictions.

## Features

- **Coach One-Pager**: Single-page reports with key projections (PTS, REB, AST) and risk flags
- **Analyst Detail**: Multi-page reports with full distributions, calibration diagnostics, and visualizations
- **Benchmark Reports**: Model comparison reports in PDF and Markdown formats
- **Structured Exports**: JSON and CSV exports for programmatic access

## Installation Notes

### PDF Generation

The module uses WeasyPrint for PDF generation. If WeasyPrint is not available (e.g., missing system libraries on Windows), the module will automatically fall back to HTML output.

To enable PDF generation on Windows:
1. Install GTK+ for Windows: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
2. Ensure WeasyPrint can find the required libraries

On Linux/Mac, WeasyPrint typically works out of the box after `pip install weasyprint`.

## Usage

### Basic Example

```python
from src.reporting.build import ReportBuilder
from src.simulation.global_sim import SimulationResult, GameContext

# Create game context
game_ctx = GameContext(
    game_id='2024_01_15_GSW_LAL',
    team_id='GSW',
    opponent_id='LAL',
    venue='home',
    pace=102.5
)

# Prepare player results (list of dicts with 'result' key containing SimulationResult)
players = [
    {'player_id': 'curry_stephen', 'result': simulation_result_1},
    {'player_id': 'thompson_klay', 'result': simulation_result_2},
]

# Create report builder
builder = ReportBuilder(output_dir='outputs/reports')

# Generate coach one-pager
builder.build_coach_one_pager(
    game_ctx=game_ctx,
    players=players,
    output_path='outputs/reports/coach_report.html'
)
```

### Coach One-Pager

```python
# Generate single-page report for coaches
pdf_bytes = builder.build_coach_one_pager(
    game_ctx=game_ctx,
    players=players,
    output_path='coach_report.pdf'  # or .html if WeasyPrint unavailable
)
```

The coach one-pager includes:
- Game header (date, opponent, venue)
- Player grid with key projections (PTS, REB, AST)
- 80% confidence intervals
- Risk flags (high variance, foul risk)

### Analyst Detail Report

```python
# Generate detailed multi-page report
pdf_bytes = builder.build_analyst_detail(
    game_ctx=game_ctx,
    players=players,
    calibration=calibration_result,  # Optional
    output_path='analyst_report.pdf'
)
```

The analyst report includes:
- Full distribution plots (violin plots)
- Capability region visualizations (2D projections)
- Calibration diagnostics (PIT histograms)
- Hypervolume metrics and risk analysis

### Benchmark Reports

```python
import pandas as pd

# Prepare benchmark tables
tables = {
    'accuracy_metrics': pd.DataFrame({
        'model': ['global_only', 'blended', 'ridge'],
        'stat': ['PTS', 'PTS', 'PTS'],
        'mae': [4.8, 4.5, 5.2],
        'rmse': [6.2, 5.9, 6.8]
    }),
    'coverage_metrics': pd.DataFrame({
        'model': ['global_only', 'blended', 'ridge'],
        'coverage_80': [0.81, 0.83, 0.79]
    })
}

text = {
    'summary': 'The blended model shows the best performance.',
    'conclusions': 'Recommend blended approach for production.'
}

# Generate PDF report
builder.build_benchmark_report(
    tables=tables,
    text=text,
    output_path='benchmark_report.pdf',
    format='pdf'
)

# Generate Markdown report
builder.build_benchmark_report(
    tables=tables,
    text=text,
    output_path='benchmark_report.md',
    format='markdown'
)
```

### JSON and CSV Exports

```python
# Create summary DataFrame
summary_df = builder.create_players_summary_dataframe(players)

# Export to CSV
builder.write_csv_summary(
    players_summary=summary_df,
    output_path='projections.csv'
)

# Export to JSON
payload = {
    'players': [
        {'player_id': 'curry_stephen', 'pts_mean': 28.5}
    ]
}
builder.write_json_report(
    game_ctx=game_ctx,
    payload=payload,
    output_path='projections.json'
)
```

### Benchmark Charts

```python
from src.reporting.build import create_benchmark_charts

# Create visualization charts
results_df = pd.DataFrame({
    'model': ['global_only', 'blended'],
    'stat': ['PTS', 'PTS'],
    'mae': [4.8, 4.5],
    'coverage_80': [0.81, 0.83]
})

charts = create_benchmark_charts(
    results_df=results_df,
    output_dir='outputs/charts'
)

# charts is a dict mapping chart names to PNG bytes
# e.g., charts['mae_comparison'], charts['coverage_comparison']
```

## Demo

Run the demo script to see all features in action:

```bash
python examples/reporting_demo.py
```

This will generate sample reports in `outputs/reports/demo/`.

## Output Formats

### Coach One-Pager
- **Format**: PDF (or HTML if WeasyPrint unavailable)
- **Pages**: 1
- **Best for**: Quick game-day decisions

### Analyst Detail
- **Format**: PDF (or HTML if WeasyPrint unavailable)
- **Pages**: 4-5
- **Best for**: Deep analysis and diagnostics

### Benchmark Report
- **Formats**: PDF and Markdown
- **Pages**: Variable (depends on number of models/metrics)
- **Best for**: Model comparison and research

### Structured Exports
- **JSON**: Full structured data with metadata
- **CSV**: Tabular summary for spreadsheet analysis

## API Reference

### ReportBuilder

Main class for generating reports.

**Constructor:**
```python
ReportBuilder(
    output_dir: str = "outputs/reports",
    template_dir: Optional[str] = None
)
```

**Methods:**
- `build_coach_one_pager(game_ctx, players, output_path=None) -> bytes`
- `build_analyst_detail(game_ctx, players, calibration=None, output_path=None) -> bytes`
- `build_benchmark_report(tables, charts=None, text=None, output_path=None, format='pdf') -> bytes`
- `write_json_report(game_ctx, payload, output_path) -> None`
- `write_csv_summary(players_summary, output_path) -> None`
- `create_players_summary_dataframe(players) -> pd.DataFrame`

### CalibrationResult

Dataclass for calibration diagnostics.

**Attributes:**
- `pit_histograms: Dict[str, np.ndarray]`
- `reliability_diagrams: Dict[str, Dict[str, np.ndarray]]`
- `ece_scores: Dict[str, float]`
- `metadata: Dict[str, Any]`

### create_benchmark_charts

Function to create visualization charts.

```python
create_benchmark_charts(
    results_df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Dict[str, bytes]
```

Returns dictionary mapping chart names to PNG image bytes.

## Requirements

- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- jinja2 >= 3.1.0
- weasyprint >= 60.0 (optional, for PDF generation)

## Notes

- All reports use 80% confidence intervals by default
- Risk flags are triggered for:
  - High variance: CV > 1.5
  - Foul risk: Expected fouls > 4.5
- Charts use seaborn's whitegrid style
- HTML fallback is automatic when WeasyPrint is unavailable
