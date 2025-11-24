# NBA Prediction System CLI

Command-line interface for the NBA Player Performance Prediction System.

## Installation

The CLI is automatically available when you install the project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the CLI using Python module syntax:

```bash
python -m src.cli.main [COMMAND] [OPTIONS]
```

Or create an alias for convenience:

```bash
alias nba-cli="python -m src.cli.main"
```

## Available Commands

### Core Prediction Pipeline

#### `build-frontiers`
Fit efficiency frontiers for a season using quantile regression.

```bash
python -m src.cli.main build-frontiers --season 2024 --quantile 0.9
```

Options:
- `--season`: Season year (required)
- `--strata`: Stratification variable (default: role)
- `--quantile`: Quantile for frontier (default: 0.9)
- `--data-dir`: Data directory path (default: Data)
- `--output-dir`: Output directory (default: artifacts/frontiers)
- `--x-attr`: X-axis attribute (default: TS%)
- `--y-attr`: Y-axis attribute (default: USG%)

#### `regions`
Construct capability regions for specified game context and players.

```bash
python -m src.cli.main regions --game-id G001 --player Stephen_Curry --opponent-id LAL
```

Options:
- `--game-id`: Game ID (required)
- `--player`: Player name(s) - can be specified multiple times (required)
- `--opponent-id`: Opponent team ID (required)
- `--output`: Output directory (default: outputs/regions)
- `--alpha`: Credibility level (default: 0.80)

#### `simulate-global`
Run global Markov-Monte Carlo simulation for a game.

```bash
python -m src.cli.main simulate-global --game-id G001 --player Stephen_Curry --trials 20000 --output-json
```

Options:
- `--game-id`: Game ID (required)
- `--player`: Player name(s) - can be specified multiple times (required)
- `--opponent-id`: Opponent team ID (required)
- `--trials`: Number of simulation trials (default: 20000)
- `--seed`: Random seed for reproducibility
- `--output-json`: Save results as JSON
- `--output-pdf`: Generate PDF report
- `--output-dir`: Output directory (default: outputs/simulations)

### Local Models

#### `train-local`
Train local sub-problem models for event-level predictions.

```bash
python -m src.cli.main train-local --event-type all --cv-folds 5
```

Options:
- `--event-type`: Event type to train (rebound, assist, shot, all) (required)
- `--data-dir`: Data directory path (default: Data)
- `--output-dir`: Output directory (default: artifacts/local_models)
- `--cv-folds`: Number of cross-validation folds (default: 5)

#### `simulate-local`
Run local model inference for specified game and players.

```bash
python -m src.cli.main simulate-local --game-id G001 --player Stephen_Curry
```

Options:
- `--game-id`: Game ID (required)
- `--player`: Player name(s) - can be specified multiple times (required)
- `--model-dir`: Directory with trained models (default: artifacts/local_models)
- `--output-dir`: Output directory (default: outputs/local_predictions)

### Blending

#### `blend`
Combine global and local predictions with configurable strategy.

```bash
python -m src.cli.main blend --game-id G001 --global-weight 0.6 --local-weight 0.4
```

Options:
- `--game-id`: Game ID (required)
- `--global-weight`: Weight for global simulation (default: 0.6)
- `--local-weight`: Weight for local predictions (default: 0.4)
- `--strategy`: Blending strategy (weighted, bootstrap) (default: weighted)
- `--output-dir`: Output directory (default: outputs/blended)

### Baseline Models

#### `baselines-train`
Train traditional ML baseline models (Ridge, XGBoost, MLP).

```bash
python -m src.cli.main baselines-train --model-type all --season 2024
```

Options:
- `--model-type`: Model type to train (ridge, xgboost, mlp, all) (required)
- `--data-dir`: Data directory path (default: Data)
- `--output-dir`: Output directory (default: artifacts/baselines)
- `--season`: Season year to train on

#### `baselines-predict`
Generate predictions using trained baseline models.

```bash
python -m src.cli.main baselines-predict --data-file test_data.csv --output-file predictions.csv
```

Options:
- `--model-dir`: Directory with trained models (default: artifacts/baselines)
- `--data-file`: Input data file for predictions (required)
- `--output-file`: Output file (default: outputs/baseline_predictions.csv)

### Benchmarking and Evaluation

#### `benchmark`
Run comprehensive model comparison and benchmarking.

```bash
python -m src.cli.main benchmark --window rolling_30_games --models all --output-pdf --output-md
```

Options:
- `--window`: Evaluation window name (required)
- `--models`: Comma-separated list of models to compare (default: all)
- `--output-pdf`: Generate PDF report
- `--output-md`: Generate Markdown report
- `--output-dir`: Output directory (default: outputs/benchmarks)

#### `calibrate`
Calibrate model predictions using validation data.

```bash
python -m src.cli.main calibrate --model-type global --validation-data val_data.csv
```

Options:
- `--model-type`: Model type to calibrate (required)
- `--validation-data`: Validation data file (required)
- `--output-dir`: Output directory (default: artifacts/calibration)

#### `evaluate`
Evaluate model performance on test data.

```bash
python -m src.cli.main evaluate --model-type global --test-data test_data.csv --metrics all
```

Options:
- `--model-type`: Model type to evaluate (required)
- `--test-data`: Test data file (required)
- `--metrics`: Comma-separated list of metrics (default: all)
- `--output-file`: Output file (default: outputs/evaluation_results.json)

### Convenience Commands

#### `full-pipeline`
Run complete prediction pipeline for a game (all steps in sequence).

```bash
python -m src.cli.main full-pipeline --game-id G001 --player Stephen_Curry --opponent-id LAL
```

Options:
- `--game-id`: Game ID (required)
- `--player`: Player name(s) - can be specified multiple times (required)
- `--opponent-id`: Opponent team ID (required)
- `--output-dir`: Output directory (default: outputs/full_pipeline)

#### `version`
Display version information.

```bash
python -m src.cli.main version
```

## Example Workflows

### Complete Game Prediction

```bash
# 1. Build frontiers (one-time setup per season)
python -m src.cli.main build-frontiers --season 2024

# 2. Train local models (one-time setup)
python -m src.cli.main train-local --event-type all

# 3. Train baseline models (for comparison)
python -m src.cli.main baselines-train --model-type all --season 2024

# 4. Run full pipeline for a game
python -m src.cli.main full-pipeline \
    --game-id G001 \
    --player Stephen_Curry \
    --player Klay_Thompson \
    --opponent-id LAL
```

### Model Benchmarking

```bash
# 1. Run benchmark comparison
python -m src.cli.main benchmark \
    --window rolling_30_games \
    --models all \
    --output-pdf \
    --output-md

# 2. Calibrate best model
python -m src.cli.main calibrate \
    --model-type blended \
    --validation-data validation_set.csv

# 3. Evaluate on test set
python -m src.cli.main evaluate \
    --model-type blended \
    --test-data test_set.csv \
    --metrics all
```

### Custom Blending Experiment

```bash
# Run global simulation
python -m src.cli.main simulate-global \
    --game-id G001 \
    --player Stephen_Curry \
    --opponent-id LAL \
    --trials 20000 \
    --output-json

# Run local models
python -m src.cli.main simulate-local \
    --game-id G001 \
    --player Stephen_Curry

# Blend with custom weights
python -m src.cli.main blend \
    --game-id G001 \
    --global-weight 0.7 \
    --local-weight 0.3 \
    --strategy weighted
```

## Output Formats

The CLI supports multiple output formats:

- **JSON**: Structured data for programmatic access
- **CSV**: Tabular data for spreadsheet analysis
- **PDF**: Publication-ready reports with visualizations
- **Markdown**: Documentation-friendly reports

## Error Handling

The CLI provides clear error messages and exits with appropriate status codes:

- `0`: Success
- `1`: Error (with descriptive message)

All errors are printed to stderr for easy filtering.

## Configuration

The CLI reads configuration from `configs/default.yaml`. You can override settings using command-line options.

## Development

To add new commands:

1. Define a new function decorated with `@cli.command()`
2. Add appropriate Click options
3. Implement the command logic
4. Update this README

## Support

For issues or questions, please refer to the main project documentation.
