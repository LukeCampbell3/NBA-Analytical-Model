# CLI Documentation

Comprehensive guide to the NBA Player Performance Prediction System command-line interface.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Commands](#commands)
  - [build-frontiers](#build-frontiers)
  - [regions](#regions)
  - [simulate-global](#simulate-global)
  - [train-local](#train-local)
  - [simulate-local](#simulate-local)
  - [blend](#blend)
  - [baselines-train](#baselines-train)
  - [baselines-predict](#baselines-predict)
  - [benchmark](#benchmark)
  - [calibrate](#calibrate)
  - [evaluate](#evaluate)
  - [full-pipeline](#full-pipeline)
  - [version](#version)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

Ensure the system is installed and dependencies are available:

```bash
pip install -r requirements.txt
```

Verify installation:

```bash
python -m src.cli.main version
```

## Getting Started

### Basic Usage

```bash
python -m src.cli.main [COMMAND] [OPTIONS]
```

### Getting Help

```bash
# General help
python -m src.cli.main --help

# Command-specific help
python -m src.cli.main simulate-global --help
```

### Common Options

Most commands support these common options:

- `--help`: Show help message
- `--output-dir PATH`: Specify output directory
- `--data-dir PATH`: Specify data directory

## Commands

### build-frontiers

Fit efficiency frontiers for a season using quantile regression.

**Usage**:
```bash
python -m src.cli.main build-frontiers [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--season` | int | Yes | - | Season year (e.g., 2024) |
| `--strata` | str | No | role | Stratification variable |
| `--quantile` | float | No | 0.9 | Quantile for frontier (0.0-1.0) |
| `--data-dir` | str | No | Data | Data directory path |
| `--output-dir` | str | No | artifacts/frontiers | Output directory |
| `--x-attr` | str | No | TS% | X-axis attribute |
| `--y-attr` | str | No | USG% | Y-axis attribute |

**Examples**:

```bash
# Build frontiers for 2024 season
python -m src.cli.main build-frontiers --season 2024

# Build frontiers with custom quantile
python -m src.cli.main build-frontiers --season 2024 --quantile 0.95

# Build frontiers for specific attribute pair
python -m src.cli.main build-frontiers \
  --season 2024 \
  --x-attr three_pa_rate \
  --y-attr rim_attempt_rate
```

**Output**:
- Frontier models saved to `artifacts/frontiers/`
- Log file with fitting details

---

### regions

Construct capability regions for specified game context and players.

**Usage**:
```bash
python -m src.cli.main regions [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--game-id` | str | Yes | - | Game ID |
| `--player` | str | Yes | - | Player name (can specify multiple) |
| `--opponent-id` | str | Yes | - | Opponent team ID |
| `--output` | str | No | outputs/regions | Output directory |
| `--alpha` | float | No | 0.80 | Credibility level (0.0-1.0) |

**Examples**:

```bash
# Construct region for single player
python -m src.cli.main regions \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL

# Construct regions for multiple players
python -m src.cli.main regions \
  --game-id G001 \
  --player Stephen_Curry \
  --player Klay_Thompson \
  --player Draymond_Green \
  --opponent-id LAL

# Use custom credibility level
python -m src.cli.main regions \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL \
  --alpha 0.90
```

**Output**:
- Region files saved to `outputs/regions/`
- Visualization plots (if enabled)

---

### simulate-global

Run global Markov-Monte Carlo simulation for player performance prediction.

**Usage**:
```bash
python -m src.cli.main simulate-global [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--game-id` | str | Yes | - | Game ID |
| `--player` | str | Yes | - | Player name (can specify multiple) |
| `--opponent-id` | str | Yes | - | Opponent team ID |
| `--trials` | int | No | 20000 | Number of simulation trials |
| `--seed` | int | No | random | Random seed for reproducibility |
| `--output-json` | flag | No | False | Save results as JSON |
| `--output-pdf` | flag | No | False | Generate PDF report |
| `--output-dir` | str | No | outputs/simulations | Output directory |

**Examples**:

```bash
# Basic simulation
python -m src.cli.main simulate-global \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL

# Simulation with JSON output
python -m src.cli.main simulate-global \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL \
  --output-json

# Simulation with PDF report
python -m src.cli.main simulate-global \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL \
  --output-pdf

# Reproducible simulation with seed
python -m src.cli.main simulate-global \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL \
  --trials 20000 \
  --seed 42 \
  --output-json

# Multiple players with all outputs
python -m src.cli.main simulate-global \
  --game-id G001 \
  --player Stephen_Curry \
  --player Klay_Thompson \
  --opponent-id LAL \
  --trials 20000 \
  --output-json \
  --output-pdf
```

**Output**:
- JSON file with distributions (if `--output-json`)
- PDF report (if `--output-pdf`)
- Console summary

---

### train-local

Train local sub-problem models for event-level predictions.

**Usage**:
```bash
python -m src.cli.main train-local [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--event-type` | choice | Yes | - | Event type: rebound, assist, shot, all |
| `--data-dir` | str | No | Data | Data directory path |
| `--output-dir` | str | No | artifacts/local_models | Output directory |
| `--cv-folds` | int | No | 5 | Number of cross-validation folds |

**Examples**:

```bash
# Train all local models
python -m src.cli.main train-local --event-type all

# Train specific model
python -m src.cli.main train-local --event-type rebound

# Train with custom CV folds
python -m src.cli.main train-local \
  --event-type all \
  --cv-folds 10

# Train with custom data directory
python -m src.cli.main train-local \
  --event-type all \
  --data-dir Data-Proc-OG
```

**Output**:
- Trained models saved to `artifacts/local_models/`
- Cross-validation results
- Feature importance plots

---

### simulate-local

Run local model inference for specified game and players.

**Usage**:
```bash
python -m src.cli.main simulate-local [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--game-id` | str | Yes | - | Game ID |
| `--player` | str | Yes | - | Player name (can specify multiple) |
| `--model-dir` | str | No | artifacts/local_models | Model directory |
| `--output-dir` | str | No | outputs/local_predictions | Output directory |

**Examples**:

```bash
# Run local model inference
python -m src.cli.main simulate-local \
  --game-id G001 \
  --player Stephen_Curry

# Multiple players
python -m src.cli.main simulate-local \
  --game-id G001 \
  --player Stephen_Curry \
  --player Klay_Thompson

# Custom model directory
python -m src.cli.main simulate-local \
  --game-id G001 \
  --player Stephen_Curry \
  --model-dir artifacts/local_models_v2
```

**Output**:
- JSON file with event probabilities
- Aggregated box-level expectations

---

### blend

Combine global and local predictions with configurable strategy.

**Usage**:
```bash
python -m src.cli.main blend [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--game-id` | str | Yes | - | Game ID |
| `--global-weight` | float | No | 0.6 | Weight for global simulation |
| `--local-weight` | float | No | 0.4 | Weight for local predictions |
| `--strategy` | choice | No | weighted | Blending strategy: weighted, bootstrap |
| `--output-dir` | str | No | outputs/blended | Output directory |

**Examples**:

```bash
# Default blending
python -m src.cli.main blend --game-id G001

# Custom weights
python -m src.cli.main blend \
  --game-id G001 \
  --global-weight 0.7 \
  --local-weight 0.3

# Bootstrap strategy
python -m src.cli.main blend \
  --game-id G001 \
  --strategy bootstrap

# Equal weighting
python -m src.cli.main blend \
  --game-id G001 \
  --global-weight 0.5 \
  --local-weight 0.5
```

**Output**:
- JSON file with blended distributions
- Comparison plots

---

### baselines-train

Train traditional ML baseline models for benchmarking.

**Usage**:
```bash
python -m src.cli.main baselines-train [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--model-type` | choice | Yes | - | Model type: ridge, xgboost, mlp, all |
| `--data-dir` | str | No | Data | Data directory path |
| `--output-dir` | str | No | artifacts/baselines | Output directory |
| `--season` | int | No | - | Season year to train on |

**Examples**:

```bash
# Train all baseline models
python -m src.cli.main baselines-train --model-type all

# Train specific model
python -m src.cli.main baselines-train --model-type xgboost

# Train for specific season
python -m src.cli.main baselines-train \
  --model-type all \
  --season 2024

# Train with custom output directory
python -m src.cli.main baselines-train \
  --model-type all \
  --output-dir artifacts/baselines_v2
```

**Output**:
- Trained models saved to `artifacts/baselines/`
- Training metrics and plots

---

### baselines-predict

Generate predictions using trained baseline models.

**Usage**:
```bash
python -m src.cli.main baselines-predict [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--model-dir` | str | No | artifacts/baselines | Model directory |
| `--data-file` | str | Yes | - | Input data file for predictions |
| `--output-file` | str | No | outputs/baseline_predictions.csv | Output file |

**Examples**:

```bash
# Generate predictions
python -m src.cli.main baselines-predict \
  --data-file test_data.csv

# Custom model directory
python -m src.cli.main baselines-predict \
  --model-dir artifacts/baselines_v2 \
  --data-file test_data.csv

# Custom output file
python -m src.cli.main baselines-predict \
  --data-file test_data.csv \
  --output-file predictions_2024.csv
```

**Output**:
- CSV file with predictions
- Prediction summary

---

### benchmark

Run comprehensive model comparison and benchmarking.

**Usage**:
```bash
python -m src.cli.main benchmark [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--window` | str | Yes | - | Evaluation window name |
| `--models` | str | No | all | Comma-separated list of models |
| `--output-pdf` | flag | No | False | Generate PDF report |
| `--output-md` | flag | No | False | Generate Markdown report |
| `--output-dir` | str | No | outputs/benchmarks | Output directory |

**Examples**:

```bash
# Benchmark all models
python -m src.cli.main benchmark \
  --window rolling_30_games \
  --models all

# Benchmark specific models
python -m src.cli.main benchmark \
  --window rolling_30_games \
  --models "original_global_only,blended_global_plus_local,baselines_xgboost"

# Generate PDF report
python -m src.cli.main benchmark \
  --window rolling_30_games \
  --models all \
  --output-pdf

# Generate both PDF and Markdown
python -m src.cli.main benchmark \
  --window rolling_30_games \
  --models all \
  --output-pdf \
  --output-md

# Benchmark playoffs only
python -m src.cli.main benchmark \
  --window playoffs_only \
  --models all \
  --output-pdf
```

**Output**:
- Benchmark results table
- PDF report (if `--output-pdf`)
- Markdown report (if `--output-md`)
- Comparison plots

---

### calibrate

Calibrate model predictions using validation data.

**Usage**:
```bash
python -m src.cli.main calibrate [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--model-type` | str | Yes | - | Model type to calibrate |
| `--validation-data` | str | Yes | - | Validation data file |
| `--output-dir` | str | No | artifacts/calibration | Output directory |

**Examples**:

```bash
# Calibrate global model
python -m src.cli.main calibrate \
  --model-type global \
  --validation-data val_data.csv

# Calibrate blended model
python -m src.cli.main calibrate \
  --model-type blended \
  --validation-data val_data.csv

# Custom output directory
python -m src.cli.main calibrate \
  --model-type global \
  --validation-data val_data.csv \
  --output-dir artifacts/calibration_v2
```

**Output**:
- Calibration models saved to `artifacts/calibration/`
- Calibration diagnostics plots

---

### evaluate

Evaluate model performance on test data.

**Usage**:
```bash
python -m src.cli.main evaluate [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--model-type` | str | Yes | - | Model type to evaluate |
| `--test-data` | str | Yes | - | Test data file |
| `--metrics` | str | No | all | Comma-separated list of metrics |
| `--output-file` | str | No | outputs/evaluation_results.json | Output file |

**Examples**:

```bash
# Evaluate with all metrics
python -m src.cli.main evaluate \
  --model-type global \
  --test-data test_data.csv

# Evaluate specific metrics
python -m src.cli.main evaluate \
  --model-type global \
  --test-data test_data.csv \
  --metrics "mae,rmse,crps"

# Custom output file
python -m src.cli.main evaluate \
  --model-type blended \
  --test-data test_data.csv \
  --output-file results_2024.json
```

**Output**:
- JSON file with evaluation metrics
- Metric summary

---

### full-pipeline

Run complete prediction pipeline for a game.

**Usage**:
```bash
python -m src.cli.main full-pipeline [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--game-id` | str | Yes | - | Game ID |
| `--player` | str | Yes | - | Player name (can specify multiple) |
| `--opponent-id` | str | Yes | - | Opponent team ID |
| `--output-dir` | str | No | outputs/full_pipeline | Output directory |

**Examples**:

```bash
# Run full pipeline for single player
python -m src.cli.main full-pipeline \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL

# Run full pipeline for multiple players
python -m src.cli.main full-pipeline \
  --game-id G001 \
  --player Stephen_Curry \
  --player Klay_Thompson \
  --player Draymond_Green \
  --opponent-id LAL

# Custom output directory
python -m src.cli.main full-pipeline \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL \
  --output-dir outputs/game_G001
```

**Pipeline Steps**:
1. Construct capability regions
2. Run global simulation
3. Run local models
4. Blend predictions
5. Generate report

**Output**:
- Complete prediction results
- PDF report
- JSON data files

---

### version

Display version information.

**Usage**:
```bash
python -m src.cli.main version
```

**Output**:
```
NBA Player Performance Prediction System
Version: 1.0.0
Python CLI for capability-region simulation
```

---

## Configuration

### Using Custom Configuration

Set environment variable to use custom configuration:

```bash
export NBA_CONFIG_PATH=configs/custom.yaml
python -m src.cli.main simulate-global ...
```

### Environment Variables

- `NBA_CONFIG_PATH`: Path to configuration file
- `NBA_DATA_DIR`: Override data directory
- `NBA_LOG_LEVEL`: Override log level (DEBUG, INFO, WARNING, ERROR)
- `NBA_N_WORKERS`: Override number of parallel workers
- `NBA_API_PORT`: Override API port

### Configuration File

Edit `configs/default.yaml` to customize default behavior. See `configs/README.md` for details.

## Examples

### Complete Workflow

```bash
# 1. Build frontiers for the season
python -m src.cli.main build-frontiers --season 2024

# 2. Train local models
python -m src.cli.main train-local --event-type all

# 3. Train baseline models
python -m src.cli.main baselines-train --model-type all --season 2024

# 4. Run full pipeline for a game
python -m src.cli.main full-pipeline \
  --game-id G001 \
  --player Stephen_Curry \
  --player Klay_Thompson \
  --opponent-id LAL

# 5. Run benchmarks
python -m src.cli.main benchmark \
  --window rolling_30_games \
  --models all \
  --output-pdf \
  --output-md
```

### Batch Processing

Process multiple games:

```bash
#!/bin/bash
# process_games.sh

GAMES=("G001" "G002" "G003")
PLAYERS=("Stephen_Curry" "Klay_Thompson")

for game in "${GAMES[@]}"; do
  echo "Processing game $game..."
  python -m src.cli.main full-pipeline \
    --game-id "$game" \
    --player "${PLAYERS[@]/#/--player }" \
    --opponent-id LAL \
    --output-dir "outputs/game_$game"
done
```

### Parallel Execution

Use GNU parallel for faster processing:

```bash
# Create list of games
echo "G001 G002 G003" | tr ' ' '\n' > games.txt

# Process in parallel
parallel -j 4 python -m src.cli.main simulate-global \
  --game-id {} \
  --player Stephen_Curry \
  --opponent-id LAL \
  --output-json :::: games.txt
```

## Troubleshooting

### Common Issues

**Command not found**:
```bash
# Ensure you're using the module syntax
python -m src.cli.main [COMMAND]
# Not: python src/cli/main.py [COMMAND]
```

**Import errors**:
```bash
# Ensure you're in the project root directory
cd /path/to/NBA_Analysis
python -m src.cli.main version
```

**Missing data**:
```bash
# Check data directory exists
ls Data/

# Specify custom data directory
python -m src.cli.main simulate-global \
  --data-dir /path/to/data \
  ...
```

**Permission errors**:
```bash
# Ensure output directories are writable
chmod -R u+w outputs/
```

### Debug Mode

Enable debug logging:

```bash
export NBA_LOG_LEVEL=DEBUG
python -m src.cli.main simulate-global ...
```

### Getting Help

```bash
# General help
python -m src.cli.main --help

# Command help
python -m src.cli.main simulate-global --help

# Check version
python -m src.cli.main version
```

## Best Practices

1. **Use Seeds**: Always use `--seed` for reproducible results
2. **Save Outputs**: Use `--output-json` and `--output-pdf` to save results
3. **Batch Processing**: Process multiple games in parallel when possible
4. **Monitor Logs**: Check `logs/system.log` for detailed execution logs
5. **Validate Data**: Ensure data files are in correct format before processing
6. **Use Full Pipeline**: Use `full-pipeline` command for complete workflow
7. **Benchmark Regularly**: Run benchmarks after configuration changes

## Support

For CLI questions or issues:
1. Check this documentation
2. Run command with `--help` flag
3. Check logs in `logs/system.log`
4. Review configuration in `configs/default.yaml`
5. Contact the development team
