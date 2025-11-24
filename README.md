<<<<<<< HEAD
# NBA-Analytical-Model
=======
# NBA Player Performance Prediction System

A sophisticated framework for NBA player performance prediction using capability-region simulation combined with local sub-problem models.

## Overview

This system predicts per-game statistics (points, rebounds, assists, etc.) by modeling player capabilities as geometric regions constrained by opponent schemes, role boundaries, and historical performance. It includes rigorous benchmarking against traditional ML baselines and produces publication-ready reports for coaches and analysts.

### Key Capabilities

- **Geometric Capability Regions**: Models player performance as the intersection of credible ellipsoids and halfspace polytopes
- **Markov-Monte Carlo Simulation**: Simulates game states (Normal, Hot, Cold, FoulRisk, WindDown) with state transitions
- **Local Event Models**: Event-specific logistic regression for rebounds, assists, and shots
- **Blended Predictions**: Combines global and local models with configurable weights
- **Comprehensive Benchmarking**: Compares against Ridge, XGBoost, and MLP baselines
- **Probabilistic Forecasts**: Generates full distributions with uncertainty quantification
- **Publication-Ready Reports**: PDF reports for coaches and analysts with visualizations

## Project Structure

```
NBA_Analysis/
├── src/                    # Source code
│   ├── utils/             # Data loading, logging, error handling
│   ├── features/          # Feature engineering and transformation
│   ├── frontiers/         # Efficiency frontier fitting
│   ├── regions/           # Capability region construction
│   ├── simulation/        # Global Markov-MC simulator
│   ├── local_models/      # Event-specific models (rebound, assist, shot)
│   ├── baselines/         # Traditional ML models
│   ├── calibration/       # Probability calibration
│   ├── benchmarks/        # Model comparison and evaluation
│   ├── reporting/         # Report generation (PDF, JSON, CSV)
│   ├── api/               # FastAPI REST endpoints
│   ├── cli/               # Command-line interface
│   └── positional/        # Positional tracking (scaffolded, disabled)
├── configs/               # Configuration files
│   └── default.yaml       # System configuration
├── tests/                 # Test suite
├── artifacts/             # Model artifacts and cached data
├── outputs/               # Generated reports and predictions
├── logs/                  # System logs
├── Data/                  # Player statistics (CSV files)
├── Data-Proc-OG/          # Processed data
└── requirements.txt       # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM recommended
- Optional: CUDA-capable GPU for faster XGBoost training

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd NBA_Analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -m src.cli.main version
```

### Data Setup

Place player statistics CSV files in the `Data/` directory with the following structure:
```
Data/
├── Player_Name_1/
│   ├── 2023_stats.csv
│   └── 2024_stats.csv
├── Player_Name_2/
│   └── 2024_stats.csv
...
```

Each CSV should contain columns: `game_id`, `date`, `minutes`, `usage`, `ts_pct`, `three_pa_rate`, `rim_attempt_rate`, `ast_pct`, `tov_pct`, `orb_pct`, `drb_pct`, `stl_pct`, `blk_pct`, `ft_rate`, `pf`, `role`.

## Quick Start

### Complete Prediction Pipeline

Run the full pipeline for a game:

```bash
python -m src.cli.main full-pipeline \
  --game-id G001 \
  --player Stephen_Curry \
  --player Klay_Thompson \
  --opponent-id LAL \
  --output-dir outputs/game_G001
```

This executes: region construction → global simulation → local models → blending → report generation.

### CLI Usage

#### 1. Build Efficiency Frontiers

Fit pairwise efficiency frontiers for a season:

```bash
python -m src.cli.main build-frontiers \
  --season 2024 \
  --quantile 0.9 \
  --strata role \
  --output-dir artifacts/frontiers
```

#### 2. Run Global Simulation

Generate probabilistic forecasts using capability regions:

```bash
python -m src.cli.main simulate-global \
  --game-id G001 \
  --player Stephen_Curry \
  --opponent-id LAL \
  --trials 20000 \
  --seed 42 \
  --output-json \
  --output-pdf
```

#### 3. Train Local Models

Train event-specific models:

```bash
# Train all local models
python -m src.cli.main train-local --event-type all --cv-folds 5

# Train specific model
python -m src.cli.main train-local --event-type rebound --cv-folds 5
```

#### 4. Blend Predictions

Combine global and local predictions:

```bash
python -m src.cli.main blend \
  --game-id G001 \
  --global-weight 0.6 \
  --local-weight 0.4 \
  --strategy weighted
```

#### 5. Train Baseline Models

Train traditional ML models for comparison:

```bash
# Train all baselines
python -m src.cli.main baselines-train --model-type all --season 2024

# Train specific model
python -m src.cli.main baselines-train --model-type xgboost --season 2024
```

#### 6. Run Benchmarks

Compare all models:

```bash
python -m src.cli.main benchmark \
  --window rolling_30_games \
  --models all \
  --output-pdf \
  --output-md
```

### API Usage

#### Start the Server

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Or with custom configuration:

```bash
uvicorn src.api.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --timeout-keep-alive 30
```

#### API Endpoints

**Health Check**

```bash
curl http://localhost:8000/health
```

**Run Global Simulation**

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "G001",
    "date": "2024-01-15",
    "team_id": "GSW",
    "opponent_id": "LAL",
    "venue": "home",
    "pace": 100.5,
    "opponent_context": {
      "opponent_id": "LAL",
      "scheme_drop_rate": 0.4,
      "scheme_switch_rate": 0.3,
      "scheme_ice_rate": 0.2,
      "blitz_rate": 0.15,
      "rim_deterrence_index": 1.2,
      "def_reb_strength": 1.1,
      "foul_discipline_index": 0.9,
      "pace": 100.5,
      "help_nail_freq": 0.35
    },
    "players": [
      {
        "player_id": "curry_stephen",
        "role": "starter",
        "exp_minutes": 34.0,
        "exp_usage": 0.32,
        "posterior_mu": [30.0, 5.0, 6.0, 1.5, 0.3, 2.5],
        "posterior_sigma": [[25, 2, 3, 0.5, 0.1, 1], [2, 4, 1, 0.2, 0.05, 0.3], [3, 1, 9, 0.3, 0.1, 0.5], [0.5, 0.2, 0.3, 1, 0.05, 0.2], [0.1, 0.05, 0.1, 0.05, 0.25, 0.1], [1, 0.3, 0.5, 0.2, 0.1, 4]]
      }
    ],
    "n_trials": 20000,
    "seed": 42
  }'
```

**Blend Global and Local Predictions**

```bash
curl -X POST http://localhost:8000/simulate-local \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "G001",
    "player_id": "curry_stephen",
    "global_distributions": {
      "PTS": [28.5, 31.2, 29.8, ...],
      "REB": [5.1, 4.8, 5.5, ...],
      "AST": [6.2, 7.1, 5.8, ...]
    },
    "local_predictions": {
      "PTS": 30.2,
      "REB": 5.3,
      "AST": 6.5
    },
    "blend_weights": {
      "global": 0.6,
      "local": 0.4
    }
  }'
```

**Run Benchmark**

```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_window": "rolling_30_games",
    "models": ["original_global_only", "blended_global_plus_local", "baselines_xgboost"],
    "metrics": ["mae", "rmse", "crps", "coverage_80"]
  }'
```

#### Interactive API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

The system is configured via `configs/default.yaml`. Key configuration sections:

### Feature Engineering

```yaml
features:
  rolling_window_games: 20        # Number of games for rolling statistics
  decay_half_life: 7              # Exponential decay half-life
  min_games_required: 10          # Minimum games for posterior computation
  missingness_threshold: 0.05     # Maximum allowed missing data (5%)
```

### Capability Regions

```yaml
regions:
  credibility_alpha: 0.80         # Credibility level for ellipsoids
  sampling_method: "hit_and_run"  # Sampling method (hit_and_run or rejection)
  n_samples: 5000                 # Number of samples per region
  burn_in: 1000                   # MCMC burn-in samples
  thinning: 10                    # MCMC thinning factor
```

### Global Simulation

```yaml
simulation:
  n_trials: 20000                 # Number of Monte Carlo trials
  n_stints: 5                     # Number of stints per game
  seed: null                      # Random seed (null = random)
  
  # State transition probabilities
  state_transitions:
    Normal: [0.70, 0.15, 0.10, 0.03, 0.02]
    Hot: [0.40, 0.50, 0.05, 0.03, 0.02]
    Cold: [0.50, 0.10, 0.35, 0.03, 0.02]
    FoulRisk: [0.60, 0.10, 0.10, 0.15, 0.05]
    WindDown: [0.30, 0.05, 0.05, 0.05, 0.55]
```

### Local Models

```yaml
local_models:
  cross_validation_folds: 5       # K-fold CV for training
  regularization_c: 1.0           # Logistic regression regularization
  
  blending:
    global_weight: 0.6            # Weight for global predictions
    local_weight: 0.4             # Weight for local predictions
```

### Baseline Models

```yaml
baselines:
  ridge:
    alpha: 1.0
  xgboost:
    max_depth: 6
    n_estimators: 500
    learning_rate: 0.05
  mlp:
    hidden_layers: [128, 64]
    dropout: 0.1
```

### Benchmarking

```yaml
benchmarking:
  evaluation_windows:
    - rolling_30_games
    - monthly
    - playoffs_only
  
  models_to_compare:
    - original_global_only
    - local_only
    - blended_global_plus_local
    - baselines_ridge
    - baselines_xgboost
    - baselines_mlp
```

See `configs/default.yaml` for complete configuration options.

## Key Features

### Global Simulator
- Capability regions (ellipsoid ∩ polytope)
- Markov-Monte Carlo simulation with game states
- Opponent-specific constraints

### Local Models
- Event-specific logistic regression
- Rebound, assist, and shot models
- Blended predictions with global simulator

### Baseline Models
- Ridge regression
- XGBoost
- Multi-layer perceptron (MLP)

### Benchmarking
- Comprehensive accuracy metrics (MAE, RMSE, CRPS, coverage, ECE)
- Efficiency metrics (runtime, memory, adaptation time)
- Statistical significance tests

### Reporting
- Coach one-pager (PDF)
- Analyst detail report (PDF)
- Benchmark comparison report (PDF/Markdown)
- JSON and CSV exports

## Performance Targets

- **Accuracy**: PTS MAE ≤ 4.6 (blended mode), coverage_80 between 0.78-0.86
- **Efficiency**: Per-player inference ≤ 2.5 seconds (blended mode)
- **Calibration**: ECE < 0.05, tail recall ≥ 0.65

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_data_loader.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/

# Run only unit tests
pytest tests/test_*.py -k "not pipeline"

# Run only integration tests
pytest tests/test_pipeline_*.py
```

### Code Style

The project follows PEP 8 guidelines with the following tools:

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Project Structure

```
src/
├── api/                 # FastAPI REST endpoints
│   └── server.py       # Main API server
├── baselines/          # Traditional ML models
│   └── models.py       # Ridge, XGBoost, MLP
├── benchmarks/         # Model comparison
│   └── compare.py      # Benchmarking logic
├── calibration/        # Probability calibration
│   └── fit.py          # Isotonic regression, copulas
├── cli/                # Command-line interface
│   └── main.py         # CLI commands
├── features/           # Feature engineering
│   └── transform.py    # Rolling windows, posteriors
├── frontiers/          # Efficiency frontiers
│   └── fit.py          # Quantile regression
├── local_models/       # Event-specific models
│   ├── rebound.py      # Rebound probability
│   ├── assist.py       # Assist probability
│   ├── shot.py         # Shot probability
│   └── aggregate.py    # Blending logic
├── positional/         # Tracking data (scaffolded)
│   └── ...             # Disabled until data available
├── regions/            # Capability regions
│   ├── build.py        # Region construction
│   └── matchup.py      # Matchup constraints
├── reporting/          # Report generation
│   └── build.py        # PDF, JSON, CSV reports
├── simulation/         # Global simulator
│   └── global_sim.py   # Markov-MC simulation
└── utils/              # Shared utilities
    ├── data_loader.py  # Data loading
    ├── errors.py       # Custom exceptions
    └── logger.py       # Structured logging
```

### Adding New Features

1. **New Local Model**: Add to `src/local_models/` following the pattern in `rebound.py`
2. **New Baseline**: Add to `src/baselines/models.py` with train/predict methods
3. **New Metric**: Add to `src/benchmarks/compare.py` in `compute_accuracy_metrics()`
4. **New Report**: Add to `src/reporting/build.py` with template and generation logic

### Contributing

1. Create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass
4. Format code with black and isort
5. Submit pull request with description

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[API Documentation](docs/API.md)**: Complete REST API reference with examples
- **[CLI Documentation](docs/CLI.md)**: Command-line interface guide with all commands
- **[Configuration Guide](configs/README.md)**: Detailed configuration options
- **[Error Handling](docs/error_handling_logging.md)**: Error handling and logging patterns
- **[Matchup Constraints](docs/matchup_constraints_implementation.md)**: Matchup constraint implementation
- **[Parallelization](docs/parallelization_implementation.md)**: Parallelization strategies

### Quick Links

- Interactive API docs: http://localhost:8000/docs (when server is running)
- Project configuration: `configs/default.yaml`
- Example scripts: `examples/`
- Test fixtures: `fixtures/`

## Troubleshooting

### Common Issues

**Import Errors**:
- Ensure you're in the project root directory
- Verify virtual environment is activated
- Check all dependencies are installed: `pip install -r requirements.txt`

**Data Loading Errors**:
- Verify CSV files are in `Data/` directory
- Check file format matches expected schema
- Ensure no missing required columns

**Simulation Errors**:
- Check for singular covariance matrices (add regularization)
- Verify capability regions are non-empty
- Reduce `n_trials` if memory issues occur

**Performance Issues**:
- Increase `n_workers` for parallelization
- Reduce `n_samples` in region construction
- Use smaller `n_trials` for faster results

See individual documentation files for detailed troubleshooting guides.

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Code Style**: Follow PEP 8, use black for formatting
2. **Type Hints**: Add type hints to all function signatures
3. **Docstrings**: Document all public functions and classes
4. **Tests**: Write tests for new functionality
5. **Documentation**: Update relevant documentation files

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and test
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Commit and push
git commit -m "Add my feature"
git push origin feature/my-feature
```

## License

Proprietary - All rights reserved

## Version

1.0.0

## Contact

For questions, issues, or support:
- Check documentation in `docs/`
- Review configuration in `configs/`
- Check logs in `logs/system.log`
- Contact the development team

## Acknowledgments

This system implements capability-region modeling with Markov-Monte Carlo simulation, combining geometric constraints with probabilistic forecasting for NBA player performance prediction.
>>>>>>> f6ac433 (Creation)
