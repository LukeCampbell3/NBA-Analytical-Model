# Configuration Guide

This document describes all configuration options available in `configs/default.yaml`.

## Table of Contents

- [Data Paths](#data-paths)
- [Feature Engineering](#feature-engineering)
- [Frontier Fitting](#frontier-fitting)
- [Capability Regions](#capability-regions)
- [Global Simulation](#global-simulation)
- [Local Models](#local-models)
- [Baseline Models](#baseline-models)
- [Calibration](#calibration)
- [Benchmarking](#benchmarking)
- [Reporting](#reporting)
- [API Configuration](#api-configuration)
- [Parallelization](#parallelization)
- [Positional Tracking](#positional-tracking)
- [Logging](#logging)
- [Performance Targets](#performance-targets)

## Data Paths

Configure paths for data storage and outputs.

```yaml
data:
  player_stats_dir: "Data"              # Directory with player CSV files
  processed_dir: "Data-Proc-OG"         # Processed data cache
  artifacts_dir: "artifacts"            # Model artifacts and frontiers
  outputs_dir: "outputs"                # Generated predictions and reports
  logs_dir: "logs"                      # System logs
```

**Usage**: These paths are relative to the project root. Ensure directories exist or the system will create them.

## Feature Engineering

Control how historical data is transformed into features.

```yaml
features:
  rolling_window_games: 20              # Number of games for rolling statistics
  decay_half_life: 7                    # Exponential decay half-life (games)
  min_games_required: 10                # Minimum games for posterior computation
  missingness_threshold: 0.05           # Maximum allowed missing data (5%)
```

**Parameters**:

- `rolling_window_games`: Larger windows provide more stable estimates but less responsiveness to recent form
- `decay_half_life`: Smaller values emphasize recent games more heavily
- `min_games_required`: Players with fewer games will use league-average priors
- `missingness_threshold`: Data with more missing values will be rejected

**Recommendations**:
- Use 15-30 games for rolling windows
- Use 5-10 games for decay half-life
- Adjust based on season length and data quality

## Frontier Fitting

Configure efficiency frontier estimation.

```yaml
frontiers:
  quantile: 0.9                         # Quantile for frontier (0.9 = 90th percentile)
  strata:                               # Stratification variables
    - role
    - opponent_scheme_bin
  attribute_pairs:                      # Attribute pairs for frontiers
    - [ts_pct, usage]
    - [three_pa_rate, rim_attempt_rate]
    - [ast_pct, tov_pct]
    - [orb_pct, drb_pct]
```

**Parameters**:

- `quantile`: Higher values (0.9-0.95) capture elite performance; lower values (0.75-0.85) are more conservative
- `strata`: Variables to stratify by (role, opponent_scheme, etc.)
- `attribute_pairs`: Pairs of attributes with trade-off relationships

**Recommendations**:
- Use 0.9 quantile for standard frontiers
- Add more attribute pairs to capture additional trade-offs
- Stratify by role and opponent scheme for context-specific frontiers

## Capability Regions

Configure geometric region construction and sampling.

```yaml
regions:
  credibility_alpha: 0.80               # Credibility level for ellipsoids (80%)
  sampling_method: "hit_and_run"        # Sampling method (hit_and_run or rejection)
  n_samples: 5000                       # Number of samples per region
  burn_in: 1000                         # MCMC burn-in samples
  thinning: 10                          # MCMC thinning factor
  volume_estimation_samples: 10000      # Samples for volume estimation
```

**Parameters**:

- `credibility_alpha`: Higher values (0.85-0.95) create larger regions with more uncertainty
- `sampling_method`: 
  - `hit_and_run`: Efficient for high dimensions, recommended
  - `rejection`: Simpler but slower for complex regions
- `n_samples`: More samples improve distribution quality but increase runtime
- `burn_in`: Discard initial samples to ensure convergence
- `thinning`: Take every Nth sample to reduce autocorrelation

**Recommendations**:
- Use 0.80 for standard credibility
- Use hit-and-run for dimensions > 5
- Use 5000-10000 samples for production
- Adjust burn-in based on convergence diagnostics

## Global Simulation

Configure Markov-Monte Carlo simulation parameters.

```yaml
simulation:
  n_trials: 20000                       # Number of Monte Carlo trials
  n_stints: 5                           # Number of stints per game
  seed: null                            # Random seed (null = random)
```

### State Transition Matrix

Probabilities of transitioning between game states.

```yaml
  state_transitions:
    Normal: [0.70, 0.15, 0.10, 0.03, 0.02]    # [Normal, Hot, Cold, FoulRisk, WindDown]
    Hot: [0.40, 0.50, 0.05, 0.03, 0.02]
    Cold: [0.50, 0.10, 0.35, 0.03, 0.02]
    FoulRisk: [0.60, 0.10, 0.10, 0.15, 0.05]
    WindDown: [0.30, 0.05, 0.05, 0.05, 0.55]
```

Each row must sum to 1.0. Order: [Normal, Hot, Cold, FoulRisk, WindDown]

### State Offsets

Multiplicative factors applied to capability vectors in each state.

```yaml
  state_offsets:
    Hot:
      scoring_efficiency: 1.10          # +10% scoring efficiency
      usage: 1.05                       # +5% usage
    Cold:
      scoring_efficiency: 0.85          # -15% scoring efficiency
      usage: 0.95                       # -5% usage
    FoulRisk:
      minutes: 0.80                     # -20% minutes (foul trouble)
      foul_rate: 1.30                   # +30% foul rate
    WindDown:
      usage: 0.90                       # -10% usage (garbage time)
      assist_rate: 1.10                 # +10% assist rate
```

### Role-Specific Parameters

Minutes and usage distributions by player role.

```yaml
  role_params:
    starter:
      minutes_mean: 33.0                # Expected minutes
      minutes_sigma: 3.0                # Standard deviation
      usage_alpha: 5.0                  # Beta distribution alpha
      usage_beta: 15.0                  # Beta distribution beta
    rotation:
      minutes_mean: 22.0
      minutes_sigma: 5.0
      usage_alpha: 4.0
      usage_beta: 20.0
    bench:
      minutes_mean: 12.0
      minutes_sigma: 6.0
      usage_alpha: 3.0
      usage_beta: 25.0
```

**Recommendations**:
- Use 20000+ trials for stable distributions
- Adjust state transitions based on historical data
- Calibrate state offsets using validation data
- Set role parameters based on team rotation patterns

## Local Models

Configure event-specific models and blending.

```yaml
local_models:
  cross_validation_folds: 5             # K-fold CV for training
  regularization_c: 1.0                 # Logistic regression C parameter
  max_iter: 1000                        # Maximum iterations
  
  blending:
    global_weight: 0.6                  # Weight for global predictions
    local_weight: 0.4                   # Weight for local predictions
```

**Parameters**:

- `cross_validation_folds`: More folds provide better validation but slower training
- `regularization_c`: Smaller values (0.1-1.0) increase regularization
- `global_weight` + `local_weight`: Should sum to 1.0

**Recommendations**:
- Use 5-10 folds for cross-validation
- Start with 0.6/0.4 blend and tune based on benchmarks
- Adjust weights per statistic if needed (e.g., more local weight for rebounds)

## Baseline Models

Configure traditional ML models for benchmarking.

### Ridge Regression

```yaml
baselines:
  ridge:
    alpha: 1.0                          # Regularization strength
```

### XGBoost

```yaml
  xgboost:
    max_depth: 6                        # Maximum tree depth
    n_estimators: 500                   # Number of trees
    learning_rate: 0.05                 # Learning rate
    subsample: 0.8                      # Row sampling ratio
    colsample_bytree: 0.8               # Column sampling ratio
```

### Multi-Layer Perceptron

```yaml
  mlp:
    hidden_layers: [128, 64]            # Hidden layer sizes
    activation: "relu"                  # Activation function
    dropout: 0.1                        # Dropout rate
    max_iter: 500                       # Maximum iterations
    learning_rate_init: 0.001           # Initial learning rate
```

**Recommendations**:
- Ridge: Use alpha 0.1-10.0 based on feature count
- XGBoost: Increase n_estimators for better performance (slower training)
- MLP: Add layers for complex patterns, but watch for overfitting

## Calibration

Configure probability calibration methods.

```yaml
calibration:
  method: "isotonic"                    # Calibration method (isotonic or platt)
  copula_type: "gaussian"               # Copula type (gaussian or vine)
  validation_split: 0.2                 # Validation data fraction
```

**Parameters**:

- `method`: 
  - `isotonic`: Non-parametric, flexible (recommended)
  - `platt`: Parametric, assumes sigmoid relationship
- `copula_type`:
  - `gaussian`: Simpler, faster
  - `vine`: More flexible, captures complex dependencies
- `validation_split`: Fraction of data for calibration fitting

**Recommendations**:
- Use isotonic for most cases
- Use Gaussian copula unless strong non-linear dependencies exist
- Reserve 20-30% of data for calibration

## Benchmarking

Configure model comparison and evaluation.

```yaml
benchmarking:
  evaluation_windows:                   # Evaluation strategies
    - rolling_30_games
    - monthly
    - playoffs_only
  
  models_to_compare:                    # Models to benchmark
    - original_global_only
    - local_only
    - blended_global_plus_local
    - baselines_ridge
    - baselines_xgboost
    - baselines_mlp
```

### Metrics

```yaml
  metrics:
    accuracy:                           # Accuracy metrics
      - mae                             # Mean Absolute Error
      - rmse                            # Root Mean Squared Error
      - crps                            # Continuous Ranked Probability Score
      - coverage_50                     # 50% interval coverage
      - coverage_80                     # 80% interval coverage
      - ece                             # Expected Calibration Error
      - tail_recall_p95                 # Tail event recall (95th percentile)
    efficiency:                         # Efficiency metrics
      - train_time_sec                  # Training time
      - infer_time_ms_per_player        # Inference time per player
      - adaptation_time_ms              # Scheme adaptation time
      - memory_mb                       # Memory usage
    overall:                            # Overall metrics
      - spearman_rank_correlation       # Rank correlation
      - decision_gain_sim               # Decision-making gain
```

**Recommendations**:
- Use rolling_30_games for temporal validation
- Include all models for comprehensive comparison
- Focus on CRPS and coverage for probabilistic evaluation

## Reporting

Configure report generation.

### Coach One-Pager

```yaml
reporting:
  coach_one_pager:
    format: "pdf"                       # Output format
    page_size: "letter"                 # Page size (letter or a4)
    key_stats: ["PTS", "REB", "AST"]    # Stats to highlight
    confidence_level: 0.80              # Confidence interval level
```

### Analyst Detail Report

```yaml
  analyst_detail:
    format: "pdf"
    include_calibration_diagnostics: true
    include_region_visualizations: true
    include_model_comparison: true
```

### Benchmark Report

```yaml
  benchmark_report:
    format: ["pdf", "markdown"]         # Output formats
    include_statistical_tests: true     # Include significance tests
    significance_level: 0.05            # Alpha for statistical tests
```

**Recommendations**:
- Use PDF for presentation, markdown for documentation
- Include calibration diagnostics for model validation
- Set confidence level to match region credibility alpha

## API Configuration

Configure REST API server.

```yaml
api:
  host: "0.0.0.0"                       # Bind address
  port: 8000                            # Port number
  workers: 4                            # Number of worker processes
  timeout: 30                           # Request timeout (seconds)
  enable_cors: true                     # Enable CORS
  log_level: "info"                     # Logging level
```

**Parameters**:

- `host`: Use "0.0.0.0" to accept external connections, "127.0.0.1" for local only
- `workers`: Set to number of CPU cores for production
- `timeout`: Increase for long-running simulations
- `enable_cors`: Enable for web frontend access

**Recommendations**:
- Use 4-8 workers for production
- Set timeout to 60+ seconds for large simulations
- Disable CORS in production if not needed

## Parallelization

Configure parallel processing.

```yaml
parallelization:
  n_workers: null                       # Number of workers (null = all cores)
  enable_progress_bars: true            # Show progress bars
```

**Parameters**:

- `n_workers`: Set to specific number or null to use all available cores
- `enable_progress_bars`: Disable for cleaner logs in production

**Recommendations**:
- Use all cores for batch processing
- Limit workers if running other processes
- Disable progress bars in automated pipelines

## Positional Tracking

Configure positional tracking module (currently scaffolded).

```yaml
positional:
  enabled: false                        # Enable positional tracking
  tracking_data_dir: null               # Directory with tracking data
  spatial_features:                     # Features to derive
    - scv_volume
    - overlap_index
    - spacing_entropy
```

**Note**: This module is scaffolded but disabled until tracking data becomes available. When enabled, it will replace proxy features with measured spatial overlaps.

## Logging

Configure system logging.

```yaml
logging:
  level: "INFO"                         # Log level (DEBUG, INFO, WARNING, ERROR)
  format: "json"                        # Log format (json or text)
  file: "logs/system.log"               # Log file path
  rotation: "daily"                     # Rotation strategy (daily, weekly, size)
  retention_days: 30                    # Days to retain logs
```

**Parameters**:

- `level`: Use DEBUG for development, INFO for production
- `format`: JSON for structured logging, text for human readability
- `rotation`: daily for high-volume, weekly for low-volume
- `retention_days`: Balance storage vs. audit requirements

**Recommendations**:
- Use JSON format for production (easier to parse)
- Set level to INFO or WARNING in production
- Rotate daily and retain 30-90 days

## Performance Targets

Define performance targets for validation.

```yaml
targets:
  accuracy:
    pts_mae_global: 5.0                 # PTS MAE target (global only)
    pts_mae_blended: 4.6                # PTS MAE target (blended)
    coverage_80_min: 0.78               # Minimum 80% coverage
    coverage_80_max: 0.86               # Maximum 80% coverage
    tail_recall_p95_min: 0.65           # Minimum tail recall
  efficiency:
    infer_time_global_max_sec: 2.0      # Max inference time (global)
    infer_time_blended_max_sec: 2.5     # Max inference time (blended)
    infer_time_baseline_max_ms: 20      # Max inference time (baselines)
    adaptation_time_max_ms: 50          # Max adaptation time
```

**Usage**: These targets are used in automated testing and benchmarking to validate system performance.

## Environment Variables

Some settings can be overridden with environment variables:

- `NBA_CONFIG_PATH`: Path to custom config file
- `NBA_DATA_DIR`: Override data directory
- `NBA_LOG_LEVEL`: Override log level
- `NBA_N_WORKERS`: Override number of workers
- `NBA_API_PORT`: Override API port

Example:

```bash
export NBA_LOG_LEVEL=DEBUG
export NBA_N_WORKERS=8
python -m src.cli.main simulate-global --game-id G001 --player Stephen_Curry
```

## Configuration Validation

The system validates configuration on startup. Common errors:

- **Invalid quantile**: Must be between 0 and 1
- **Invalid state transitions**: Each row must sum to 1.0
- **Invalid blend weights**: Must sum to 1.0
- **Missing directories**: Will be created automatically
- **Invalid role**: Must be starter, rotation, or bench

## Custom Configurations

Create custom configurations for different scenarios:

```bash
# Development config with faster settings
cp configs/default.yaml configs/dev.yaml
# Edit dev.yaml: reduce n_trials, n_samples, etc.

# Production config with optimal settings
cp configs/default.yaml configs/prod.yaml
# Edit prod.yaml: increase n_trials, enable all features

# Use custom config
export NBA_CONFIG_PATH=configs/dev.yaml
python -m src.cli.main simulate-global ...
```

## Best Practices

1. **Start with defaults**: The default configuration is tuned for balanced performance
2. **Tune incrementally**: Change one parameter at a time and measure impact
3. **Validate changes**: Run benchmarks after configuration changes
4. **Document changes**: Add comments explaining custom settings
5. **Version control**: Track configuration changes in git
6. **Environment-specific**: Use different configs for dev/staging/prod

## Troubleshooting

### Slow Performance

- Reduce `n_trials` in simulation
- Reduce `n_samples` in regions
- Increase `n_workers` for parallelization
- Use rejection sampling instead of hit-and-run for simple regions

### Poor Accuracy

- Increase `rolling_window_games` for more stable estimates
- Adjust `decay_half_life` to emphasize recent games
- Tune `blend_weights` based on benchmark results
- Increase `credibility_alpha` for wider regions

### Memory Issues

- Reduce `n_trials` and `n_samples`
- Reduce `n_workers` to limit parallel memory usage
- Process games in smaller batches
- Clear cache between runs

### Calibration Issues

- Increase `validation_split` for more calibration data
- Use isotonic calibration instead of Platt
- Check for data leakage in rolling windows
- Verify state transitions sum to 1.0

## Support

For configuration questions or issues:
1. Check this documentation
2. Review `configs/default.yaml` comments
3. Run with `--help` flag for CLI options
4. Check logs in `logs/system.log`
5. Contact the development team
