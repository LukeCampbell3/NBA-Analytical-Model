# Requirements Document

## Introduction

This system provides a unified framework for NBA player performance prediction using capability-region simulation combined with local sub-problem models. The system predicts per-game statistics (points, rebounds, assists, etc.) by modeling player capabilities as geometric regions constrained by opponent schemes, role boundaries, and historical performance. It includes rigorous benchmarking against traditional ML baselines and produces publication-ready reports for coaches and analysts. The positional tracking module is scaffolded but disabled until tracking data becomes available.

## Glossary

- **Capability Region**: A geometric region (ellipsoid intersected with polytope) representing the feasible performance space for a player given their historical posterior, frontier constraints, opponent scheme, and role bounds.
- **Global Simulator**: The Markov-Monte Carlo simulation engine that samples from capability regions and applies game-state transitions (Normal, Hot, Cold, FoulRisk, WindDown).
- **Local Models**: Factorized sub-problem models for specific events (rebounds, assists, shots) using logistic regression on game-slice features.
- **Frontier Model**: Pairwise efficiency frontiers fitted to historical data defining trade-off boundaries between performance attributes.
- **Baseline Models**: Traditional ML approaches (Ridge, XGBoost, MLP) used for comparison benchmarking.
- **Scheme Constraints**: Defensive strategy parameters (drop rate, switch rate, ice rate, blitz rate) that constrain player capability regions.
- **Box Stats**: Traditional counting statistics (PTS, REB, AST, STL, BLK, TOV, FGA, 3PA, FTA, PF).
- **CRPS**: Continuous Ranked Probability Score, a metric for evaluating probabilistic forecasts.
- **ECE**: Expected Calibration Error, measuring how well predicted probabilities match observed frequencies.
- **PIT**: Probability Integral Transform, used for calibration diagnostics.

## Requirements

### Requirement 1

**User Story:** As a data scientist, I want to load and validate player game statistics from CSV files, so that I can ensure data quality before model training.

#### Acceptance Criteria

1. WHEN the DataLoader receives a player name and season year, THE DataLoader SHALL load the corresponding CSV file from the Data directory.
2. THE DataLoader SHALL validate that missingness does not exceed 5% for any required field.
3. THE DataLoader SHALL apply outlier capping based on role and season quantiles.
4. THE DataLoader SHALL enforce leakage control by using rolling windows that cut before the forecast date.
5. THE DataLoader SHALL return a validated DataFrame with standardized column names matching the data_requirements specification.

### Requirement 2

**User Story:** As a data scientist, I want to compute player posteriors from historical data, so that I can define the ellipsoidal component of capability regions.

#### Acceptance Criteria

1. THE FeatureTransform SHALL compute rolling window statistics over 15-30 games with exponential decay (half-life of 7 games).
2. THE FeatureTransform SHALL compute player posterior distributions (mu, Sigma) for each player based on their rolling window data.
3. THE FeatureTransform SHALL apply RobustScaler transformations to normalize features.
4. THE FeatureTransform SHALL join player data with opponent features and rotation priors.
5. THE FeatureTransform SHALL return a dictionary mapping player_id to posterior parameters.

### Requirement 3

**User Story:** As a data scientist, I want to fit efficiency frontiers from historical data, so that I can define pairwise trade-off constraints for capability regions.

#### Acceptance Criteria

1. THE FrontierFitter SHALL fit frontier models for specified attribute pairs (x, y) stratified by role and opponent scheme.
2. THE FrontierFitter SHALL use the 90th percentile quantile for frontier estimation.
3. THE FrontierFitter SHALL linearize fitted frontiers into halfspace representations.
4. THE FrontierFitter SHALL save frontier models to disk in a serialized format.
5. THE FrontierFitter SHALL load previously saved frontier models from disk.

### Requirement 4

**User Story:** As a data scientist, I want to construct capability regions for players, so that I can define the feasible performance space for simulation.

#### Acceptance Criteria

1. THE RegionBuilder SHALL construct a credible ellipsoid from player posterior with alpha=0.80.
2. THE RegionBuilder SHALL assemble halfspaces from frontier constraints, scheme constraints, role bounds, and attribute bounds.
3. THE RegionBuilder SHALL intersect the ellipsoid with the halfspace polytope to create the capability region.
4. THE RegionBuilder SHALL sample N points from the capability region using rejection sampling or hit-and-run.
5. THE RegionBuilder SHALL estimate region volume and compute hypervolume above baseline.

### Requirement 5

**User Story:** As a data scientist, I want to apply opponent-specific matchup constraints, so that capability regions reflect defensive schemes and player roles.

#### Acceptance Criteria

1. THE MatchupConstraintBuilder SHALL convert opponent scheme parameters (drop_rate, switch_rate, ice_rate, blitz_rate) into halfspace constraints.
2. THE MatchupConstraintBuilder SHALL apply role-specific attribute bounds based on player role (starter, rotation, bench).
3. THE MatchupConstraintBuilder SHALL retrieve pairwise frontier constraints for the player role and opponent scheme bin.
4. THE MatchupConstraintBuilder SHALL return a list of halfspace constraints for region construction.

### Requirement 6

**User Story:** As a simulation engineer, I want to run global game simulations, so that I can generate probabilistic forecasts for player performance.

#### Acceptance Criteria

1. THE GlobalSimulator SHALL sample player minutes from a role-specific distribution with configured sigma values.
2. THE GlobalSimulator SHALL sample player usage from a Beta distribution with role-specific parameters.
3. THE GlobalSimulator SHALL simulate stint states (Normal, Hot, Cold, FoulRisk, WindDown) using a Markov transition matrix.
4. THE GlobalSimulator SHALL apply state-specific offsets to sampled capability vectors.
5. THE GlobalSimulator SHALL project capability vectors to box statistics using minutes and opponent context.
6. THE GlobalSimulator SHALL run N=20000 trials and return distributions, risk metrics, and hypervolume index.

### Requirement 7

**User Story:** As a data scientist, I want to train local sub-problem models for rebounds, assists, and shots, so that I can capture event-level dynamics not modeled globally.

#### Acceptance Criteria

1. THE LocalModelTrainer SHALL featurize rebound events using time_to_ball_proxy, crowd_index, reach_margin, and seal_angle_proxy.
2. THE LocalModelTrainer SHALL featurize assist events using passer_usage, passer_ast_pct, receiver_shot_quality_proxy, and opponent_help_nail_freq.
3. THE LocalModelTrainer SHALL featurize shot events using shooter_ts_context, distance_bin, pullup_vs_catch_proxy, and opponent_rim_deterrence.
4. THE LocalModelTrainer SHALL fit logistic regression models for each event type using k-fold cross-validation.
5. THE LocalModelTrainer SHALL save trained local models to disk.

### Requirement 8

**User Story:** As a simulation engineer, I want to aggregate local model predictions into box-level expectations, so that I can blend local and global forecasts.

#### Acceptance Criteria

1. THE LocalAggregator SHALL map local event probabilities to expected counting stats using minutes, usage, and pace.
2. THE LocalAggregator SHALL blend global simulation distributions with local expectations using configured weights (default: global=0.6, local=0.4).
3. THE LocalAggregator SHALL recalibrate blended distributions to maintain proper uncertainty quantification.
4. THE LocalAggregator SHALL return a dictionary with blended distributions for all box statistics.

### Requirement 9

**User Story:** As a data scientist, I want to train traditional ML baseline models, so that I can benchmark the capability-region approach.

#### Acceptance Criteria

1. THE BaselineTrainer SHALL build feature matrices from box rolling means/variances, opponent features, role, and pace.
2. THE BaselineTrainer SHALL train Ridge regression models for each target statistic.
3. THE BaselineTrainer SHALL train XGBoost models with configured hyperparameters (max_depth=6, n_estimators=500, learning_rate=0.05).
4. THE BaselineTrainer SHALL train MLP models with layout [128, 64], ReLU activation, and dropout=0.1.
5. THE BaselineTrainer SHALL save trained baseline models to disk.

### Requirement 10

**User Story:** As a researcher, I want to run comprehensive benchmarks comparing all models, so that I can evaluate accuracy, efficiency, and calibration.

#### Acceptance Criteria

1. THE BenchmarkRunner SHALL evaluate models on configured evaluation windows (rolling_30_games, monthly, playoffs_only).
2. THE BenchmarkRunner SHALL compute accuracy metrics (MAE, RMSE, CRPS, coverage_50_80, ECE, tail_recall_p95) for each statistic.
3. THE BenchmarkRunner SHALL measure efficiency metrics (train_time_sec, infer_time_ms_per_player, adaptation_time_ms, memory_mb).
4. THE BenchmarkRunner SHALL compute overall metrics (Spearman rank correlation, decision_gain_sim).
5. THE BenchmarkRunner SHALL compare models (original_global_only, local_only, blended_global_plus_local, baselines_ridge, baselines_xgboost, baselines_mlp).
6. THE BenchmarkRunner SHALL produce a side-by-side comparison DataFrame with all metrics.

### Requirement 11

**User Story:** As a data scientist, I want to calibrate model predictions, so that predicted probabilities match observed frequencies.

#### Acceptance Criteria

1. THE Calibrator SHALL compute PIT values from true outcomes and predicted samples.
2. THE Calibrator SHALL fit isotonic regression models per statistic for calibration.
3. THE Calibrator SHALL apply calibration transformations to new predictions.
4. THE Calibrator SHALL fit copula models to capture multivariate dependencies between statistics.
5. THE Calibrator SHALL sample from fitted copulas to generate correlated predictions.

### Requirement 12

**User Story:** As a coach, I want to receive a one-page PDF report with key player projections, so that I can make informed game-day decisions.

#### Acceptance Criteria

1. THE ReportBuilder SHALL generate a coach one-pager PDF with player projections, confidence intervals, and risk flags.
2. THE ReportBuilder SHALL include visualizations of key statistics (PTS, REB, AST) with uncertainty bands.
3. THE ReportBuilder SHALL highlight players with high variance or tail risk.
4. THE ReportBuilder SHALL format the report for printing on a single page.

### Requirement 13

**User Story:** As an analyst, I want to receive a detailed PDF report with full distributions and diagnostics, so that I can perform deep analysis.

#### Acceptance Criteria

1. THE ReportBuilder SHALL generate an analyst detail PDF with full distribution plots for all statistics.
2. THE ReportBuilder SHALL include calibration diagnostics (PIT histograms, reliability diagrams).
3. THE ReportBuilder SHALL show capability region visualizations and hypervolume metrics.
4. THE ReportBuilder SHALL include model comparison tables and efficiency metrics.

### Requirement 14

**User Story:** As a researcher, I want to generate benchmark reports comparing all models, so that I can publish results.

#### Acceptance Criteria

1. THE ReportBuilder SHALL generate a benchmark report PDF with side-by-side model comparison tables.
2. THE ReportBuilder SHALL include accuracy metrics (MAE, RMSE, CRPS, coverage, ECE, tail recall) for all models and statistics.
3. THE ReportBuilder SHALL include efficiency metrics (runtime, memory, adaptation time) for all models.
4. THE ReportBuilder SHALL include statistical significance tests and confidence intervals for metric differences.
5. THE ReportBuilder SHALL generate a markdown version of the benchmark report for documentation.

### Requirement 15

**User Story:** As a developer, I want to expose a REST API for simulation requests, so that external systems can integrate with the prediction engine.

#### Acceptance Criteria

1. THE API SHALL expose a GET /health endpoint that returns service status.
2. WHEN a POST /simulate request is received with game context and player list, THE API SHALL return global simulation results with distributions and calibration badges.
3. WHEN a POST /simulate-local request is received with game context and event subset, THE API SHALL return local model predictions (rebound_prob, assist_prob, shot_prob) and blended expectations.
4. WHEN a POST /benchmark request is received with evaluation window and model list, THE API SHALL return benchmark summary tables and per-model metrics.
5. THE API SHALL validate all request payloads and return appropriate error messages for invalid inputs.

### Requirement 16

**User Story:** As a developer, I want to use a CLI for all system operations, so that I can automate workflows and integrate with scripts.

#### Acceptance Criteria

1. THE CLI SHALL provide a build-frontiers command that fits frontiers for a specified season, strata, and quantile.
2. THE CLI SHALL provide a regions command that constructs capability regions for specified game context and players.
3. THE CLI SHALL provide a simulate-global command that runs global simulation with configurable trials, seed, and output formats.
4. THE CLI SHALL provide a train-local command that trains local models for specified event types with cross-validation.
5. THE CLI SHALL provide a simulate-local command that runs local model inference for specified game and players.
6. THE CLI SHALL provide a blend command that combines global and local predictions with configurable strategy.
7. THE CLI SHALL provide baselines-train and baselines-predict commands for traditional ML models.
8. THE CLI SHALL provide a benchmark command that runs comprehensive model comparisons with configurable windows and output formats.
9. THE CLI SHALL provide calibrate and evaluate commands for model calibration and evaluation.

### Requirement 17

**User Story:** As a researcher, I want the system to meet accuracy targets, so that predictions are reliable for decision-making.

#### Acceptance Criteria

1. THE System SHALL achieve PTS_MAE ≤ 5.0 for global_only mode.
2. THE System SHALL achieve coverage_80 between 0.78 and 0.84 for global_only mode.
3. THE System SHALL achieve PTS_MAE ≤ 4.6 for blended mode.
4. THE System SHALL achieve coverage_80 between 0.78 and 0.86 for blended mode.
5. THE System SHALL achieve tail_recall_p95 ≥ 0.65 for blended mode.

### Requirement 18

**User Story:** As a developer, I want the system to meet efficiency targets, so that it can be used in real-time applications.

#### Acceptance Criteria

1. THE System SHALL complete per-player inference in ≤ 2.0 seconds for global_only mode.
2. THE System SHALL complete per-player inference in ≤ 2.5 seconds for blended mode.
3. THE System SHALL complete baseline ML inference in ≤ 20 milliseconds per player.
4. THE System SHALL adapt to scheme toggles in ≤ 50 milliseconds for global or blended mode.

### Requirement 19

**User Story:** As a future developer, I want the positional tracking module scaffolded, so that it can be enabled when tracking data becomes available.

#### Acceptance Criteria

1. THE System SHALL include a positional tracking module in src/positional/ with enabled=false configuration.
2. THE PositionalModule SHALL include placeholder functions for ingest_tracking, derive_features, build_spatial_region, and simulate_play_states.
3. THE PositionalModule SHALL document the interface for replacing proxy features with measured spatial overlaps.
4. THE PositionalModule SHALL not execute any positional tracking code when enabled=false.
5. WHEN enabled=true, THE PositionalModule SHALL derive SCV (spatial capability volume) and overlap metrics from tracking data.

### Requirement 20

**User Story:** As a developer, I want comprehensive unit and integration tests, so that I can ensure system reliability.

#### Acceptance Criteria

1. THE TestSuite SHALL include unit tests for frontiers, regions, global simulation, and all local models.
2. THE TestSuite SHALL include unit tests for benchmarking functions.
3. THE TestSuite SHALL include integration tests for the full pregame pipeline.
4. THE TestSuite SHALL include integration tests for the benchmark pipeline.
5. THE TestSuite SHALL use fixtures with toy game inputs and small evaluation windows for reproducible testing.
