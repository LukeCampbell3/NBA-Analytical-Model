# Implementation Plan

- [x] 1. Set up project structure and dependencies





  - Create directory structure (src/, configs/, tests/, artifacts/, outputs/, logs/)
  - Create requirements.txt with core dependencies (numpy, pandas, scipy, scikit-learn, statsmodels, pydantic, joblib)
  - Add geometry dependencies (pypoman, cvxpy)
  - Add simulation dependencies (numba)
  - Add reporting dependencies (matplotlib, seaborn, jinja2, weasyprint)
  - Add API dependencies (fastapi, uvicorn)
  - Create configs/default.yaml with system configuration
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement data loading and validation





  - Create src/utils/data_loader.py with DataLoader class
  - Implement load_player_data() to read CSV files from Data directory
  - Implement validate_data() to check missingness < 5%
  - Implement apply_outlier_caps() using role and season quantiles
  - Implement enforce_leakage_control() for temporal ordering
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Implement feature engineering





  - Create src/features/transform.py with FeatureTransform class
  - Implement compute_rolling_features() with exponential decay (half-life=7 games)
  - Implement compute_player_posteriors() to calculate mu and Sigma
  - Implement compute_scalers() and apply_scalers() using RobustScaler
  - Implement join_context() to merge player, opponent, and rotation data
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 4. Implement frontier fitting





  - Create src/frontiers/fit.py with FrontierFitter class
  - Implement fit_frontier() using quantile regression at 90th percentile
  - Implement linearize_frontier() to convert to halfspace representation
  - Implement save_frontier() and load_frontier() for model persistence
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 5. Implement capability region construction






  - Create src/regions/build.py with RegionBuilder class
  - Implement credible_ellipsoid() from posterior with alpha=0.80
  - Implement assemble_halfspaces() from frontiers, schemes, and bounds
  - Implement intersect_ellipsoid_polytope() for region construction
  - Implement sample_region() using hit-and-run MCMC with Numba optimization
  - Implement estimate_volume() and hypervolume_above_baseline()
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 6. Implement matchup constraints





  - Create src/regions/matchup.py with MatchupConstraintBuilder class
  - Implement scheme_to_constraints() to convert opponent schemes to halfspaces
  - Implement role_bounds() for role-specific attribute constraints
  - Implement pairwise_frontiers_for() to retrieve frontier constraints
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 7. Implement global simulator





  - Create src/simulation/global_sim.py with GlobalSimulator class
  - Define GameState enum (Normal, Hot, Cold, FoulRisk, WindDown)
  - Implement sample_minutes() with role-specific distributions
  - Implement sample_usage() with Beta distributions
  - Implement sample_stint_states() with Markov transition matrix
  - Implement apply_state_offsets() for state-specific adjustments
  - Implement project_to_box() to convert capability vectors to box stats
  - Implement simulate_player_game() to run N trials and return distributions
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 8. Implement local rebound model





  - Create src/local_models/rebound.py with ReboundModel class
  - Implement featurize_rebound() with time_to_ball_proxy, crowd_index, reach_margin, seal_angle_proxy
  - Implement fit_rebound_logit() using scikit-learn LogisticRegression
  - Implement predict_rebound_prob() for inference
  - _Requirements: 7.1, 7.4, 7.5_

- [x] 9. Implement local assist model





  - Create src/local_models/assist.py with AssistModel class
  - Implement featurize_assist() with passer_usage, passer_ast_pct, receiver_shot_quality_proxy, opponent_help_nail_freq, lane_risk_proxy
  - Implement fit_assist_logit() using scikit-learn LogisticRegression
  - Implement predict_assist_prob() for inference
  - _Requirements: 7.2, 7.4, 7.5_

- [x] 10. Implement local shot model





  - Create src/local_models/shot.py with ShotModel class
  - Implement featurize_shot() with shooter_ts_context, distance_bin, pullup_vs_catch_proxy, opponent_rim_deterrence
  - Implement fit_shot_logit() using scikit-learn LogisticRegression
  - Implement predict_shot_prob() for inference
  - _Requirements: 7.3, 7.4, 7.5_

- [x] 11. Implement local model aggregation and blending





  - Create src/local_models/aggregate.py with LocalAggregator class
  - Implement local_to_box_expectations() to convert event probs to expected counts
  - Implement blend_global_local() with weighted blending (default: global=0.6, local=0.4)
  - Add recalibration logic to maintain proper uncertainty
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 12. Implement baseline models





  - Create src/baselines/models.py with BaselineModels class
  - Implement build_features() for rolling means/variances, opponent features, role, pace
  - Implement train_ridge() for Ridge regression
  - Implement train_xgboost() with default params (max_depth=6, n_estimators=500, lr=0.05)
  - Implement train_mlp() with layout [128, 64], ReLU, dropout=0.1
  - Implement predict(), save_model(), and load_model()
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 13. Implement calibration





  - Create src/calibration/fit.py with Calibrator class
  - Implement compute_pit() for Probability Integral Transform
  - Implement fit_isotonic() for per-statistic calibration
  - Implement apply_calibration() to transform predictions
  - Implement fit_copula() for multivariate dependencies
  - Implement sample_copula() for correlated predictions
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 14. Implement benchmarking





  - Create src/benchmarks/compare.py with BenchmarkRunner class
  - Implement run_eval_window() to evaluate models on configured windows
  - Implement compute_accuracy_metrics() for MAE, RMSE, CRPS, coverage, ECE, tail_recall_p95
  - Implement compute_efficiency_metrics() for train_time, infer_time, adaptation_time, memory
  - Implement compare_models() to generate side-by-side DataFrame
  - Implement ablation_study() for hyperparameter grid search
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [x] 15. Implement reporting





  - Create src/reporting/build.py with ReportBuilder class
  - Implement build_coach_one_pager() for single-page PDF with key projections
  - Implement build_analyst_detail() for multi-page PDF with full distributions and diagnostics
  - Implement build_benchmark_report() for model comparison PDF with tables and charts
  - Implement write_json_report() and write_csv_summary() for structured outputs
  - Add markdown export for benchmark reports
  - _Requirements: 12.1, 12.2, 12.3, 13.1, 13.2, 13.3, 13.4, 14.1, 14.2, 14.3, 14.4, 14.5_

- [x] 16. Implement REST API





  - Create src/api/server.py with FastAPI application
  - Implement GET /health endpoint
  - Implement POST /simulate endpoint for global simulation
  - Implement POST /simulate-local endpoint for local model predictions
  - Implement POST /benchmark endpoint for model comparison
  - Add request validation using Pydantic models
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

- [x] 17. Implement CLI





  - Create src/cli/main.py with Click command group
  - Implement build-frontiers command
  - Implement regions command
  - Implement simulate-global command
  - Implement train-local command
  - Implement simulate-local command
  - Implement blend command
  - Implement baselines-train and baselines-predict commands
  - Implement benchmark command
  - Implement calibrate and evaluate commands
  - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9_

- [x] 18. Scaffold positional tracking module





  - Create src/positional/ directory with __init__.py
  - Create placeholder ingest_tracking.py with function stubs
  - Create placeholder derive_features.py with function stubs
  - Create placeholder build_spatial_region.py with function stubs
  - Create placeholder simulate_play_states.py with function stubs
  - Add enabled=false to configs/default.yaml
  - Document interface for replacing proxy features
  - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5_

- [x] 19. Add error handling and logging





  - Create src/utils/errors.py with custom exception classes
  - Add JSON structured logging to src/utils/logger.py
  - Add error handling to DataLoader for missing files and invalid data
  - Add error handling to RegionBuilder for singular matrices and empty regions
  - Add error handling to API endpoints with appropriate HTTP status codes
  - Add logging for all major operations (data load, model train, simulation run)
  - _Requirements: All requirements (cross-cutting concern)_

- [x] 20. Implement parallelization





  - Add multiprocessing support to GlobalSimulator for parallel player simulation
  - Add parallel model evaluation to BenchmarkRunner
  - Configure n_workers from environment variable or config
  - Add progress bars using tqdm for long-running operations
  - _Requirements: 6.6, 10.5, 18.1, 18.2, 18.3, 18.4_

- [x] 21. Write unit tests





- [x] 21.1 Write tests for data loading and validation (tests/test_data_loader.py)

  - Test valid CSV loads correctly
  - Test missing file raises error
  - Test validation catches missingness > 5%
  - Test outlier capping
  - _Requirements: 20.1_

- [x] 21.2 Write tests for feature engineering (tests/test_features.py)

  - Test rolling window computation
  - Test exponential decay
  - Test posterior computation
  - Test scaler transformations
  - _Requirements: 20.1_

- [x] 21.3 Write tests for frontiers (tests/test_frontiers.py)

  - Test frontier fitting on toy data
  - Test linearization produces valid halfspaces
  - Test save/load preserves model
  - _Requirements: 20.1_

- [x] 21.4 Write tests for regions (tests/test_regions.py)

  - Test ellipsoid construction
  - Test polytope assembly
  - Test intersection is non-empty
  - Test sampling produces valid points
  - _Requirements: 20.1_

- [x] 21.5 Write tests for global simulation (tests/test_global_sim.py)

  - Test state transitions
  - Test state offsets
  - Test box projection
  - Test reproducibility with seed
  - _Requirements: 20.1_

- [x] 21.6 Write tests for local models (tests/test_local_rebound.py, tests/test_local_assist.py, tests/test_local_shot.py)


  - Test featurization
  - Test model training
  - Test predictions in [0,1]
  - Test aggregation consistency
  - _Requirements: 20.1_

- [x] 21.7 Write tests for benchmarking (tests/test_benchmarks.py)

  - Test metric computation on toy data
  - Test comparison table structure
  - Test efficiency measurement
  - _Requirements: 20.1_

- [x] 22. Write integration tests








- [x] 22.1 Write full pipeline test (tests/test_pipeline_pregame.py)




  - Test end-to-end: load → features → frontiers → regions → simulate → report
  - Verify output files created
  - _Requirements: 20.2_

- [x] 22.2 Write benchmark pipeline test (tests/test_pipeline_benchmark.py)



  - Test: train all models → predict → compute metrics → generate report
  - Verify all models complete
  - Verify benchmark report structure
  - _Requirements: 20.2_

- [x] 23. Create test fixtures





  - Create fixtures/toy_game_inputs.json with sample game context
  - Create fixtures/small_eval_window.parquet with sample evaluation data
  - _Requirements: 20.5_

- [x] 24. Add documentation





  - Create README.md with project overview and quickstart
  - Add docstrings to all public functions and classes
  - Create API documentation using FastAPI auto-docs
  - Create CLI help text for all commands
  - Document configuration options in configs/README.md
