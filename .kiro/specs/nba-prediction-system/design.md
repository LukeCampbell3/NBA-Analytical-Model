# Design Document

## Overview

The NBA Player Performance Prediction System is a sophisticated framework that combines geometric capability-region modeling with local event-level models to generate probabilistic forecasts of player statistics. The system architecture consists of three main prediction pathways:

1. **Global Simulator**: Uses capability regions (ellipsoid ∩ polytope) with Markov-Monte Carlo simulation
2. **Local Models**: Event-specific logistic regression for rebounds, assists, and shots
3. **Baseline Models**: Traditional ML (Ridge, XGBoost, MLP) for benchmarking

The system is designed for modularity, allowing each component to operate independently or in blended mode. All predictions are rigorously calibrated and benchmarked against multiple evaluation metrics.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Player Stats │  │ Opponent     │  │ Rotation     │          │
│  │ (CSV)        │  │ Features     │  │ Priors       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Feature Engineering                         │
│  • Rolling windows (15-30 games, decay half-life=7)             │
│  • Posterior computation (mu, Sigma)                             │
│  • Context joining (player + opponent + rotation)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Global     │  │    Local     │  │   Baseline   │
    │  Simulator   │  │   Models     │  │   Models     │
    └──────────────┘  └──────────────┘  └──────────────┘
           │                 │                 │
           └────────┬────────┴────────┬────────┘
                    ▼                 ▼
         ┌──────────────────┐  ┌──────────────────┐
         │   Calibration    │  │   Benchmarking   │
         └──────────────────┘  └──────────────────┘
                    │                 │
                    └────────┬────────┘
                             ▼
              ┌──────────────────────────┐
              │   Reporting & API        │
              │  • Coach PDF             │
              │  • Analyst PDF           │
              │  • Benchmark Report      │
              │  • REST API              │
              │  • CLI                   │
              └──────────────────────────┘
```

### Module Organization

- **src/features/**: Feature engineering and transformation
- **src/frontiers/**: Efficiency frontier fitting and linearization
- **src/regions/**: Capability region construction and sampling
- **src/simulation/**: Global Markov-MC simulator
- **src/local_models/**: Event-specific models (rebound, assist, shot)
- **src/baselines/**: Traditional ML models for comparison
- **src/calibration/**: Probability calibration and copula fitting
- **src/benchmarks/**: Model comparison and evaluation
- **src/reporting/**: PDF/JSON/CSV report generation
- **src/api/**: FastAPI REST endpoints
- **src/cli/**: Command-line interface
- **src/positional/**: Tracking data module (scaffolded, disabled)
- **src/utils/**: Shared utilities

## Components and Interfaces

### 1. Data Loading and Validation

**Module**: `src/utils/data_loader.py`

**Purpose**: Load and validate player statistics from CSV files.

**Key Classes**:

```python
class DataLoader:
    def load_player_data(self, player_name: str, year: int, data_dir: str = "Data") -> pd.DataFrame
    def validate_data(self, df: pd.DataFrame) -> ValidationResult
    def apply_outlier_caps(self, df: pd.DataFrame, role: str, season: int) -> pd.DataFrame
    def enforce_leakage_control(self, df: pd.DataFrame, forecast_date: datetime) -> pd.DataFrame
```

**Interface**:
- Input: Player name, season year, data directory path
- Output: Validated DataFrame with standardized columns
- Validation: Checks missingness < 5%, applies outlier caps, enforces temporal ordering

### 2. Feature Engineering

**Module**: `src/features/transform.py`

**Purpose**: Compute rolling statistics, player posteriors, and join contextual features.

**Key Classes**:

```python
class FeatureTransform:
    def __init__(self, window_games: int = 20, decay_half_life: int = 7)
    
    def compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame
    def compute_player_posteriors(self, df: pd.DataFrame) -> Dict[str, PosteriorParams]
    def compute_scalers(self, X: pd.DataFrame) -> RobustScalerParams
    def apply_scalers(self, X: pd.DataFrame, params: RobustScalerParams) -> pd.DataFrame
    def join_context(self, df_player: pd.DataFrame, df_opponent: pd.DataFrame, 
                     df_rotation: pd.DataFrame) -> pd.DataFrame

class PosteriorParams:
    mu: np.ndarray  # Mean vector
    Sigma: np.ndarray  # Covariance matrix
    player_id: str
    as_of_date: datetime
```

**Design Decisions**:
- Use exponential decay weighting for rolling windows to emphasize recent games
- Compute posteriors in the original feature space before scaling
- RobustScaler chosen over StandardScaler for outlier resistance

### 3. Frontier Fitting

**Module**: `src/frontiers/fit.py`

**Purpose**: Fit efficiency frontiers defining trade-offs between performance attributes.

**Key Classes**:

```python
class FrontierModel:
    x_attr: str
    y_attr: str
    strata: Dict[str, Any]  # role, opponent_scheme_bin
    quantile: float
    coefficients: np.ndarray
    
class FrontierFitter:
    def fit_frontier(self, data: pd.DataFrame, x: str, y: str, 
                     strata: Dict, quantile: float = 0.9) -> FrontierModel
    def linearize_frontier(self, model: FrontierModel, grid: np.ndarray) -> List[Halfspace]
    def save_frontier(self, model: FrontierModel, path: str) -> None
    def load_frontier(self, path: str) -> FrontierModel
```

**Halfspace Representation**:

```python
class Halfspace:
    normal: np.ndarray  # Normal vector a
    offset: float       # Offset b, defines a^T x <= b
```

**Design Decisions**:
- Use quantile regression at 90th percentile to capture efficient frontier
- Stratify by role and opponent scheme for context-specific frontiers
- Linearize using piecewise approximation for polytope representation

### 4. Capability Region Construction

**Module**: `src/regions/build.py`

**Purpose**: Construct geometric capability regions from posteriors, frontiers, and constraints.

**Key Classes**:

```python
class Ellipsoid:
    center: np.ndarray
    shape_matrix: np.ndarray  # A in (x-c)^T A (x-c) <= 1
    alpha: float  # Credibility level

class HPolytope:
    halfspaces: List[Halfspace]  # A x <= b representation
    
class CapabilityRegion:
    ellipsoid: Ellipsoid
    polytope: HPolytope
    volume_estimate: float
    hypervolume_above_baseline: float

class RegionBuilder:
    def credible_ellipsoid(self, mu: np.ndarray, Sigma: np.ndarray, 
                          alpha: float = 0.80) -> Ellipsoid
    def assemble_halfspaces(self, frontiers: List[FrontierModel], 
                           scheme_constraints: List[Halfspace],
                           role_bounds: List[Halfspace]) -> HPolytope
    def intersect_ellipsoid_polytope(self, E: Ellipsoid, H: HPolytope) -> CapabilityRegion
    def sample_region(self, region: CapabilityRegion, n: int, seed: int) -> np.ndarray
    def estimate_volume(self, region: CapabilityRegion, n_samples: int = 10000) -> float
    def hypervolume_above_baseline(self, region: CapabilityRegion, 
                                   baseline: Dict[str, float]) -> float
```

**Sampling Strategy**:
- Use hit-and-run MCMC for efficient sampling from ellipsoid ∩ polytope
- Burn-in period of 1000 samples, then collect every 10th sample
- Validate samples satisfy all constraints before returning

### 5. Matchup Constraints

**Module**: `src/regions/matchup.py`

**Purpose**: Convert opponent schemes and player roles into geometric constraints.

**Key Classes**:

```python
class MatchupConstraintBuilder:
    def scheme_to_constraints(self, opponent_row: pd.Series, 
                             toggles: Dict[str, bool]) -> List[Halfspace]
    def role_bounds(self, role: str, attribute_bounds: Dict) -> List[Halfspace]
    def pairwise_frontiers_for(self, player_role: str, 
                               opponent_scheme_bin: str) -> List[Halfspace]
```

**Constraint Examples**:

- High blitz_rate → constraint on usage and turnover rate
- High rim_deterrence → constraint on rim_attempt_rate and efficiency
- Starter role → tighter bounds on minutes (28-38), higher usage floor
- Bench role → wider bounds on minutes (5-20), lower usage ceiling

### 6. Global Simulator

**Module**: `src/simulation/global_sim.py`

**Purpose**: Run Markov-Monte Carlo simulation with game-state transitions.

**Key Classes**:

```python
class GameState(Enum):
    NORMAL = "Normal"
    HOT = "Hot"
    COLD = "Cold"
    FOUL_RISK = "FoulRisk"
    WIND_DOWN = "WindDown"

class GlobalSimulator:
    def __init__(self, n_trials: int = 20000, n_stints: int = 5, seed: int = None)
    
    def sample_minutes(self, player_id: str, ctx: GameContext) -> float
    def sample_usage(self, player_id: str, ctx: GameContext, minutes: float) -> float
    def sample_stint_states(self, T: np.ndarray, P: np.ndarray, seed: int) -> List[GameState]
    def apply_state_offsets(self, x: np.ndarray, states: List[GameState], 
                           minutes_split: List[float]) -> np.ndarray
    def project_to_box(self, x: np.ndarray, minutes: float, 
                      opp_ctx: OpponentContext) -> Dict[str, float]
    def simulate_player_game(self, region: CapabilityRegion, opp_ctx: OpponentContext, 
                            N: int, seed: int) -> SimulationResult

class SimulationResult:
    distributions: Dict[str, np.ndarray]  # Stat name -> N samples
    risk_metrics: Dict[str, float]  # VaR, CVaR, tail probabilities
    hypervolume_index: float
    metadata: Dict[str, Any]
```

**State Transition Matrix** (example):
```
         Normal  Hot   Cold  FoulRisk  WindDown
Normal   0.70   0.15  0.10  0.03      0.02
Hot      0.40   0.50  0.05  0.03      0.02
Cold     0.50   0.10  0.35  0.03      0.02
FoulRisk 0.60   0.10  0.10  0.15      0.05
WindDown 0.30   0.05  0.05  0.05      0.55
```

**State Offsets** (applied to capability vector):
- Hot: +10% to scoring efficiency, +5% to usage
- Cold: -15% to scoring efficiency, -5% to usage
- FoulRisk: -20% to minutes, +30% to foul rate
- WindDown: -10% to usage, +10% to assist rate

### 7. Local Models

**Module**: `src/local_models/`

**Purpose**: Event-specific models for rebounds, assists, and shots.

**Rebound Model** (`rebound.py`):

```python
class ReboundModel:
    def featurize_rebound(self, game_slice_df: pd.DataFrame) -> pd.DataFrame:
        # Features: time_to_ball_proxy, crowd_index, reach_margin, seal_angle_proxy
        pass
    
    def fit_rebound_logit(self, df: pd.DataFrame) -> LogisticRegression:
        pass
    
    def predict_rebound_prob(self, model: LogisticRegression, df: pd.DataFrame) -> np.ndarray:
        pass
```

**Assist Model** (`assist.py`):

```python
class AssistModel:
    def featurize_assist(self, game_slice_df: pd.DataFrame) -> pd.DataFrame:
        # Features: passer_usage, passer_ast_pct, receiver_shot_quality_proxy, 
        #           opponent_help_nail_freq, lane_risk_proxy
        pass
    
    def fit_assist_logit(self, df: pd.DataFrame) -> LogisticRegression:
        pass
    
    def predict_assist_prob(self, model: LogisticRegression, df: pd.DataFrame) -> np.ndarray:
        pass
```

**Shot Model** (`shot.py`):

```python
class ShotModel:
    def featurize_shot(self, game_slice_df: pd.DataFrame) -> pd.DataFrame:
        # Features: shooter_ts_context, distance_bin, pullup_vs_catch_proxy, 
        #           opponent_rim_deterrence
        pass
    
    def fit_shot_logit(self, df: pd.DataFrame) -> LogisticRegression:
        pass
    
    def predict_shot_prob(self, model: LogisticRegression, df: pd.DataFrame) -> np.ndarray:
        pass
```

**Aggregation** (`aggregate.py`):

```python
class LocalAggregator:
    def local_to_box_expectations(self, local_probs: Dict[str, np.ndarray], 
                                  minutes: float, usage: float, pace: float) -> Dict[str, float]:
        # Convert event probabilities to expected counts
        # E[REB] = sum(rebound_probs) * expected_opportunities
        # E[AST] = sum(assist_probs) * expected_possessions
        # E[PTS] = sum(shot_probs * expected_points_per_shot)
        pass
    
    def blend_global_local(self, global_summary: Dict[str, np.ndarray], 
                          local_expect: Dict[str, float], 
                          weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        # Weighted blend: w_g * global + w_l * local
        # Recalibrate to maintain proper uncertainty
        pass
```

**Design Decisions**:
- Use logistic regression for interpretability and speed
- Proxy features (time_to_ball, seal_angle) until tracking data available
- Train on game-slice level, aggregate to game level for comparison

### 8. Baseline Models

**Module**: `src/baselines/models.py`

**Purpose**: Traditional ML models for benchmarking.

**Key Classes**:

```python
class BaselineModels:
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rolling means/variances, opponent features, role, pace
        pass
    
    def train_ridge(self, X: pd.DataFrame, y: np.ndarray, 
                   alpha: float = 1.0) -> Ridge:
        pass
    
    def train_xgboost(self, X: pd.DataFrame, y: np.ndarray, 
                     params: Dict = None) -> xgb.XGBRegressor:
        # Default: max_depth=6, n_estimators=500, learning_rate=0.05
        pass
    
    def train_mlp(self, X: pd.DataFrame, y: np.ndarray, 
                 layers: List[int] = [128, 64], dropout: float = 0.1) -> MLPRegressor:
        pass
    
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        pass
    
    def save_model(self, model: Any, path: str) -> None:
        pass
    
    def load_model(self, path: str) -> Any:
        pass
```

**Feature Engineering for Baselines**:
- Rolling means (5, 10, 20 games) for all box stats
- Rolling variances for volatility measures
- Opponent defensive rating, pace, scheme frequencies
- Player role indicator
- Days rest, home/away indicator
- Month/day-of-week cyclical encoding

### 9. Calibration

**Module**: `src/calibration/fit.py`

**Purpose**: Calibrate probabilistic predictions and fit copulas.

**Key Classes**:

```python
class Calibrator:
    def compute_pit(self, y_true: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
        # Probability Integral Transform: F(y_true) where F is empirical CDF
        pass
    
    def fit_isotonic(self, stat: str, pits: np.ndarray) -> IsotonicRegression:
        pass
    
    def apply_calibration(self, stat: str, samples: np.ndarray, 
                         model: IsotonicRegression) -> np.ndarray:
        pass
    
    def fit_copula(self, stats_matrix: np.ndarray) -> CopulaModel:
        # Fit Gaussian or vine copula to capture dependencies
        pass
    
    def sample_copula(self, model: CopulaModel, marginals: Dict[str, np.ndarray]) -> np.ndarray:
        pass
```

**Calibration Strategy**:
1. Compute PIT values for each statistic on validation set
2. Fit isotonic regression to map predicted quantiles to calibrated quantiles
3. Apply calibration to new predictions
4. Fit copula to capture multivariate dependencies (e.g., PTS-AST correlation)
5. Sample from copula with calibrated marginals for final predictions

### 10. Benchmarking

**Module**: `src/benchmarks/compare.py`

**Purpose**: Compare models on accuracy, efficiency, and calibration.

**Key Classes**:

```python
class BenchmarkRunner:
    def run_eval_window(self, window_df: pd.DataFrame, models: List[str], 
                       cfg: Dict) -> Dict[str, Any]:
        pass
    
    def compute_accuracy_metrics(self, y_true: pd.DataFrame, 
                                 preds: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        # Returns: {model_name: {stat: {mae, rmse, crps, coverage_50, coverage_80, 
        #                                ece, tail_recall_p95}}}
        pass
    
    def compute_efficiency_metrics(self, model_name: str, inference_fn: Callable, 
                                   *args, **kwargs) -> Dict[str, float]:
        # Returns: {train_time_sec, infer_time_ms_per_player, adaptation_time_ms, memory_mb}
        pass
    
    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        # Side-by-side comparison table
        pass
    
    def ablation_study(self, config_grid: List[Dict]) -> pd.DataFrame:
        # Test different blending weights, state amplitudes, etc.
        pass
```

**Metrics Definitions**:

- **MAE**: Mean Absolute Error = mean(|y_true - y_pred|)
- **RMSE**: Root Mean Squared Error = sqrt(mean((y_true - y_pred)^2))
- **CRPS**: Continuous Ranked Probability Score = integral of (F(x) - 1{y_true <= x})^2
- **Coverage_X**: Fraction of true values within X% prediction interval
- **ECE**: Expected Calibration Error = sum over bins of |predicted_prob - observed_freq| * bin_weight
- **Tail Recall P95**: Fraction of true values > 95th percentile correctly identified
- **Spearman Rank Correlation**: Rank correlation between predicted and true values

**Evaluation Windows**:
- Rolling 30 games: Sliding window of 30 games for temporal validation
- Monthly: Aggregate by calendar month
- Playoffs only: Subset to playoff games for high-stakes evaluation

### 11. Reporting

**Module**: `src/reporting/build.py`

**Purpose**: Generate PDF, JSON, and CSV reports.

**Key Classes**:

```python
class ReportBuilder:
    def build_coach_one_pager(self, game_ctx: GameContext, 
                             players: List[SimulationResult]) -> bytes:
        # Returns PDF bytes
        pass
    
    def build_analyst_detail(self, game_ctx: GameContext, 
                            players: List[SimulationResult],
                            calibration: CalibrationResult) -> bytes:
        pass
    
    def build_benchmark_report(self, tables: Dict[str, pd.DataFrame], 
                              charts: Dict[str, bytes],
                              text: Dict[str, str]) -> bytes:
        pass
    
    def write_json_report(self, game_ctx: GameContext, payload: Dict, path: str) -> None:
        pass
    
    def write_csv_summary(self, players_summary: pd.DataFrame, path: str) -> None:
        pass
```

**Coach One-Pager Layout**:

- Header: Game info (date, opponent, venue)
- Player grid (3x3): Each cell shows player name, projected PTS/REB/AST with 80% intervals
- Risk flags: Icons for high variance, foul risk, matchup disadvantage
- Footer: Model version, confidence badge

**Analyst Detail Layout**:
- Page 1: Full distribution plots for all stats (violin plots)
- Page 2: Capability region visualization (2D projections)
- Page 3: Calibration diagnostics (PIT histograms, reliability diagrams)
- Page 4: Hypervolume metrics, state transition probabilities
- Page 5: Model comparison table (if multiple models run)

**Benchmark Report Layout**:
- Executive summary: TL;DR of best model, key findings
- Table 1: Accuracy metrics (MAE, RMSE, CRPS) by model and stat
- Table 2: Coverage and calibration (coverage_50, coverage_80, ECE)
- Table 3: Efficiency metrics (runtime, memory, adaptation time)
- Charts: Box plots of metric distributions, scatter plots of predicted vs actual
- Statistical tests: Paired t-tests or Wilcoxon tests for metric differences

### 12. API

**Module**: `src/api/server.py`

**Purpose**: REST API for external integrations.

**Endpoints**:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class SimulateRequest(BaseModel):
    game_id: str
    date: str
    team_id: str
    opponent_id: str
    players: List[str]
    toggles: Dict[str, bool] = {}

class SimulateResponse(BaseModel):
    players: List[Dict[str, Any]]
    team_level: Dict[str, Any]
    calibration_badge: Dict[str, Any]

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest):
    # Run global simulation
    pass

@app.post("/simulate-local")
async def simulate_local(request: Dict):
    # Run local models
    pass

@app.post("/benchmark")
async def benchmark(request: Dict):
    # Run benchmark comparison
    pass
```

**Design Decisions**:
- Use FastAPI for automatic OpenAPI documentation
- Pydantic models for request/response validation
- Async endpoints for potential parallelization
- JWT authentication (optional, for production)

### 13. CLI

**Module**: `src/cli/main.py`

**Purpose**: Command-line interface for all operations.

**Commands**:

```python
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--season', required=True, type=int)
@click.option('--strata', default='role')
@click.option('--quantile', default=0.9, type=float)
def build_frontiers(season, strata, quantile):
    """Fit efficiency frontiers for a season."""
    pass

@cli.command()
@click.option('--game-id', required=True)
@click.option('--players', required=True, multiple=True)
@click.option('--N', default=20000, type=int)
@click.option('--seed', type=int)
@click.option('--save-json', is_flag=True)
@click.option('--save-pdf', is_flag=True)
def simulate_global(game_id, players, n, seed, save_json, save_pdf):
    """Run global simulation for a game."""
    pass

@cli.command()
@click.option('--window', required=True)
@click.option('--models', default='all')
@click.option('--save-pdf', is_flag=True)
@click.option('--save-md', is_flag=True)
def benchmark(window, models, save_pdf, save_md):
    """Run benchmark comparison."""
    pass
```

## Data Models

### Core Data Structures

**GameContext**:

```python
@dataclass
class GameContext:
    game_id: str
    date: datetime
    team_id: str
    opponent_id: str
    venue: str  # "home" or "away"
    pace: float
    
@dataclass
class OpponentContext:
    opponent_id: str
    scheme_drop_rate: float
    scheme_switch_rate: float
    scheme_ice_rate: float
    blitz_rate: float
    rim_deterrence_index: float
    def_reb_strength: float
    foul_discipline_index: float
    pace: float
    help_nail_freq: float
    
@dataclass
class PlayerContext:
    player_id: str
    role: str  # "starter", "rotation", "bench"
    exp_minutes: float
    exp_usage: float
    posterior: PosteriorParams
```

### Database Schema (if using SQL)

**players_per_game**:
```sql
CREATE TABLE players_per_game (
    player_id VARCHAR(50),
    game_id VARCHAR(50),
    date DATE,
    team_id VARCHAR(10),
    opponent_id VARCHAR(10),
    minutes FLOAT,
    usage FLOAT,
    ts_pct FLOAT,
    three_pa_rate FLOAT,
    rim_attempt_rate FLOAT,
    mid_attempt_rate FLOAT,
    ast_pct FLOAT,
    tov_pct FLOAT,
    orb_pct FLOAT,
    drb_pct FLOAT,
    stl_pct FLOAT,
    blk_pct FLOAT,
    ft_rate FLOAT,
    pf FLOAT,
    role VARCHAR(20),
    PRIMARY KEY (player_id, game_id)
);
```

**opponent_features**:
```sql
CREATE TABLE opponent_features (
    opponent_id VARCHAR(10),
    date DATE,
    scheme_drop_rate FLOAT,
    scheme_switch_rate FLOAT,
    scheme_ice_rate FLOAT,
    blitz_rate FLOAT,
    rim_deterrence_index FLOAT,
    def_reb_strength FLOAT,
    foul_discipline_index FLOAT,
    pace FLOAT,
    help_nail_freq FLOAT,
    PRIMARY KEY (opponent_id, date)
);
```

## Error Handling

### Error Categories

1. **Data Errors**:
   - Missing files: Return clear error with expected file path
   - Invalid data: Log validation errors, skip invalid rows
   - Missingness > 5%: Raise DataQualityError with details

2. **Computation Errors**:
   - Singular covariance matrix: Add regularization (ridge = 1e-6)
   - Empty capability region: Relax constraints iteratively, log warnings
   - Sampling failure: Retry with different seed, fall back to ellipsoid-only

3. **Model Errors**:
   - Training failure: Log error, skip model, continue with others
   - Prediction out of bounds: Clip to reasonable ranges, log warning
   - Calibration failure: Use uncalibrated predictions, flag in report

4. **API Errors**:
   - Invalid request: Return 400 with validation details
   - Model not found: Return 404 with available models
   - Timeout: Return 504 with partial results if available

### Logging Strategy

```python
import logging
import json

# JSON structured logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler('logs/system.log')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

def log_event(event_type: str, details: Dict):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details
    }
    logger.info(json.dumps(log_entry))
```

**Log Events**:
- data_loaded: File path, rows, columns, validation status
- model_trained: Model type, training time, metrics
- simulation_run: Game ID, players, trials, runtime
- error: Error type, message, stack trace

## Testing Strategy

### Unit Tests

**Test Coverage Targets**: > 80% for all modules

**Key Test Cases**:

1. **Data Loading** (`tests/test_data_loader.py`):
   - Valid CSV loads correctly
   - Missing file raises FileNotFoundError
   - Validation catches missingness > 5%
   - Outlier capping works correctly

2. **Feature Engineering** (`tests/test_features.py`):
   - Rolling windows computed correctly
   - Exponential decay applied properly
   - Posteriors have correct dimensions
   - Scalers transform reversibly

3. **Frontiers** (`tests/test_frontiers.py`):
   - Frontier fits to toy data
   - Linearization produces valid halfspaces
   - Save/load preserves model

4. **Regions** (`tests/test_regions.py`):
   - Ellipsoid construction from posterior
   - Polytope assembly from halfspaces
   - Intersection is non-empty for valid inputs
   - Sampling produces points in region

5. **Simulation** (`tests/test_global_sim.py`):
   - State transitions follow Markov property
   - State offsets applied correctly
   - Box projection sums to reasonable totals
   - Reproducible with same seed

6. **Local Models** (`tests/test_local_*.py`):
   - Featurization produces expected columns
   - Model training converges
   - Predictions in [0, 1] range
   - Aggregation to box stats is consistent

7. **Benchmarks** (`tests/test_benchmarks.py`):
   - Metrics computed correctly on toy data
   - Comparison table has expected structure
   - Efficiency measurement is accurate

### Integration Tests

**Test Scenarios**:

1. **Full Pipeline** (`tests/test_pipeline_pregame.py`):
   - Load data → features → frontiers → regions → simulate → report
   - Verify end-to-end execution
   - Check output files created

2. **Benchmark Pipeline** (`tests/test_pipeline_benchmark.py`):
   - Train all models → predict on eval window → compute metrics → generate report
   - Verify all models complete
   - Check benchmark report structure

### Fixtures

**Toy Data** (`fixtures/toy_game_inputs.json`):

```json
{
  "game_id": "TEST_001",
  "date": "2024-01-15",
  "team_id": "GSW",
  "opponent_id": "LAL",
  "players": [
    {
      "player_id": "curry_stephen",
      "role": "starter",
      "posterior": {
        "mu": [30.0, 5.0, 6.0, 1.5, 0.3, 2.5],
        "Sigma": [[25, 2, 3, 0.5, 0.1, 1], [2, 4, 1, 0.2, 0.05, 0.3], ...]
      }
    }
  ],
  "opponent": {
    "scheme_drop_rate": 0.4,
    "scheme_switch_rate": 0.3,
    "blitz_rate": 0.15,
    "rim_deterrence_index": 1.2,
    "pace": 100.5
  }
}
```

## Performance Optimization

### Computational Bottlenecks

1. **Region Sampling**: Hit-and-run MCMC can be slow for high dimensions
   - **Solution**: Use Numba JIT compilation for inner loops
   - **Alternative**: Pre-compute samples for common region types

2. **Simulation Trials**: 20,000 trials per player can be expensive
   - **Solution**: Parallelize across players using multiprocessing
   - **Target**: < 2 seconds per player

3. **Benchmark Evaluation**: Running all models on large eval windows
   - **Solution**: Cache model predictions, parallelize model evaluation
   - **Target**: < 5 minutes for 30-game window with 6 models

### Memory Management

- **Streaming**: Process games in batches for large evaluations
- **Caching**: Cache frontiers, posteriors, and model artifacts
- **Cleanup**: Delete intermediate samples after aggregation

### Parallelization Strategy

```python
from multiprocessing import Pool
from functools import partial

def simulate_player_parallel(players: List[PlayerContext], 
                             game_ctx: GameContext, 
                             n_workers: int = 4) -> List[SimulationResult]:
    simulate_fn = partial(simulate_single_player, game_ctx=game_ctx)
    with Pool(n_workers) as pool:
        results = pool.map(simulate_fn, players)
    return results
```

## Configuration Management

### Configuration Files

**configs/default.yaml**:
```yaml
geometry:
  credible_alpha: 0.80
  n_region_samples: 2000

simulation:
  trials: 20000
  n_stints: 5
  seed: null

priors:
  minutes_sigma_by_role:
    starter: 3.5
    rotation: 5.0
    bench: 6.5
  usage_beta_by_role:
    starter: [8, 12]
    rotation: [6, 14]
    bench: [4, 16]

blending:
  strategy: weighted
  weights:
    global: 0.6
    local: 0.4

positional:
  enabled: false

logging:
  level: INFO
  format: json
  file: logs/system.log
```

### Environment Variables

- `DATA_DIR`: Path to data directory (default: "Data")
- `ARTIFACTS_DIR`: Path to artifacts (default: "artifacts")
- `LOG_LEVEL`: Logging level (default: "INFO")
- `N_WORKERS`: Number of parallel workers (default: 4)

## Deployment Considerations

### Docker Container

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/

EXPOSE 8000

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Monitoring

- **Health Checks**: `/health` endpoint returns service status
- **Metrics**: Track request latency, error rates, model performance
- **Alerts**: Notify on prediction failures, data quality issues

## Future Extensions

### Positional Tracking Module

**When Enabled** (tracking data available):

1. **Spatial Capability Volume (SCV)**:
   - Replace proxy features with measured spatial overlaps
   - Compute SCV from tracking coordinates
   - Integrate into capability region construction

2. **Play State Simulation**:
   - Model player movements and spacing
   - Simulate defensive rotations
   - Predict shot quality from spatial configuration

3. **Interface Changes**:
   - `src/positional/ingest_tracking.py`: Load tracking data
   - `src/positional/derive_features.py`: Compute spatial features
   - `src/positional/build_spatial_region.py`: Construct SCV
   - `src/local_models/`: Replace proxies with measured features

**Design Principle**: Keep positional module isolated with clear interfaces so it can be enabled without changing core logic.

## Acceptance Criteria Validation

### Accuracy Targets

- **Global Only**: PTS_MAE ≤ 5.0, coverage_80 ∈ [0.78, 0.84]
- **Blended**: PTS_MAE ≤ 4.6, coverage_80 ∈ [0.78, 0.86], tail_recall_p95 ≥ 0.65

**Validation Strategy**:
- Evaluate on held-out test set (last 30 games of season)
- Report metrics with 95% confidence intervals
- Compare against baseline models

### Efficiency Targets

- **Global Only**: ≤ 2.0 sec per player
- **Blended**: ≤ 2.5 sec per player
- **Baseline ML**: ≤ 20 ms per player
- **Adaptation**: ≤ 50 ms on scheme toggle

**Validation Strategy**:
- Benchmark on standard hardware (4-core CPU, 16GB RAM)
- Average over 100 runs with different seeds
- Profile to identify bottlenecks

## Summary

This design provides a comprehensive framework for NBA player performance prediction combining geometric capability regions, local event models, and traditional ML baselines. The modular architecture allows independent development and testing of components while maintaining clear interfaces for integration. The system is designed for both research (comprehensive benchmarking) and production (fast inference, API access) use cases.
