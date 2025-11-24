# Test Fixtures

This directory contains test fixtures for the NBA Player Performance Prediction System.

## Files

### toy_game_inputs.json

Sample game context with player data for testing simulation and prediction workflows.

**Structure:**
- `game_id`: Unique game identifier
- `date`: Game date
- `team_id`: Team identifier (e.g., "GSW")
- `opponent_id`: Opponent team identifier (e.g., "LAL")
- `venue`: Home or away
- `pace`: Game pace
- `players`: Array of player objects, each containing:
  - `player_id`: Unique player identifier
  - `role`: Player role (starter, rotation, bench)
  - `exp_minutes`: Expected minutes
  - `exp_usage`: Expected usage rate
  - `posterior`: Posterior distribution parameters
    - `mu`: Mean vector (7 dimensions for core attributes)
    - `Sigma`: Covariance matrix (7x7)
    - `feature_names`: Names of the 7 core attributes (TS%, USG%, AST%, TOV%, TRB%, STL%, BLK%)
- `opponent`: Opponent context with defensive scheme parameters

**Usage:**
```python
import json

with open("fixtures/toy_game_inputs.json", 'r') as f:
    game_context = json.load(f)

# Access player posteriors
for player in game_context["players"]:
    mu = player["posterior"]["mu"]
    Sigma = player["posterior"]["Sigma"]
    # Use for capability region construction
```

### small_eval_window.parquet

Sample evaluation window with synthetic player game data for benchmarking tests.

**Structure:**
- 50 rows (5 players × 10 games each)
- Columns include:
  - Identifiers: `player_id`, `game_id`, `Date`, `team_id`, `opponent_id`
  - Box stats: `PTS`, `TRB`, `AST`, `STL`, `BLK`, `TOV`, `MP`
  - Shooting: `FG%`, `3P%`, `FT%`, `TS%`
  - Advanced: `USG%`, `AST%`, `TOV%`, `TRB%`, `STL%`, `BLK%`
  - Ratings: `ORTG`, `DRTG`, `BPM`, `GmSc`
  - Context: `role`, opponent features

**Usage:**
```python
import pandas as pd

df = pd.read_parquet("fixtures/small_eval_window.parquet")

# Use for benchmark testing
ground_truth = df[['player_id', 'game_id', 'PTS', 'TRB', 'AST']]
```

## Purpose

These fixtures are used for:

1. **Unit Testing**: Verify individual components work correctly with realistic data structures
2. **Integration Testing**: Test end-to-end pipelines without requiring full datasets
3. **Benchmarking**: Compare model performance on a small, reproducible dataset
4. **Development**: Quick iteration during feature development

## Maintenance

- Fixtures should remain small (< 1 MB) for fast test execution
- Data should be synthetic but realistic
- Update fixtures when data schemas change
- Document any changes to fixture structure in this README
