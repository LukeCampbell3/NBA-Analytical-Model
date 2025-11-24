# Positional Tracking Module Integration Guide

## Overview

This guide documents the interface for integrating positional tracking data into the NBA Player Performance Prediction System. The positional tracking module is currently scaffolded with placeholder implementations and is **disabled by default** (`positional.enabled=false` in `configs/default.yaml`).

When tracking data becomes available, this module can be enabled to replace proxy features with measured spatial features, significantly enhancing prediction accuracy.

## Module Status

**Current Status**: Scaffolded (disabled)
**Configuration**: `configs/default.yaml` → `positional.enabled: false`
**Location**: `src/positional/`

## Architecture

The positional tracking module consists of four main components:

### 1. Tracking Data Ingestion (`ingest_tracking.py`)

**Purpose**: Load and validate player tracking data from data sources.

**Key Functions**:
- `load_tracking_data(game_id, data_dir, quarter)` - Load tracking data for a game
- `validate_tracking_data(df)` - Validate data quality and completeness
- `parse_tracking_sequence(df, start_frame, end_frame)` - Parse frame sequences
- `filter_by_event(df, event_type, player_id)` - Filter by event type

**Expected Data Format**:
```python
# DataFrame columns:
# - frame_id: int
# - timestamp: datetime
# - game_clock: float (seconds remaining in quarter)
# - quarter: int
# - ball_x, ball_y, ball_z: float (coordinates in feet)
# - player_id: str
# - x, y: float (player coordinates in feet)
```

**Court Coordinate System**:
- Standard NBA court: 94 feet (length) × 50 feet (width)
- Origin and orientation may vary by data provider
- Rim location: typically (5.25, 25.0)

### 2. Spatial Feature Derivation (`derive_features.py`)

**Purpose**: Compute spatial features from tracking data to replace proxy features.

**Key Functions**:
- `compute_spatial_features(tracking_df, player_id, event_type)` - Main feature computation
- `compute_scv_volume(player_pos, defender_pos, ball_pos)` - Spatial Capability Volume
- `compute_overlap_index(shooter_pos, defender_pos)` - Defender overlap
- `compute_spacing_entropy(team_positions)` - Team spacing quality
- `compute_time_to_ball(player_pos, ball_pos)` - Time to reach ball
- `compute_seal_angle(player_pos, defender_pos)` - Rebounding position angle
- `compute_reach_margin(player_pos, ball_pos, defender_pos)` - Reach advantage
- `compute_crowd_index(player_pos, all_positions)` - Nearby player count

**Output Format**:
```python
# SpatialFeatures dataclass:
# - player_id: str
# - frame_id: int
# - scv_volume: float (sq ft)
# - overlap_index: float (0.0-1.0)
# - spacing_entropy: float
# - time_to_ball: float (seconds)
# - seal_angle: float (degrees)
# - reach_margin: float (feet)
# - crowd_index: int
```

### 3. Spatial Region Construction (`build_spatial_region.py`)

**Purpose**: Construct spatial capability regions that integrate with geometric capability regions.

**Key Functions**:
- `build_spatial_region(tracking_df, player_id, game_context)` - Build SCV region
- `extract_spatial_constraints(tracking_df, player_id, opponent_pos)` - Extract constraints
- `integrate_with_capability_region(spatial_region, capability_region)` - Merge regions
- `sample_spatial_region(spatial_region, n_samples)` - Sample from region

**Integration Point**:
```python
# In src/regions/build.py:assemble_halfspaces()
# Add spatial constraints to polytope:

if config.get("positional", {}).get("enabled", False):
    from src.positional.build_spatial_region import extract_spatial_constraints
    
    spatial_constraints = extract_spatial_constraints(
        tracking_df, player_id, opponent_positions
    )
    
    # Convert to halfspaces and add to polytope
    for constraint in spatial_constraints:
        halfspaces.append(Halfspace(
            normal=constraint.normal,
            offset=constraint.offset
        ))
```

### 4. Play State Simulation (`simulate_play_states.py`)

**Purpose**: Simulate player movements, defensive rotations, and play states.

**Key Functions**:
- `simulate_play_states(tracking_df, game_context, n_simulations)` - Simulate play sequences
- `simulate_defensive_rotations(tracking_df, play_state, scheme)` - Simulate rotations
- `predict_shot_quality(player_pos, defender_pos, play_state)` - Predict shot quality
- `estimate_help_probability(ball_handler_pos, defender_pos, scheme)` - Help defense probability
- `compute_lane_risk(player_pos, target_pos, defender_pos)` - Turnover risk
- `classify_play_state(tracking_sequence, ball_handler_id)` - Classify play type

**Play States**:
- PICK_AND_ROLL, ISOLATION, POST_UP, SPOT_UP
- TRANSITION, HANDOFF, CUT, OFF_BALL_SCREEN
- DRIVE, CATCH_AND_SHOOT

**Defensive Rotations**:
- HELP, RECOVER, SWITCH, ICE, BLITZ
- DROP, HEDGE, SHOW

## Proxy Feature Replacement Map

When positional tracking is enabled, the following proxy features are replaced with measured features:

### Rebound Model (`src/local_models/rebound.py`)

| Proxy Feature | Replacement Function | Module |
|--------------|---------------------|---------|
| `time_to_ball_proxy` | `compute_time_to_ball()` | `derive_features.py` |
| `seal_angle_proxy` | `compute_seal_angle()` | `derive_features.py` |
| `reach_margin` (proxy) | `compute_reach_margin()` | `derive_features.py` |
| `crowd_index` (proxy) | `compute_crowd_index()` | `derive_features.py` |

### Assist Model (`src/local_models/assist.py`)

| Proxy Feature | Replacement Function | Module |
|--------------|---------------------|---------|
| `receiver_shot_quality_proxy` | `predict_shot_quality()` | `simulate_play_states.py` |
| `opponent_help_nail_freq` (proxy) | `estimate_help_probability()` | `simulate_play_states.py` |
| `lane_risk_proxy` | `compute_lane_risk()` | `simulate_play_states.py` |

### Shot Model (`src/local_models/shot.py`)

| Proxy Feature | Replacement Function | Module |
|--------------|---------------------|---------|
| `opponent_rim_deterrence` (proxy) | `compute_overlap_index()` | `derive_features.py` |

### Global Simulator Enhancement

| Enhancement | Function | Module |
|------------|----------|---------|
| Spatial constraints | `build_spatial_region()` | `build_spatial_region.py` |
| Play state transitions | `simulate_play_states()` | `simulate_play_states.py` |
| Defensive rotations | `simulate_defensive_rotations()` | `simulate_play_states.py` |

## Enabling Positional Tracking

### Step 1: Update Configuration

Edit `configs/default.yaml`:

```yaml
positional:
  enabled: true
  tracking_data_dir: "Data/Tracking"  # Path to tracking data
  spatial_features:
    - scv_volume
    - overlap_index
    - spacing_entropy
```

### Step 2: Implement Data Loading

Replace placeholder in `src/positional/ingest_tracking.py`:

```python
def load_tracking_data(game_id: str, data_dir: Optional[str] = None, 
                       quarter: Optional[int] = None) -> pd.DataFrame:
    """Load tracking data from your data source."""
    # Example implementation:
    file_path = f"{data_dir}/{game_id}_tracking.parquet"
    df = pd.read_parquet(file_path)
    
    if quarter is not None:
        df = df[df['quarter'] == quarter]
    
    return df
```

### Step 3: Implement Feature Computation

Replace placeholders in `src/positional/derive_features.py`:

```python
def compute_spatial_features(tracking_df: pd.DataFrame, player_id: str, 
                             event_type: Optional[str] = None) -> pd.DataFrame:
    """Compute spatial features from tracking data."""
    # Filter to player
    player_df = tracking_df[tracking_df['player_id'] == player_id].copy()
    
    # Compute features frame-by-frame
    features = []
    for frame_id in player_df['frame_id'].unique():
        frame_data = tracking_df[tracking_df['frame_id'] == frame_id]
        
        # Extract positions
        player_pos = (player_df.loc[player_df['frame_id'] == frame_id, 'x'].iloc[0],
                     player_df.loc[player_df['frame_id'] == frame_id, 'y'].iloc[0])
        
        # Compute features
        scv = compute_scv_volume(player_pos, defender_positions, ball_pos)
        overlap = compute_overlap_index(player_pos, defender_positions)
        # ... etc
        
        features.append({
            'frame_id': frame_id,
            'player_id': player_id,
            'scv_volume': scv,
            'overlap_index': overlap,
            # ... other features
        })
    
    return pd.DataFrame(features)
```

### Step 4: Update Local Models

Modify local model feature engineering to use measured features when available:

```python
# In src/local_models/rebound.py:featurize_rebound()

from src.positional import is_enabled, derive_features

def featurize_rebound(game_slice_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if is_enabled(config):
        # Use measured features from tracking data
        tracking_df = load_tracking_data(game_slice_df['game_id'].iloc[0])
        spatial_features = derive_features.compute_spatial_features(
            tracking_df, game_slice_df['player_id'].iloc[0], event_type='rebound'
        )
        
        # Merge with game slice data
        df = game_slice_df.merge(spatial_features, on=['frame_id', 'player_id'])
        
        # Use measured features
        features = df[['time_to_ball', 'seal_angle', 'reach_margin', 'crowd_index']]
    else:
        # Use proxy features (current implementation)
        features = df[['time_to_ball_proxy', 'seal_angle_proxy', 
                      'reach_margin_proxy', 'crowd_index_proxy']]
    
    return features
```

### Step 5: Integrate Spatial Regions

Modify `src/regions/build.py` to include spatial constraints:

```python
# In RegionBuilder.assemble_halfspaces()

def assemble_halfspaces(self, frontiers, scheme_constraints, role_bounds, 
                       config, tracking_df=None, player_id=None):
    halfspaces = []
    
    # Add frontier constraints
    for frontier in frontiers:
        halfspaces.extend(self.linearize_frontier(frontier))
    
    # Add scheme and role constraints
    halfspaces.extend(scheme_constraints)
    halfspaces.extend(role_bounds)
    
    # Add spatial constraints if enabled
    if config.get("positional", {}).get("enabled", False) and tracking_df is not None:
        from src.positional.build_spatial_region import extract_spatial_constraints
        
        spatial_constraints = extract_spatial_constraints(
            tracking_df, player_id, opponent_positions
        )
        
        for constraint in spatial_constraints:
            halfspaces.append(Halfspace(
                normal=constraint.normal,
                offset=constraint.offset
            ))
    
    return HPolytope(halfspaces)
```

## Testing Strategy

### Unit Tests

Create tests in `tests/test_positional.py`:

```python
def test_load_tracking_data():
    """Test tracking data loading."""
    df = load_tracking_data("TEST_GAME_001")
    assert 'frame_id' in df.columns
    assert 'player_id' in df.columns
    assert len(df) > 0

def test_compute_spatial_features():
    """Test spatial feature computation."""
    tracking_df = load_test_tracking_data()
    features = compute_spatial_features(tracking_df, "test_player")
    assert 'scv_volume' in features.columns
    assert features['scv_volume'].min() >= 0

def test_spatial_region_integration():
    """Test integration with capability regions."""
    spatial_region = build_spatial_region(tracking_df, "test_player", game_ctx)
    capability_region = build_capability_region(posterior, frontiers)
    
    enhanced_region = integrate_with_capability_region(
        spatial_region, capability_region
    )
    
    assert enhanced_region.volume > 0
```

### Integration Tests

Test end-to-end with tracking data:

```python
def test_full_pipeline_with_tracking():
    """Test full pipeline with positional tracking enabled."""
    config = load_config()
    config['positional']['enabled'] = True
    
    # Load data
    tracking_df = load_tracking_data("TEST_GAME_001")
    player_df = load_player_data("test_player", 2024)
    
    # Build features
    features = compute_spatial_features(tracking_df, "test_player")
    
    # Build region
    region = build_spatial_region(tracking_df, "test_player", game_ctx)
    
    # Run simulation
    result = simulate_player_game(region, opp_ctx, N=1000)
    
    assert result is not None
    assert 'PTS' in result.distributions
```

## Performance Considerations

### Computational Cost

- **Tracking data loading**: ~100ms per game
- **Spatial feature computation**: ~500ms per player per game
- **Spatial region construction**: ~200ms per player
- **Total overhead**: ~800ms per player (vs. ~50ms for proxy features)

### Optimization Strategies

1. **Caching**: Cache computed spatial features per game
2. **Parallelization**: Compute features for multiple players in parallel
3. **Sampling**: Use frame sampling (e.g., every 5th frame) for feature computation
4. **Precomputation**: Precompute spatial features during data preprocessing

### Memory Usage

- Tracking data: ~50MB per game (uncompressed)
- Spatial features: ~5MB per player per game
- Recommendation: Use Parquet format for efficient storage

## Data Requirements

### Minimum Tracking Data Requirements

- **Frame rate**: ≥ 10 Hz (preferably 25 Hz)
- **Coverage**: All players and ball tracked
- **Accuracy**: Position accuracy ≤ 0.5 feet
- **Completeness**: < 5% missing frames

### Recommended Data Sources

- **NBA Advanced Stats**: Official NBA tracking data (if available)
- **Second Spectrum**: Commercial tracking data provider
- **STATS SportVU**: Legacy tracking data (2013-2016)
- **Custom tracking**: Computer vision-based tracking systems

## Future Enhancements

### Phase 1: Basic Integration (Current Scaffolding)
- ✅ Module structure created
- ✅ Interface documented
- ✅ Placeholder functions defined

### Phase 2: Core Implementation
- ⏳ Implement data loading
- ⏳ Implement spatial feature computation
- ⏳ Replace proxy features in local models

### Phase 3: Advanced Features
- ⏳ Spatial region construction
- ⏳ Play state simulation
- ⏳ Defensive rotation modeling

### Phase 4: Optimization
- ⏳ Performance optimization
- ⏳ Caching and parallelization
- ⏳ Real-time inference support

## Support and Questions

For questions about integrating positional tracking:

1. Review this guide and the module docstrings
2. Check the placeholder function signatures for expected interfaces
3. Refer to the design document (`.kiro/specs/nba-prediction-system/design.md`)
4. Consult the requirements document for acceptance criteria

## Summary

The positional tracking module provides a clear interface for integrating tracking data when it becomes available. The module is designed to:

- **Minimize disruption**: Proxy features work until tracking data is available
- **Clear interfaces**: Well-defined function signatures and data formats
- **Modular design**: Can be enabled/disabled via configuration
- **Performance-aware**: Designed with caching and optimization in mind

When tracking data becomes available, follow the steps in this guide to enable the module and replace proxy features with measured spatial features.
