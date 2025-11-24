# Positional Tracking Module

## Status: Scaffolded (Disabled)

This module is currently scaffolded with placeholder implementations and is **disabled by default**.

## Configuration

```yaml
# In configs/default.yaml
positional:
  enabled: false  # Set to true when tracking data is available
  tracking_data_dir: null
  spatial_features:
    - scv_volume
    - overlap_index
    - spacing_entropy
```

## Module Structure

```
src/positional/
├── __init__.py                    # Module initialization and utilities
├── ingest_tracking.py             # Load and validate tracking data
├── derive_features.py             # Compute spatial features (SCV, overlap, etc.)
├── build_spatial_region.py        # Construct spatial capability regions
├── simulate_play_states.py        # Simulate player movements and play states
├── INTEGRATION_GUIDE.md           # Detailed integration documentation
└── README.md                      # This file
```

## Purpose

When player tracking data becomes available, this module will:

1. **Replace proxy features** with measured spatial features in local models
2. **Enhance capability regions** with spatial constraints from tracking data
3. **Improve prediction accuracy** by incorporating spatial dynamics

## Key Features

### Spatial Features
- **SCV (Spatial Capability Volume)**: Court space a player can control
- **Overlap Index**: Defender pressure on shooter
- **Spacing Entropy**: Team spacing quality
- **Time to Ball**: Measured time to reach ball position
- **Seal Angle**: Rebounding position quality
- **Reach Margin**: Reach advantage over defenders

### Play State Simulation
- Offensive play types (pick-and-roll, isolation, etc.)
- Defensive rotations (help, switch, drop, etc.)
- Shot quality prediction from spatial configuration
- Lane risk and help defense probability

## Usage

### When Disabled (Current State)

The module is inactive and proxy features are used:

```python
from src.positional import is_enabled

config = load_config()
if is_enabled(config):
    # Use tracking data (not yet implemented)
    pass
else:
    # Use proxy features (current behavior)
    pass
```

### When Enabled (Future)

After implementing the placeholder functions:

```python
from src.positional import ingest_tracking, derive_features

# Load tracking data
tracking_df = ingest_tracking.load_tracking_data(game_id="GSW_LAL_20240115")

# Compute spatial features
features = derive_features.compute_spatial_features(
    tracking_df, 
    player_id="curry_stephen",
    event_type="shot"
)

# Features replace proxies in local models
print(features[['scv_volume', 'overlap_index', 'time_to_ball']])
```

## Integration Points

### Local Models

The module replaces proxy features in:

- **Rebound Model**: `time_to_ball`, `seal_angle`, `reach_margin`, `crowd_index`
- **Assist Model**: `shot_quality`, `help_probability`, `lane_risk`
- **Shot Model**: `overlap_index` (defender pressure)

### Capability Regions

Spatial constraints are added to the polytope in `src/regions/build.py`:

```python
if config['positional']['enabled']:
    spatial_constraints = extract_spatial_constraints(tracking_df, player_id)
    halfspaces.extend(spatial_constraints)
```

## Implementation Checklist

When tracking data becomes available:

- [ ] Implement `load_tracking_data()` in `ingest_tracking.py`
- [ ] Implement `validate_tracking_data()` for data quality checks
- [ ] Implement spatial feature computation functions in `derive_features.py`
- [ ] Implement spatial region construction in `build_spatial_region.py`
- [ ] Implement play state simulation in `simulate_play_states.py`
- [ ] Update local models to use measured features when enabled
- [ ] Update capability region construction to include spatial constraints
- [ ] Add unit tests for all implemented functions
- [ ] Add integration tests for end-to-end pipeline
- [ ] Benchmark performance impact
- [ ] Update configuration with tracking data path

## Documentation

See **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** for:
- Detailed implementation instructions
- Proxy feature replacement map
- Data format specifications
- Testing strategy
- Performance considerations

## Requirements

This module implements requirements from the specification:

- **Requirement 19.1**: Positional tracking module with `enabled=false` configuration
- **Requirement 19.2**: Placeholder functions for tracking data processing
- **Requirement 19.3**: Interface documentation for replacing proxy features
- **Requirement 19.4**: Module does not execute when `enabled=false`
- **Requirement 19.5**: SCV and overlap metrics derivation (when enabled)

## Notes

- All functions raise `NotImplementedError` until tracking data is available
- The module is designed to be non-intrusive - existing functionality works without it
- Configuration flag (`positional.enabled`) controls whether module is active
- Clear interfaces ensure smooth integration when tracking data becomes available
