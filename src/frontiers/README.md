# Frontier Fitting Module

This module implements efficiency frontier fitting for NBA player performance prediction. Frontiers define trade-off boundaries between performance attributes using quantile regression.

## Overview

The frontier fitting module provides:

1. **FrontierModel**: Data class representing a fitted efficiency frontier
2. **Halfspace**: Geometric constraint representation (a^T x ≤ b)
3. **FrontierFitter**: Main class for fitting, linearizing, and persisting frontiers

## Usage

### Basic Frontier Fitting

```python
import pandas as pd
from src.frontiers import FrontierFitter

# Load player performance data
data = pd.DataFrame({
    'usage': [...],  # Usage rate
    'efficiency': [...],  # True shooting percentage
    'role': [...]  # Player role (starter, rotation, bench)
})

# Create fitter
fitter = FrontierFitter(min_samples=30)

# Fit frontier for starters at 90th percentile
model = fitter.fit_frontier(
    data=data,
    x='usage',
    y='efficiency',
    strata={'role': 'starter'},
    quantile=0.9
)

print(f"Fitted coefficients: {model.coefficients}")
print(f"X range: {model.x_range}")
print(f"Y range: {model.y_range}")
```

### Linearizing Frontiers

Convert the fitted frontier curve into halfspace constraints:

```python
# Generate 10 linear segments
halfspaces = fitter.linearize_frontier(model, n_segments=10)

# Each halfspace represents: a^T x ≤ b
for hs in halfspaces:
    print(f"Normal: {hs.normal}, Offset: {hs.offset}")
```

### Saving and Loading Models

```python
# Save frontier model
fitter.save_frontier(model, 'artifacts/frontiers/starter_usage_efficiency.pkl')

# Load frontier model
loaded_model = fitter.load_frontier('artifacts/frontiers/starter_usage_efficiency.pkl')
```

## Key Classes

### FrontierModel

Represents a fitted efficiency frontier with the following attributes:

- `x_attr`: Name of x-axis attribute
- `y_attr`: Name of y-axis attribute
- `strata`: Stratification parameters (e.g., {'role': 'starter'})
- `quantile`: Quantile level used for fitting (e.g., 0.9)
- `coefficients`: Fitted quantile regression coefficients [intercept, slope]
- `x_range`: Valid range for x attribute (min, max)
- `y_range`: Valid range for y attribute (min, max)

### Halfspace

Represents a linear constraint: a^T x ≤ b

- `normal`: Normal vector (a)
- `offset`: Offset scalar (b)

### FrontierFitter

Main class for frontier operations:

- `fit_frontier()`: Fit frontier using quantile regression
- `linearize_frontier()`: Convert to halfspace representation
- `save_frontier()`: Persist model to disk
- `load_frontier()`: Load model from disk

## Design Decisions

1. **Quantile Regression**: Uses 90th percentile by default to capture efficient frontier
2. **Stratification**: Frontiers are stratified by role and opponent scheme for context-specific constraints
3. **Linearization**: Piecewise linear approximation for polytope representation
4. **Boundary Constraints**: Automatically adds x and y bounds to ensure feasible regions

## Requirements Satisfied

This implementation satisfies requirements 3.1-3.5:

- 3.1: Fit frontier models stratified by role and opponent scheme
- 3.2: Use 90th percentile quantile for frontier estimation
- 3.3: Linearize fitted frontiers into halfspace representations
- 3.4: Save frontier models to disk
- 3.5: Load previously saved frontier models

## Testing

Run tests with:

```bash
# If pytest is installed
pytest tests/test_frontiers.py -v

# Or run manual test script
python test_frontier_manual.py
```

## Dependencies

- numpy: Array operations
- pandas: Data manipulation
- statsmodels: Quantile regression
- joblib: Model serialization
- scipy: Interpolation (optional)
