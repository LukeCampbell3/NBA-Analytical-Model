# Utils Module

This module contains utility functions and classes for the NBA prediction system.

## data_loader.py

The `DataLoader` class provides functionality for loading and validating player game statistics from CSV files.

### Key Features

- **Load Player Data**: Read CSV files from the Data directory structure
- **Data Validation**: Check for missing values, invalid data, and data quality issues
- **Outlier Capping**: Apply quantile-based capping to prevent extreme values from distorting models
- **Leakage Control**: Enforce temporal ordering to prevent future information from leaking into training

### Usage Example

```python
from src.utils.data_loader import DataLoader
from datetime import datetime

# Initialize loader
loader = DataLoader(data_dir="Data")

# Load player data
df = loader.load_player_data("Stephen_Curry", 2024)

# Validate data quality
result = loader.validate_data(df)
if result.is_valid:
    print("Data is valid!")
else:
    print(f"Validation errors: {result.errors}")

# Apply outlier capping
df_capped = loader.apply_outlier_caps(df, role="starter", season=2024)

# Enforce temporal ordering (prevent data leakage)
forecast_date = datetime(2024, 3, 1)
df_train = loader.enforce_leakage_control(df_capped, forecast_date, strict=True)

# Load multiple players
players = ["Stephen_Curry", "LeBron_James"]
data_dict = loader.load_multiple_players(players, 2024, validate=True)
```

### Validation Checks

The `validate_data()` method performs the following checks:

1. **Required Columns**: Ensures all required columns are present
2. **Missingness**: Checks that missing values don't exceed 5% threshold
3. **Date Validity**: Validates date format and values
4. **Negative Values**: Warns about negative values in counting stats
5. **Percentage Ranges**: Warns about percentage values outside [0, 1]

### Outlier Capping

The `apply_outlier_caps()` method:

- Computes quantiles (default: 1st and 99th percentile) from stratified data
- Stratifies by role and season when available
- Applies caps to all numeric columns
- Preserves data distribution while removing extreme outliers

### Leakage Control

The `enforce_leakage_control()` method:

- Filters data to only include games before the forecast date
- Supports strict mode (exclude forecast date) and non-strict mode (include forecast date)
- Ensures temporal ordering by sorting by date
- Raises errors if Date column is missing or invalid

## errors.py

Custom exception classes for clear error handling:

- `DataQualityError`: Data quality checks failed
- `DataLeakageError`: Temporal ordering violated
- `ModelTrainingError`: Model training failed
- `RegionConstructionError`: Capability region construction failed
- `SimulationError`: Simulation failed
- `CalibrationError`: Calibration failed
