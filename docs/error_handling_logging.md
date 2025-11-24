# Error Handling and Logging Implementation

## Overview

This document describes the comprehensive error handling and logging system implemented for the NBA Player Performance Prediction System.

## Components Implemented

### 1. Custom Exception Classes (`src/utils/errors.py`)

A hierarchy of custom exceptions for different error scenarios:

#### Base Exception
- `NBASystemError`: Base exception with message and details dictionary

#### Data Errors
- `DataError`: Base for data-related errors
- `DataQualityError`: Data validation failures (missingness, invalid values)
- `DataLeakageError`: Temporal ordering violations
- `DataNotFoundError`: Missing data files

#### Model Errors
- `ModelError`: Base for model-related errors
- `ModelTrainingError`: Model training failures
- `ModelNotFoundError`: Missing model files

#### Region Errors
- `RegionError`: Base for capability region errors
- `RegionConstructionError`: Region construction failures
- `SingularMatrixError`: Singular or non-positive definite matrices
- `EmptyRegionError`: Empty regions or sampling failures

#### Other Errors
- `SimulationError`: Simulation failures
- `CalibrationError`: Calibration failures
- `ConfigurationError`: Invalid configuration
- `ValidationError`: Input validation failures

### 2. Structured Logging (`src/utils/logger.py`)

JSON-formatted logging system with:

#### JSONFormatter
- Outputs logs in JSON format for easy parsing
- Includes timestamp, level, logger name, message, context, and exception details
- Handles exception information gracefully

#### StructuredLogger
- Wrapper around Python's logging module
- Methods: `debug()`, `info()`, `warning()`, `error()`, `critical()`
- Special methods:
  - `log_event()`: Log structured events
  - `log_operation_start()`: Log operation start
  - `log_operation_complete()`: Log operation completion with duration
  - `log_operation_failed()`: Log operation failures

#### Features
- Context data support for rich logging
- File and console output
- Configurable log levels
- Automatic log directory creation

### 3. Error Handling in DataLoader (`src/utils/data_loader.py`)

Enhanced with comprehensive error handling:

- **File Not Found**: Raises `DataNotFoundError` with file path details
- **Data Validation**: Raises `DataQualityError` with validation errors and missingness report
- **Leakage Control**: Raises `DataLeakageError` with date information
- **Logging**: All operations logged with context (file paths, row counts, durations)

### 4. Error Handling in RegionBuilder (`src/regions/build.py`)

Enhanced with:

- **Singular Matrix**: Raises `SingularMatrixError` with condition number
- **Dimension Mismatch**: Raises `ValidationError` with shape information
- **Empty Region**: Raises `EmptyRegionError` with attempt count and constraints
- **Logging**: Region construction operations logged with dimensions and status

### 5. Error Handling in API (`src/api/server.py`)

Enhanced with:

- **HTTP Status Codes**:
  - 400 Bad Request: Validation errors
  - 404 Not Found: Missing resources
  - 422 Unprocessable Entity: Region construction failures
  - 500 Internal Server Error: System errors
- **Structured Error Responses**: Include error type and details
- **Operation Logging**: All API operations logged with timing and context

## Usage Examples

### Raising Custom Exceptions

```python
from src.utils.errors import DataQualityError

# Raise with validation details
raise DataQualityError(
    "Data validation failed",
    validation_errors=["Missing column: PTS"],
    missingness_report={"AST": 0.08}
)
```

### Using Structured Logger

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Log with context
logger.info(
    "Data loaded successfully",
    context={
        "player_name": "Stephen_Curry",
        "rows": 82,
        "year": 2024
    }
)

# Log operation lifecycle
logger.log_operation_start("train_model", details={"model_type": "ridge"})
# ... do work ...
logger.log_operation_complete("train_model", duration_sec=12.5)

# Log errors
try:
    # ... code ...
except Exception as e:
    logger.log_operation_failed("train_model", error=e)
```

### Handling Errors in API

```python
from fastapi import HTTPException, status
from src.utils.errors import ValidationError, NBASystemError

try:
    # ... API logic ...
except ValidationError as e:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid input: {str(e)}"
    )
except NBASystemError as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"System error: {str(e)}"
    )
```

## Log Format

Logs are output in JSON format:

```json
{
  "timestamp": "2024-11-12T03:30:49.424806Z",
  "level": "INFO",
  "logger": "src.utils.data_loader",
  "message": "Data loaded successfully",
  "context": {
    "player_name": "Stephen_Curry",
    "rows": 82,
    "year": 2024,
    "duration_sec": 0.15
  }
}
```

Error logs include exception details:

```json
{
  "timestamp": "2024-11-12T03:30:49.424806Z",
  "level": "ERROR",
  "logger": "src.regions.build",
  "message": "Region construction failed",
  "context": {
    "operation": "credible_ellipsoid",
    "dimension": 6
  },
  "exception": {
    "type": "SingularMatrixError",
    "message": "Covariance matrix is not positive definite",
    "traceback": ["..."]
  }
}
```

## Testing

Comprehensive tests in `tests/test_error_handling.py`:

- Custom exception creation and details
- JSON log formatting
- Structured logging with context
- DataLoader error scenarios
- RegionBuilder error scenarios
- All 14 tests passing

## Benefits

1. **Clear Error Messages**: Custom exceptions provide detailed context
2. **Structured Logging**: JSON format enables easy log aggregation and analysis
3. **Debugging**: Rich context in logs helps diagnose issues quickly
4. **Monitoring**: Structured logs can be easily parsed by monitoring tools
5. **API Error Handling**: Appropriate HTTP status codes and error messages
6. **Maintainability**: Consistent error handling patterns across the codebase

## Configuration

Logging can be configured via `configs/default.yaml`:

```yaml
logging:
  level: INFO
  format: json
  file: logs/system.log
```

Or programmatically:

```python
from src.utils.logger import setup_logging

setup_logging(
    level="DEBUG",
    log_file="logs/debug.log",
    console_output=True
)
```

## Future Enhancements

- Add log rotation for production deployments
- Integrate with centralized logging systems (e.g., ELK stack)
- Add performance metrics logging
- Implement log sampling for high-volume operations
- Add structured error reporting dashboard
