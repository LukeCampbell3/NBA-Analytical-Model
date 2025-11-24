# NBA Prediction API

REST API for probabilistic NBA player performance forecasting.

## Overview

The API provides endpoints for:
- **Health checks**: Service status monitoring
- **Global simulation**: Capability-region based performance prediction
- **Local model blending**: Combining global and local predictions
- **Model benchmarking**: Comparing prediction models

## Quick Start

### Starting the Server

```bash
# Development mode with auto-reload
uvicorn src.api.server:app --reload

# Production mode
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}
```

### POST /simulate

Run global simulation for player performance prediction.

**Request:**
```json
{
  "game_id": "TEST_001",
  "date": "2024-01-15",
  "team_id": "GSW",
  "opponent_id": "LAL",
  "venue": "home",
  "pace": 100.5,
  "opponent_context": {
    "opponent_id": "LAL",
    "scheme_drop_rate": 0.4,
    "scheme_switch_rate": 0.3,
    "scheme_ice_rate": 0.15,
    "blitz_rate": 0.15,
    "rim_deterrence_index": 1.2,
    "def_reb_strength": 1.1,
    "foul_discipline_index": 0.95,
    "pace": 100.5,
    "help_nail_freq": 0.25
  },
  "players": [
    {
      "player_id": "curry_stephen",
      "role": "starter",
      "exp_minutes": 34.0,
      "exp_usage": 0.30,
      "posterior_mu": [0.62, 0.30, 0.35, 0.12, 0.08, 0.02, 0.01],
      "posterior_sigma": [[...]]
    }
  ],
  "n_trials": 20000,
  "seed": 42
}
```

**Response:**
```json
{
  "game_id": "TEST_001",
  "players": [
    {
      "player_id": "curry_stephen",
      "distributions": {
        "PTS": {
          "mean": 28.5,
          "median": 28.0,
          "std": 6.2,
          "p10": 20.0,
          "p90": 37.0
        },
        ...
      },
      "risk_metrics": {
        "PTS_VaR_05": 18.5,
        "PTS_VaR_95": 38.2,
        ...
      },
      "hypervolume_index": 0.85,
      "metadata": {...}
    }
  ],
  "team_level": {
    "team_PTS_mean": 110.5,
    ...
  },
  "calibration_badge": {
    "overall": "good",
    "coverage": "within_target",
    "sharpness": "acceptable"
  },
  "execution_time_sec": 1.85
}
```

### POST /simulate-local

Blend global simulation with local model predictions.

**Request:**
```json
{
  "game_id": "TEST_001",
  "player_id": "curry_stephen",
  "global_distributions": {
    "PTS": [28.0, 30.5, 26.3, ...],
    "REB": [5.0, 4.5, 6.0, ...],
    "AST": [6.5, 7.0, 5.5, ...]
  },
  "local_predictions": {
    "PTS": 30.0,
    "REB": 4.5,
    "AST": 7.0
  },
  "blend_weights": {
    "global": 0.6,
    "local": 0.4
  }
}
```

**Response:**
```json
{
  "game_id": "TEST_001",
  "player_id": "curry_stephen",
  "blended_distributions": {
    "PTS": {
      "mean": 29.0,
      "median": 28.8,
      "std": 5.8,
      ...
    },
    ...
  },
  "blend_weights_used": {
    "global": 0.6,
    "local": 0.4
  }
}
```

### POST /benchmark

Run comprehensive model benchmarking.

**Request:**
```json
{
  "evaluation_window": "rolling_30_games",
  "models": [
    "original_global_only",
    "blended_global_plus_local",
    "baselines_ridge",
    "baselines_xgboost"
  ],
  "metrics": ["mae", "rmse", "crps", "coverage_80"]
}
```

**Response:**
```json
{
  "evaluation_window": "rolling_30_games",
  "models_compared": [...],
  "accuracy_metrics": {
    "original_global_only": {
      "PTS_MAE": 4.8,
      "PTS_RMSE": 6.5,
      "coverage_80": 0.82,
      ...
    },
    ...
  },
  "efficiency_metrics": {
    "original_global_only": {
      "train_time_sec": 120.0,
      "infer_time_ms_per_player": 1800.0,
      ...
    },
    ...
  },
  "overall_metrics": {
    "spearman_rank_correlation": 0.78,
    "decision_gain_sim": 0.12
  },
  "best_model": "blended_global_plus_local",
  "execution_time_sec": 45.2
}
```

## Configuration

API configuration is managed in `configs/default.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30
  enable_cors: true
  log_level: "info"
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid input)
- **404**: Not Found (resource not found)
- **500**: Internal Server Error (server-side error)
- **504**: Gateway Timeout (request timeout)

Error responses include detailed messages:

```json
{
  "detail": "Failed to simulate player curry_stephen: Invalid posterior covariance matrix"
}
```

## Authentication

Currently, the API does not require authentication. For production deployment, consider adding:
- JWT token authentication
- API key validation
- Rate limiting

## Performance

Target performance metrics:
- Global simulation: ≤ 2.0 seconds per player
- Local blending: ≤ 0.5 seconds per player
- Benchmark: ≤ 60 seconds for 30-game window

## Examples

See `examples/api_demo.py` for complete usage examples.

```bash
# Run the demo (requires server to be running)
python examples/api_demo.py
```

## Development

### Running Tests

```bash
# Run API tests
pytest tests/test_api.py -v
```

### Adding New Endpoints

1. Define request/response models using Pydantic
2. Implement endpoint function with proper error handling
3. Add logging for monitoring
4. Update documentation
5. Add tests

## Deployment

### Docker

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

Build and run:
```bash
docker build -t nba-prediction-api .
docker run -p 8000:8000 nba-prediction-api
```

### Monitoring

Recommended monitoring setup:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **ELK Stack**: Log aggregation

## Support

For issues or questions:
- Check the interactive documentation at `/docs`
- Review examples in `examples/api_demo.py`
- Check logs in `logs/system.log`
