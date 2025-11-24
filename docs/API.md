# API Documentation

This document provides comprehensive documentation for the NBA Player Performance Prediction System REST API.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Global Simulation](#global-simulation)
  - [Local Model Simulation](#local-model-simulation)
  - [Benchmarking](#benchmarking)
- [Request/Response Models](#requestresponse-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Overview

The NBA Prediction API provides RESTful endpoints for:

- Running probabilistic player performance simulations
- Blending global and local model predictions
- Benchmarking model performance
- Health monitoring

**Base URL**: `http://localhost:8000` (default)

**API Version**: 1.0.0

**Content Type**: `application/json`

## Getting Started

### Starting the Server

```bash
# Development mode with auto-reload
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

# Production mode with multiple workers
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Interactive Documentation

Once the server is running, access interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Quick Test

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Authentication

**Current Version**: No authentication required

**Future Versions**: Will support JWT token authentication for production deployments.

## Endpoints

### Health Check

Check API service status.

**Endpoint**: `GET /health`

**Parameters**: None

**Response**: `200 OK`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Example**:

```bash
curl http://localhost:8000/health
```

---

### Global Simulation

Run global Markov-Monte Carlo simulation for player performance prediction.

**Endpoint**: `POST /simulate`

**Request Body**:

```json
{
  "game_id": "string",
  "date": "YYYY-MM-DD",
  "team_id": "string",
  "opponent_id": "string",
  "venue": "home" | "away",
  "pace": number,
  "opponent_context": {
    "opponent_id": "string",
    "scheme_drop_rate": number,      // 0.0 - 1.0
    "scheme_switch_rate": number,    // 0.0 - 1.0
    "scheme_ice_rate": number,       // 0.0 - 1.0
    "blitz_rate": number,            // 0.0 - 1.0
    "rim_deterrence_index": number,  // >= 0.0
    "def_reb_strength": number,      // >= 0.0
    "foul_discipline_index": number, // >= 0.0
    "pace": number,                  // > 0.0
    "help_nail_freq": number         // 0.0 - 1.0
  },
  "players": [
    {
      "player_id": "string",
      "role": "starter" | "rotation" | "bench",
      "exp_minutes": number,         // 0.0 - 48.0
      "exp_usage": number,           // 0.0 - 1.0
      "posterior_mu": [number],      // Mean vector
      "posterior_sigma": [[number]]  // Covariance matrix
    }
  ],
  "n_trials": number,                // Optional, 1000-50000
  "seed": number                     // Optional, for reproducibility
}
```

**Response**: `200 OK`

```json
{
  "game_id": "string",
  "players": [
    {
      "player_id": "string",
      "distributions": {
        "PTS": {
          "mean": number,
          "median": number,
          "std": number,
          "p10": number,
          "p25": number,
          "p75": number,
          "p90": number
        },
        "REB": { ... },
        "AST": { ... },
        // ... other stats
      },
      "risk_metrics": {
        "var_95": number,
        "cvar_95": number,
        "tail_prob_p95": number
      },
      "hypervolume_index": number,
      "metadata": {
        "n_trials": number,
        "seed": number,
        "execution_time_ms": number
      }
    }
  ],
  "team_level": {
    "team_PTS_mean": number,
    "team_REB_mean": number,
    "team_AST_mean": number,
    "n_players": number
  },
  "calibration_badge": {
    "overall": "good" | "acceptable" | "poor",
    "coverage": "within_target" | "under_coverage" | "over_coverage",
    "sharpness": "acceptable" | "too_wide" | "too_narrow"
  },
  "execution_time_sec": number
}
```

**Error Responses**:

- `400 Bad Request`: Invalid input parameters
- `422 Unprocessable Entity`: Cannot construct capability region
- `500 Internal Server Error`: Simulation failed

**Example**:

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "G001",
    "date": "2024-01-15",
    "team_id": "GSW",
    "opponent_id": "LAL",
    "venue": "home",
    "pace": 100.5,
    "opponent_context": {
      "opponent_id": "LAL",
      "scheme_drop_rate": 0.4,
      "scheme_switch_rate": 0.3,
      "scheme_ice_rate": 0.2,
      "blitz_rate": 0.15,
      "rim_deterrence_index": 1.2,
      "def_reb_strength": 1.1,
      "foul_discipline_index": 0.9,
      "pace": 100.5,
      "help_nail_freq": 0.35
    },
    "players": [
      {
        "player_id": "curry_stephen",
        "role": "starter",
        "exp_minutes": 34.0,
        "exp_usage": 0.32,
        "posterior_mu": [30.0, 5.0, 6.0, 1.5, 0.3, 2.5],
        "posterior_sigma": [
          [25, 2, 3, 0.5, 0.1, 1],
          [2, 4, 1, 0.2, 0.05, 0.3],
          [3, 1, 9, 0.3, 0.1, 0.5],
          [0.5, 0.2, 0.3, 1, 0.05, 0.2],
          [0.1, 0.05, 0.1, 0.05, 0.25, 0.1],
          [1, 0.3, 0.5, 0.2, 0.1, 4]
        ]
      }
    ],
    "n_trials": 20000,
    "seed": 42
  }'
```

---

### Local Model Simulation

Blend global simulation distributions with local model predictions.

**Endpoint**: `POST /simulate-local`

**Request Body**:

```json
{
  "game_id": "string",
  "player_id": "string",
  "global_distributions": {
    "PTS": [number],  // Array of samples
    "REB": [number],
    "AST": [number],
    // ... other stats
  },
  "local_predictions": {
    "PTS": number,    // Point estimate
    "REB": number,
    "AST": number,
    // ... other stats
  },
  "blend_weights": {  // Optional
    "global": number, // 0.0 - 1.0
    "local": number   // 0.0 - 1.0
  }
}
```

**Response**: `200 OK`

```json
{
  "game_id": "string",
  "player_id": "string",
  "blended_distributions": {
    "PTS": {
      "mean": number,
      "median": number,
      "std": number,
      "p10": number,
      "p25": number,
      "p75": number,
      "p90": number
    },
    // ... other stats
  },
  "blend_weights_used": {
    "global": number,
    "local": number
  }
}
```

**Error Responses**:

- `400 Bad Request`: Invalid input parameters
- `500 Internal Server Error`: Blending failed

**Example**:

```bash
curl -X POST http://localhost:8000/simulate-local \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "G001",
    "player_id": "curry_stephen",
    "global_distributions": {
      "PTS": [28.5, 31.2, 29.8, 30.1, 27.9],
      "REB": [5.1, 4.8, 5.5, 5.0, 4.9],
      "AST": [6.2, 7.1, 5.8, 6.5, 6.0]
    },
    "local_predictions": {
      "PTS": 30.2,
      "REB": 5.3,
      "AST": 6.5
    },
    "blend_weights": {
      "global": 0.6,
      "local": 0.4
    }
  }'
```

---

### Benchmarking

Run comprehensive model comparison and benchmarking.

**Endpoint**: `POST /benchmark`

**Request Body**:

```json
{
  "evaluation_window": "rolling_30_games" | "monthly" | "playoffs_only",
  "models": ["string"],  // List of model names
  "data_path": "string", // Optional
  "metrics": ["string"]  // Optional, specific metrics to compute
}
```

**Response**: `200 OK`

```json
{
  "evaluation_window": "string",
  "models_compared": ["string"],
  "accuracy_metrics": {
    "model_name": {
      "PTS_MAE": number,
      "PTS_RMSE": number,
      "PTS_CRPS": number,
      "coverage_50": number,
      "coverage_80": number,
      "ECE": number,
      "tail_recall_p95": number
    }
  },
  "efficiency_metrics": {
    "model_name": {
      "train_time_sec": number,
      "infer_time_ms_per_player": number,
      "adaptation_time_ms": number,
      "memory_mb": number
    }
  },
  "overall_metrics": {
    "spearman_rank_correlation": number,
    "decision_gain_sim": number
  },
  "best_model": "string",
  "execution_time_sec": number
}
```

**Valid Models**:
- `original_global_only`
- `local_only`
- `blended_global_plus_local`
- `baselines_ridge`
- `baselines_xgboost`
- `baselines_mlp`

**Valid Evaluation Windows**:
- `rolling_30_games`: Rolling 30-game window
- `monthly`: Monthly aggregation
- `playoffs_only`: Playoff games only

**Error Responses**:

- `400 Bad Request`: Invalid model or window
- `500 Internal Server Error`: Benchmark failed

**Example**:

```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_window": "rolling_30_games",
    "models": [
      "original_global_only",
      "blended_global_plus_local",
      "baselines_xgboost"
    ],
    "metrics": ["mae", "rmse", "crps", "coverage_80"]
  }'
```

---

## Request/Response Models

### Common Data Types

**Player Role**:
- `starter`: Starting lineup player
- `rotation`: Regular rotation player
- `bench`: Bench player

**Venue**:
- `home`: Home game
- `away`: Away game

**Statistics**:
- `PTS`: Points
- `REB` or `TRB`: Total rebounds
- `AST`: Assists
- `STL`: Steals
- `BLK`: Blocks
- `TOV`: Turnovers
- `FGA`: Field goal attempts
- `3PA`: Three-point attempts
- `FTA`: Free throw attempts
- `PF`: Personal fouls

### Validation Rules

**Opponent Context**:
- All rate fields (scheme_drop_rate, etc.) must be between 0.0 and 1.0
- Index fields (rim_deterrence_index, etc.) must be >= 0.0
- Pace must be > 0.0

**Player Context**:
- `exp_minutes`: 0.0 - 48.0
- `exp_usage`: 0.0 - 1.0
- `posterior_mu`: Array of numbers (length must match dimension)
- `posterior_sigma`: Square matrix (must be positive definite)

**Simulation Parameters**:
- `n_trials`: 1000 - 50000 (recommended: 20000)
- `seed`: Any integer for reproducibility

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid input parameters
- `404 Not Found`: Endpoint not found
- `422 Unprocessable Entity`: Valid input but cannot process (e.g., singular matrix)
- `500 Internal Server Error`: Server-side error
- `504 Gateway Timeout`: Request timeout

### Common Errors

**Invalid Date Format**:
```json
{
  "detail": "Date must be in YYYY-MM-DD format"
}
```

**Invalid Role**:
```json
{
  "detail": "Role must be one of ['starter', 'rotation', 'bench']"
}
```

**Singular Matrix**:
```json
{
  "detail": "Cannot construct region for player curry_stephen: Covariance matrix is singular"
}
```

**Empty Region**:
```json
{
  "detail": "Cannot construct region for player curry_stephen: Region is empty after constraint intersection"
}
```

## Rate Limiting

**Current Version**: No rate limiting

**Future Versions**: Will implement rate limiting based on:
- Requests per minute per IP
- Concurrent simulations per user
- Total compute time per hour

## Examples

### Python Client

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Run simulation
simulation_request = {
    "game_id": "G001",
    "date": "2024-01-15",
    "team_id": "GSW",
    "opponent_id": "LAL",
    "venue": "home",
    "pace": 100.5,
    "opponent_context": {
        "opponent_id": "LAL",
        "scheme_drop_rate": 0.4,
        "scheme_switch_rate": 0.3,
        "scheme_ice_rate": 0.2,
        "blitz_rate": 0.15,
        "rim_deterrence_index": 1.2,
        "def_reb_strength": 1.1,
        "foul_discipline_index": 0.9,
        "pace": 100.5,
        "help_nail_freq": 0.35
    },
    "players": [
        {
            "player_id": "curry_stephen",
            "role": "starter",
            "exp_minutes": 34.0,
            "exp_usage": 0.32,
            "posterior_mu": [30.0, 5.0, 6.0, 1.5, 0.3, 2.5],
            "posterior_sigma": [
                [25, 2, 3, 0.5, 0.1, 1],
                [2, 4, 1, 0.2, 0.05, 0.3],
                [3, 1, 9, 0.3, 0.1, 0.5],
                [0.5, 0.2, 0.3, 1, 0.05, 0.2],
                [0.1, 0.05, 0.1, 0.05, 0.25, 0.1],
                [1, 0.3, 0.5, 0.2, 0.1, 4]
            ]
        }
    ],
    "n_trials": 20000,
    "seed": 42
}

response = requests.post(
    f"{BASE_URL}/simulate",
    json=simulation_request,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    result = response.json()
    print(f"Game ID: {result['game_id']}")
    print(f"Execution time: {result['execution_time_sec']:.2f}s")
    
    for player in result['players']:
        print(f"\nPlayer: {player['player_id']}")
        pts = player['distributions']['PTS']
        print(f"  PTS: {pts['mean']:.1f} ± {pts['std']:.1f}")
        print(f"  80% interval: [{pts['p10']:.1f}, {pts['p90']:.1f}]")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### JavaScript Client

```javascript
const BASE_URL = 'http://localhost:8000';

// Health check
fetch(`${BASE_URL}/health`)
  .then(response => response.json())
  .then(data => console.log(data));

// Run simulation
const simulationRequest = {
  game_id: 'G001',
  date: '2024-01-15',
  team_id: 'GSW',
  opponent_id: 'LAL',
  venue: 'home',
  pace: 100.5,
  opponent_context: {
    opponent_id: 'LAL',
    scheme_drop_rate: 0.4,
    scheme_switch_rate: 0.3,
    scheme_ice_rate: 0.2,
    blitz_rate: 0.15,
    rim_deterrence_index: 1.2,
    def_reb_strength: 1.1,
    foul_discipline_index: 0.9,
    pace: 100.5,
    help_nail_freq: 0.35
  },
  players: [
    {
      player_id: 'curry_stephen',
      role: 'starter',
      exp_minutes: 34.0,
      exp_usage: 0.32,
      posterior_mu: [30.0, 5.0, 6.0, 1.5, 0.3, 2.5],
      posterior_sigma: [
        [25, 2, 3, 0.5, 0.1, 1],
        [2, 4, 1, 0.2, 0.05, 0.3],
        [3, 1, 9, 0.3, 0.1, 0.5],
        [0.5, 0.2, 0.3, 1, 0.05, 0.2],
        [0.1, 0.05, 0.1, 0.05, 0.25, 0.1],
        [1, 0.3, 0.5, 0.2, 0.1, 4]
      ]
    }
  ],
  n_trials: 20000,
  seed: 42
};

fetch(`${BASE_URL}/simulate`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(simulationRequest)
})
  .then(response => response.json())
  .then(data => {
    console.log(`Game ID: ${data.game_id}`);
    console.log(`Execution time: ${data.execution_time_sec.toFixed(2)}s`);
    
    data.players.forEach(player => {
      console.log(`\nPlayer: ${player.player_id}`);
      const pts = player.distributions.PTS;
      console.log(`  PTS: ${pts.mean.toFixed(1)} ± ${pts.std.toFixed(1)}`);
      console.log(`  80% interval: [${pts.p10.toFixed(1)}, ${pts.p90.toFixed(1)}]`);
    });
  })
  .catch(error => console.error('Error:', error));
```

## Best Practices

1. **Use Seeds for Reproducibility**: Always provide a seed when you need reproducible results
2. **Batch Requests**: For multiple players, include them in a single request rather than multiple requests
3. **Handle Errors Gracefully**: Check status codes and handle errors appropriately
4. **Monitor Execution Time**: Track execution times to identify performance issues
5. **Validate Inputs**: Validate inputs client-side before sending to reduce errors
6. **Use Appropriate n_trials**: Balance accuracy (higher trials) with speed (lower trials)
7. **Cache Results**: Cache simulation results when appropriate to reduce API calls

## Changelog

### Version 1.0.0 (Current)

- Initial release
- Global simulation endpoint
- Local model blending endpoint
- Benchmarking endpoint
- Health check endpoint
- OpenAPI documentation

### Planned Features

- Authentication and authorization
- Rate limiting
- Batch processing endpoints
- Webhook notifications
- Real-time streaming updates
- Model versioning support
- A/B testing endpoints

## Support

For API questions or issues:
1. Check interactive documentation at `/docs`
2. Review this documentation
3. Check server logs for errors
4. Contact the development team

## License

Proprietary - All rights reserved
