"""
Demo script for NBA Prediction API usage.

This script demonstrates how to interact with the REST API endpoints.
"""

import requests
import json
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_simulate():
    """Test the global simulation endpoint."""
    print("Testing /simulate endpoint...")
    
    # Create sample request
    request_data = {
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
                "posterior_sigma": [
                    [0.04, 0.01, 0.005, -0.002, 0.001, 0.0, 0.0],
                    [0.01, 0.02, 0.003, 0.001, 0.0, 0.0, 0.0],
                    [0.005, 0.003, 0.015, -0.004, 0.001, 0.0, 0.0],
                    [-0.002, 0.001, -0.004, 0.008, 0.0, 0.0, 0.0],
                    [0.001, 0.0, 0.001, 0.0, 0.01, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
                ]
            }
        ],
        "n_trials": 5000,
        "seed": 42
    }
    
    response = requests.post(f"{BASE_URL}/simulate", json=request_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Game ID: {result['game_id']}")
        print(f"Execution time: {result['execution_time_sec']:.2f}s")
        print(f"\nPlayer: {result['players'][0]['player_id']}")
        print(f"PTS distribution: {result['players'][0]['distributions']['PTS']}")
        print(f"Hypervolume index: {result['players'][0]['hypervolume_index']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()


def test_simulate_local():
    """Test the local model blending endpoint."""
    print("Testing /simulate-local endpoint...")
    
    # Create sample request with mock distributions
    request_data = {
        "game_id": "TEST_001",
        "player_id": "curry_stephen",
        "global_distributions": {
            "PTS": list(np.random.normal(28.0, 6.0, 1000).tolist()),
            "REB": list(np.random.normal(5.0, 2.0, 1000).tolist()),
            "AST": list(np.random.normal(6.5, 2.5, 1000).tolist())
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
    
    response = requests.post(f"{BASE_URL}/simulate-local", json=request_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Player: {result['player_id']}")
        print(f"Blend weights: {result['blend_weights_used']}")
        print(f"Blended PTS: {result['blended_distributions']['PTS']}")
    else:
        print(f"Error: {response.text}")
    print()


def test_benchmark():
    """Test the benchmark endpoint."""
    print("Testing /benchmark endpoint...")
    
    request_data = {
        "evaluation_window": "rolling_30_games",
        "models": [
            "original_global_only",
            "blended_global_plus_local",
            "baselines_ridge"
        ]
    }
    
    response = requests.post(f"{BASE_URL}/benchmark", json=request_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Evaluation window: {result['evaluation_window']}")
        print(f"Best model: {result['best_model']}")
        print(f"Execution time: {result['execution_time_sec']:.2f}s")
        print(f"\nAccuracy metrics:")
        for model, metrics in result['accuracy_metrics'].items():
            print(f"  {model}: MAE={metrics['PTS_MAE']:.2f}, Coverage_80={metrics['coverage_80']:.3f}")
    else:
        print(f"Error: {response.text}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("NBA Prediction API Demo")
    print("=" * 60)
    print()
    print("Make sure the API server is running:")
    print("  uvicorn src.api.server:app --reload")
    print()
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_simulate()
        test_simulate_local()
        test_benchmark()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server.")
        print("Please start the server with: uvicorn src.api.server:app --reload")
    except Exception as e:
        print(f"ERROR: {str(e)}")
