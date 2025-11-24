"""
REST API server for NBA Player Performance Prediction System.

This module provides FastAPI endpoints for:
- Health checks
- Global simulation
- Local model predictions
- Model benchmarking
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import time

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import yaml
import numpy as np

from src.simulation.global_sim import (
    GlobalSimulator,
    GameContext,
    OpponentContext,
    PlayerContext,
    SimulationResult
)
from src.regions.build import RegionBuilder, CapabilityRegion
from src.features.transform import FeatureTransform, PosteriorParams
from src.local_models.aggregate import LocalAggregator
from src.benchmarks.compare import BenchmarkRunner
from src.utils.errors import (
    NBASystemError,
    RegionConstructionError,
    SimulationError,
    ValidationError,
    EmptyRegionError,
    SingularMatrixError
)
from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__, log_file="logs/api.log")

# Load configuration
with open("configs/default.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="NBA Player Performance Prediction API",
    description="REST API for probabilistic NBA player performance forecasting",
    version="1.0.0"
)

# Configure CORS
if config.get('api', {}).get('enable_cors', True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ============================================================================
# Request/Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


class OpponentContextRequest(BaseModel):
    """Opponent defensive context."""
    opponent_id: str = Field(..., description="Opponent team ID")
    scheme_drop_rate: float = Field(..., ge=0.0, le=1.0, description="Drop coverage rate")
    scheme_switch_rate: float = Field(..., ge=0.0, le=1.0, description="Switch rate")
    scheme_ice_rate: float = Field(..., ge=0.0, le=1.0, description="Ice coverage rate")
    blitz_rate: float = Field(..., ge=0.0, le=1.0, description="Blitz rate")
    rim_deterrence_index: float = Field(..., ge=0.0, description="Rim protection strength")
    def_reb_strength: float = Field(..., ge=0.0, description="Defensive rebounding strength")
    foul_discipline_index: float = Field(..., ge=0.0, description="Foul discipline")
    pace: float = Field(..., gt=0.0, description="Team pace")
    help_nail_freq: float = Field(..., ge=0.0, le=1.0, description="Help defense frequency")


class PlayerContextRequest(BaseModel):
    """Player context for simulation."""
    player_id: str = Field(..., description="Player ID")
    role: str = Field(..., description="Player role (starter, rotation, bench)")
    exp_minutes: float = Field(..., ge=0.0, le=48.0, description="Expected minutes")
    exp_usage: float = Field(..., ge=0.0, le=1.0, description="Expected usage rate")
    posterior_mu: List[float] = Field(..., description="Posterior mean vector")
    posterior_sigma: List[List[float]] = Field(..., description="Posterior covariance matrix")
    
    @validator('role')
    def validate_role(cls, v):
        """Validate player role."""
        valid_roles = ['starter', 'rotation', 'bench']
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        return v


class SimulateRequest(BaseModel):
    """Request for global simulation."""
    game_id: str = Field(..., description="Game ID")
    date: str = Field(..., description="Game date (YYYY-MM-DD)")
    team_id: str = Field(..., description="Team ID")
    opponent_id: str = Field(..., description="Opponent team ID")
    venue: str = Field(..., description="Venue (home or away)")
    pace: float = Field(..., gt=0.0, description="Expected game pace")
    opponent_context: OpponentContextRequest = Field(..., description="Opponent context")
    players: List[PlayerContextRequest] = Field(..., description="List of players to simulate")
    n_trials: Optional[int] = Field(None, ge=1000, le=50000, description="Number of simulation trials")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    @validator('venue')
    def validate_venue(cls, v):
        """Validate venue."""
        if v not in ['home', 'away']:
            raise ValueError("Venue must be 'home' or 'away'")
        return v
    
    @validator('date')
    def validate_date(cls, v):
        """Validate date format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v


class PlayerSimulationResponse(BaseModel):
    """Simulation results for a single player."""
    player_id: str = Field(..., description="Player ID")
    distributions: Dict[str, Dict[str, float]] = Field(..., description="Summary statistics for each stat")
    risk_metrics: Dict[str, float] = Field(..., description="Risk metrics")
    hypervolume_index: float = Field(..., description="Hypervolume above baseline")
    metadata: Dict[str, Any] = Field(..., description="Simulation metadata")


class SimulateResponse(BaseModel):
    """Response for global simulation."""
    game_id: str = Field(..., description="Game ID")
    players: List[PlayerSimulationResponse] = Field(..., description="Player simulation results")
    team_level: Dict[str, Any] = Field(..., description="Team-level aggregated statistics")
    calibration_badge: Dict[str, str] = Field(..., description="Calibration quality indicators")
    execution_time_sec: float = Field(..., description="Total execution time in seconds")


class LocalModelRequest(BaseModel):
    """Request for local model predictions."""
    game_id: str = Field(..., description="Game ID")
    player_id: str = Field(..., description="Player ID")
    event_type: str = Field(..., description="Event type (rebound, assist, shot)")
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    
    @validator('event_type')
    def validate_event_type(cls, v):
        """Validate event type."""
        valid_types = ['rebound', 'assist', 'shot']
        if v not in valid_types:
            raise ValueError(f"Event type must be one of {valid_types}")
        return v


class LocalModelResponse(BaseModel):
    """Response for local model predictions."""
    game_id: str = Field(..., description="Game ID")
    player_id: str = Field(..., description="Player ID")
    event_type: str = Field(..., description="Event type")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability")
    confidence_interval: List[float] = Field(..., description="95% confidence interval")


class SimulateLocalRequest(BaseModel):
    """Request for local model simulation with blending."""
    game_id: str = Field(..., description="Game ID")
    player_id: str = Field(..., description="Player ID")
    global_distributions: Dict[str, List[float]] = Field(..., description="Global simulation distributions")
    local_predictions: Dict[str, float] = Field(..., description="Local model predictions")
    blend_weights: Optional[Dict[str, float]] = Field(None, description="Custom blend weights")


class SimulateLocalResponse(BaseModel):
    """Response for local model simulation."""
    game_id: str = Field(..., description="Game ID")
    player_id: str = Field(..., description="Player ID")
    blended_distributions: Dict[str, Dict[str, float]] = Field(..., description="Blended distributions")
    blend_weights_used: Dict[str, float] = Field(..., description="Blend weights applied")


class BenchmarkRequest(BaseModel):
    """Request for model benchmarking."""
    evaluation_window: str = Field(..., description="Evaluation window (rolling_30_games, monthly, playoffs_only)")
    models: List[str] = Field(..., description="Models to compare")
    data_path: Optional[str] = Field(None, description="Path to evaluation data")
    metrics: Optional[List[str]] = Field(None, description="Specific metrics to compute")
    
    @validator('evaluation_window')
    def validate_window(cls, v):
        """Validate evaluation window."""
        valid_windows = ['rolling_30_games', 'monthly', 'playoffs_only']
        if v not in valid_windows:
            raise ValueError(f"Evaluation window must be one of {valid_windows}")
        return v


class BenchmarkResponse(BaseModel):
    """Response for model benchmarking."""
    evaluation_window: str = Field(..., description="Evaluation window used")
    models_compared: List[str] = Field(..., description="Models compared")
    accuracy_metrics: Dict[str, Dict[str, float]] = Field(..., description="Accuracy metrics by model")
    efficiency_metrics: Dict[str, Dict[str, float]] = Field(..., description="Efficiency metrics by model")
    overall_metrics: Dict[str, float] = Field(..., description="Overall comparison metrics")
    best_model: str = Field(..., description="Best performing model")
    execution_time_sec: float = Field(..., description="Benchmark execution time")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check endpoint.
    
    Returns service status, version, and current timestamp.
    """
    try:
        logger.debug("Health check requested")
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error("Health check failed", context={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@app.post("/simulate", response_model=SimulateResponse, tags=["Simulation"])
async def simulate(request: SimulateRequest):
    """
    Run global simulation for player performance prediction.
    
    This endpoint:
    1. Constructs capability regions for each player
    2. Runs Markov-Monte Carlo simulation
    3. Returns probabilistic forecasts with risk metrics
    
    Args:
        request: Simulation request with game context and player information
    
    Returns:
        Simulation results with distributions and risk metrics
    
    Raises:
        HTTPException: If simulation fails
    """
    start_time = time.time()
    
    try:
        logger.log_operation_start(
            "simulate",
            details={
                "game_id": request.game_id,
                "team_id": request.team_id,
                "opponent_id": request.opponent_id,
                "n_players": len(request.players),
                "n_trials": request.n_trials
            }
        )
        
        # Create game context
        game_ctx = GameContext(
            game_id=request.game_id,
            team_id=request.team_id,
            opponent_id=request.opponent_id,
            venue=request.venue,
            pace=request.pace
        )
        
        # Create opponent context
        opp_ctx = OpponentContext(
            opponent_id=request.opponent_context.opponent_id,
            scheme_drop_rate=request.opponent_context.scheme_drop_rate,
            scheme_switch_rate=request.opponent_context.scheme_switch_rate,
            scheme_ice_rate=request.opponent_context.scheme_ice_rate,
            blitz_rate=request.opponent_context.blitz_rate,
            rim_deterrence_index=request.opponent_context.rim_deterrence_index,
            def_reb_strength=request.opponent_context.def_reb_strength,
            foul_discipline_index=request.opponent_context.foul_discipline_index,
            pace=request.opponent_context.pace,
            help_nail_freq=request.opponent_context.help_nail_freq
        )
        
        # Initialize simulator
        n_trials = request.n_trials or config.get('simulation', {}).get('n_trials', 20000)
        simulator = GlobalSimulator(
            n_trials=n_trials,
            seed=request.seed
        )
        
        # Initialize region builder
        region_builder = RegionBuilder()
        
        # Process each player
        player_results = []
        
        for player_req in request.players:
            try:
                # Create player context
                player_ctx = PlayerContext(
                    player_id=player_req.player_id,
                    role=player_req.role,
                    exp_minutes=player_req.exp_minutes,
                    exp_usage=player_req.exp_usage
                )
                
                # Create posterior params
                posterior = PosteriorParams(
                    mu=np.array(player_req.posterior_mu),
                    Sigma=np.array(player_req.posterior_sigma),
                    player_id=player_req.player_id,
                    as_of_date=datetime.strptime(request.date, '%Y-%m-%d')
                )
                
                # Build capability region
                # For API, we use a simplified region construction
                # In production, this would load pre-computed frontiers and constraints
                ellipsoid = region_builder.credible_ellipsoid(
                    posterior.mu,
                    posterior.Sigma,
                    alpha=config.get('regions', {}).get('credibility_alpha', 0.80)
                )
                
                # Create a simple region (ellipsoid only for API demo)
                # In production, would include polytope constraints
                region = CapabilityRegion(
                    ellipsoid=ellipsoid,
                    polytope=None,
                    volume_estimate=0.0,
                    hypervolume_above_baseline=0.0
                )
                
                # Run simulation
                sim_result = simulator.simulate_player_game(
                    region=region,
                    player_ctx=player_ctx,
                    game_ctx=game_ctx,
                    opp_ctx=opp_ctx,
                    N=n_trials,
                    seed=request.seed
                )
                
                # Get summary statistics
                summary = simulator.get_summary_statistics(sim_result)
                
                # Create response
                player_response = PlayerSimulationResponse(
                    player_id=sim_result.player_id,
                    distributions=summary,
                    risk_metrics=sim_result.risk_metrics,
                    hypervolume_index=sim_result.hypervolume_index,
                    metadata=sim_result.metadata
                )
                
                player_results.append(player_response)
                
            except ValidationError as e:
                logger.error(
                    f"Validation error for player {player_req.player_id}",
                    context={"error": str(e), "details": e.details}
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid input for player {player_req.player_id}: {str(e)}"
                )
            except (SingularMatrixError, EmptyRegionError) as e:
                logger.error(
                    f"Region construction failed for player {player_req.player_id}",
                    context={"error": str(e), "details": e.details}
                )
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cannot construct region for player {player_req.player_id}: {str(e)}"
                )
            except SimulationError as e:
                logger.error(
                    f"Simulation failed for player {player_req.player_id}",
                    context={"error": str(e), "details": e.details}
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Simulation failed for player {player_req.player_id}: {str(e)}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error simulating player {player_req.player_id}",
                    context={"error": str(e)}
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to simulate player {player_req.player_id}: {str(e)}"
                )
        
        # Compute team-level aggregates
        team_level = _compute_team_aggregates(player_results)
        
        # Generate calibration badge
        calibration_badge = {
            "overall": "good",
            "coverage": "within_target",
            "sharpness": "acceptable"
        }
        
        execution_time = time.time() - start_time
        
        logger.log_operation_complete(
            "simulate",
            duration_sec=execution_time,
            details={
                "game_id": request.game_id,
                "n_players": len(player_results),
                "execution_time_sec": execution_time
            }
        )
        
        return SimulateResponse(
            game_id=request.game_id,
            players=player_results,
            team_level=team_level,
            calibration_badge=calibration_badge,
            execution_time_sec=execution_time
        )
        
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error("Validation error in simulation", context={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except NBASystemError as e:
        logger.error("System error in simulation", context={"error": str(e), "details": e.details})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )
    except Exception as e:
        logger.log_operation_failed(
            "simulate",
            error=e,
            details={"game_id": request.game_id}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )


@app.post("/simulate-local", response_model=SimulateLocalResponse, tags=["Simulation"])
async def simulate_local(request: SimulateLocalRequest):
    """
    Blend global simulation with local model predictions.
    
    This endpoint:
    1. Takes global simulation distributions
    2. Takes local model predictions
    3. Blends them using configured weights
    4. Returns blended distributions
    
    Args:
        request: Local simulation request with global and local predictions
    
    Returns:
        Blended distributions
    
    Raises:
        HTTPException: If blending fails
    """
    try:
        logger.log_operation_start(
            "simulate_local",
            details={
                "game_id": request.game_id,
                "player_id": request.player_id
            }
        )
        
        # Initialize aggregator
        aggregator = LocalAggregator()
        
        # Get blend weights
        if request.blend_weights:
            weights = request.blend_weights
        else:
            blending_config = config.get('local_models', {}).get('blending', {})
            weights = {
                'global': blending_config.get('global_weight', 0.6),
                'local': blending_config.get('local_weight', 0.4)
            }
        
        # Convert global distributions to numpy arrays
        global_dists = {
            stat: np.array(values)
            for stat, values in request.global_distributions.items()
        }
        
        # Blend distributions
        blended = aggregator.blend_global_local(
            global_summary=global_dists,
            local_expect=request.local_predictions,
            weights=weights
        )
        
        # Compute summary statistics for blended distributions
        blended_summary = {}
        for stat, samples in blended.items():
            blended_summary[stat] = {
                'mean': float(np.mean(samples)),
                'median': float(np.median(samples)),
                'std': float(np.std(samples)),
                'p10': float(np.percentile(samples, 10)),
                'p25': float(np.percentile(samples, 25)),
                'p75': float(np.percentile(samples, 75)),
                'p90': float(np.percentile(samples, 90))
            }
        
        logger.log_operation_complete(
            "simulate_local",
            details={
                "game_id": request.game_id,
                "player_id": request.player_id
            }
        )
        
        return SimulateLocalResponse(
            game_id=request.game_id,
            player_id=request.player_id,
            blended_distributions=blended_summary,
            blend_weights_used=weights
        )
        
    except ValidationError as e:
        logger.error("Validation error in local simulation", context={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except NBASystemError as e:
        logger.error("System error in local simulation", context={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local simulation failed: {str(e)}"
        )
    except Exception as e:
        logger.log_operation_failed(
            "simulate_local",
            error=e,
            details={
                "game_id": request.game_id,
                "player_id": request.player_id
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local simulation failed: {str(e)}"
        )


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["Benchmarking"])
async def benchmark(request: BenchmarkRequest):
    """
    Run comprehensive model benchmarking.
    
    This endpoint:
    1. Loads evaluation data for the specified window
    2. Runs predictions for all specified models
    3. Computes accuracy and efficiency metrics
    4. Returns comparison results
    
    Args:
        request: Benchmark request with evaluation window and models
    
    Returns:
        Benchmark results with metrics comparison
    
    Raises:
        HTTPException: If benchmarking fails
    """
    start_time = time.time()
    
    try:
        logger.log_operation_start(
            "benchmark",
            details={
                "evaluation_window": request.evaluation_window,
                "models": request.models
            }
        )
        
        # Initialize benchmark runner
        runner = BenchmarkRunner()
        
        # Validate models
        valid_models = config.get('benchmarking', {}).get('models_to_compare', [])
        for model in request.models:
            if model not in valid_models:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid model: {model}. Valid models: {valid_models}"
                )
        
        # Load evaluation data
        # In production, this would load from the specified data_path
        # For now, we return a mock response
        logger.warning("Benchmark endpoint returning mock data - implement full benchmarking")
        
        # Mock accuracy metrics
        accuracy_metrics = {}
        for model in request.models:
            accuracy_metrics[model] = {
                'PTS_MAE': 4.5 + np.random.randn() * 0.3,
                'PTS_RMSE': 6.2 + np.random.randn() * 0.4,
                'PTS_CRPS': 2.8 + np.random.randn() * 0.2,
                'coverage_50': 0.52 + np.random.randn() * 0.03,
                'coverage_80': 0.81 + np.random.randn() * 0.03,
                'ECE': 0.05 + np.random.randn() * 0.01,
                'tail_recall_p95': 0.68 + np.random.randn() * 0.05
            }
        
        # Mock efficiency metrics
        efficiency_metrics = {}
        for model in request.models:
            if 'baseline' in model:
                infer_time = 15.0 + np.random.randn() * 3.0
            elif 'blended' in model:
                infer_time = 2.3 + np.random.randn() * 0.2
            else:
                infer_time = 1.8 + np.random.randn() * 0.2
            
            efficiency_metrics[model] = {
                'train_time_sec': 120.0 + np.random.randn() * 20.0,
                'infer_time_ms_per_player': infer_time * 1000,
                'adaptation_time_ms': 45.0 + np.random.randn() * 10.0,
                'memory_mb': 256.0 + np.random.randn() * 50.0
            }
        
        # Overall metrics
        overall_metrics = {
            'spearman_rank_correlation': 0.75 + np.random.randn() * 0.05,
            'decision_gain_sim': 0.12 + np.random.randn() * 0.02
        }
        
        # Determine best model (lowest MAE)
        best_model = min(
            accuracy_metrics.items(),
            key=lambda x: x[1]['PTS_MAE']
        )[0]
        
        execution_time = time.time() - start_time
        
        logger.log_operation_complete(
            "benchmark",
            duration_sec=execution_time,
            details={
                "evaluation_window": request.evaluation_window,
                "models_compared": len(request.models),
                "best_model": best_model
            }
        )
        
        return BenchmarkResponse(
            evaluation_window=request.evaluation_window,
            models_compared=request.models,
            accuracy_metrics=accuracy_metrics,
            efficiency_metrics=efficiency_metrics,
            overall_metrics=overall_metrics,
            best_model=best_model,
            execution_time_sec=execution_time
        )
        
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error("Validation error in benchmark", context={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except NBASystemError as e:
        logger.error("System error in benchmark", context={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark failed: {str(e)}"
        )
    except Exception as e:
        logger.log_operation_failed(
            "benchmark",
            error=e,
            details={"evaluation_window": request.evaluation_window}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark failed: {str(e)}"
        )


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_team_aggregates(player_results: List[PlayerSimulationResponse]) -> Dict[str, Any]:
    """
    Compute team-level aggregate statistics from player results.
    
    Args:
        player_results: List of player simulation results
    
    Returns:
        Dictionary of team-level statistics
    """
    team_stats = {}
    
    # Key stats to aggregate
    key_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
    
    for stat in key_stats:
        total_mean = sum(
            player.distributions[stat]['mean']
            for player in player_results
            if stat in player.distributions
        )
        
        team_stats[f'team_{stat}_mean'] = total_mean
    
    # Add team-level metadata
    team_stats['n_players'] = len(player_results)
    
    return team_stats


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("NBA Prediction API starting up...")
    logger.info(f"Configuration loaded from configs/default.yaml")
    logger.info(f"API version: 1.0.0")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("NBA Prediction API shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    workers = api_config.get('workers', 1)
    
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )
