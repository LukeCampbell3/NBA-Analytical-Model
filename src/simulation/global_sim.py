"""
Global simulation module for NBA player performance prediction.

This module implements the Markov-Monte Carlo simulation engine that samples from
capability regions and applies game-state transitions to generate probabilistic
forecasts of player performance.
"""

import os
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import yaml
from tqdm import tqdm

from src.regions.build import CapabilityRegion, RegionBuilder


def _simulate_player_worker(args: Tuple) -> 'SimulationResult':
    """
    Worker function for parallel player simulation.
    
    This function is defined at module level to be picklable for multiprocessing.
    
    Args:
        args: Tuple containing all arguments needed for simulation
    
    Returns:
        SimulationResult for the player
    """
    (
        region,
        player_ctx,
        game_ctx,
        opp_ctx,
        n_trials,
        seed,
        n_stints,
        transition_matrix,
        state_offsets,
        role_params
    ) = args
    
    # Create a temporary simulator instance for this worker
    # We can't pass the full simulator due to pickling issues
    from src.regions.build import RegionBuilder
    
    region_builder = RegionBuilder()
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # Sample capability vectors from region
    region_seed = rng.integers(0, 2**31) if seed is not None else None
    capability_samples = region_builder.sample_region(
        region,
        n=n_trials,
        seed=region_seed
    )
    
    # Initialize storage for box stats
    box_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', '3PA', 'FTA', 'PF']
    box_stats_samples = {stat: np.zeros(n_trials) for stat in box_stats}
    
    # Run N trials
    for trial in range(n_trials):
        # Sample minutes
        role = player_ctx.role
        role_config = role_params.get(role, role_params.get('rotation', {}))
        minutes_mean = role_config.get('minutes_mean', player_ctx.exp_minutes)
        minutes_sigma = role_config.get('minutes_sigma', 5.0)
        minutes = np.clip(rng.normal(minutes_mean, minutes_sigma), 0.0, 48.0)
        
        # Sample usage
        usage_alpha = role_config.get('usage_alpha', 5.0)
        usage_beta = role_config.get('usage_beta', 15.0)
        usage = rng.beta(usage_alpha, usage_beta)
        if minutes < 20:
            usage = usage * (0.8 + 0.2 * minutes / 20)
        
        # Split minutes across stints
        minutes_split = [minutes / n_stints] * n_stints
        
        # Sample game states
        states = _sample_stint_states_worker(n_stints, transition_matrix, rng)
        
        # Get capability vector
        capability_vector = capability_samples[trial]
        
        # Apply state offsets
        adjusted_capability = _apply_state_offsets_worker(
            capability_vector,
            states,
            minutes_split,
            state_offsets
        )
        
        # Project to box statistics
        box_stats_trial = _project_to_box_worker(
            adjusted_capability,
            minutes,
            usage,
            opp_ctx,
            states,
            minutes_split,
            state_offsets
        )
        
        # Store results
        for stat, value in box_stats_trial.items():
            box_stats_samples[stat][trial] = value
    
    # Compute risk metrics
    risk_metrics = _compute_risk_metrics_worker(box_stats_samples)
    
    # Compute hypervolume index
    hypervolume_index = region.hypervolume_above_baseline or 0.0
    
    # Prepare metadata
    metadata = {
        'n_trials': n_trials,
        'n_stints': n_stints,
        'seed': seed,
        'player_role': player_ctx.role,
        'game_id': game_ctx.game_id,
        'opponent_id': opp_ctx.opponent_id
    }
    
    return SimulationResult(
        player_id=player_ctx.player_id,
        distributions=box_stats_samples,
        risk_metrics=risk_metrics,
        hypervolume_index=hypervolume_index,
        metadata=metadata
    )


def _sample_stint_states_worker(
    n_stints: int,
    transition_matrix: np.ndarray,
    rng: np.random.Generator
) -> List['GameState']:
    """Helper function to sample stint states for worker."""
    state_to_idx = {
        GameState.NORMAL: 0,
        GameState.HOT: 1,
        GameState.COLD: 2,
        GameState.FOUL_RISK: 3,
        GameState.WIND_DOWN: 4
    }
    idx_to_state = {v: k for k, v in state_to_idx.items()}
    
    states = []
    current_idx = 0  # Start with NORMAL
    
    for _ in range(n_stints):
        states.append(idx_to_state[current_idx])
        transition_probs = transition_matrix[current_idx]
        current_idx = rng.choice(len(transition_probs), p=transition_probs)
    
    return states


def _apply_state_offsets_worker(
    capability_vector: np.ndarray,
    states: List['GameState'],
    minutes_split: List[float],
    state_offsets: Dict
) -> np.ndarray:
    """Helper function to apply state offsets for worker."""
    adjusted = capability_vector.copy()
    total_minutes = sum(minutes_split)
    
    if total_minutes == 0:
        return adjusted
    
    for state, minutes in zip(states, minutes_split):
        weight = minutes / total_minutes
        
        if state == GameState.HOT:
            offsets = state_offsets.get('Hot', {})
            adjusted[0] *= (1.0 + weight * (offsets.get('scoring_efficiency', 1.10) - 1.0))
            adjusted[1] *= (1.0 + weight * (offsets.get('usage', 1.05) - 1.0))
        elif state == GameState.COLD:
            offsets = state_offsets.get('Cold', {})
            adjusted[0] *= (1.0 + weight * (offsets.get('scoring_efficiency', 0.85) - 1.0))
            adjusted[1] *= (1.0 + weight * (offsets.get('usage', 0.95) - 1.0))
        elif state == GameState.WIND_DOWN:
            offsets = state_offsets.get('WindDown', {})
            adjusted[1] *= (1.0 + weight * (offsets.get('usage', 0.90) - 1.0))
            adjusted[2] *= (1.0 + weight * (offsets.get('assist_rate', 1.10) - 1.0))
    
    return adjusted


def _project_to_box_worker(
    capability_vector: np.ndarray,
    minutes: float,
    usage: float,
    opp_ctx: 'OpponentContext',
    states: List['GameState'],
    minutes_split: List[float],
    state_offsets: Dict
) -> Dict[str, float]:
    """Helper function to project to box stats for worker."""
    ts_pct = capability_vector[0]
    usg_pct = capability_vector[1]
    ast_pct = capability_vector[2]
    tov_pct = capability_vector[3]
    trb_pct = capability_vector[4]
    stl_pct = capability_vector[5]
    blk_pct = capability_vector[6]
    
    team_possessions = (minutes / 48.0) * opp_ctx.pace
    player_poss = team_possessions * usg_pct
    
    tsa = player_poss * 0.75
    pts = ts_pct * 2.0 * tsa
    
    fga = player_poss * 0.70
    three_pa = fga * 0.35
    
    fta_rate = 0.28 * (2.0 - opp_ctx.foul_discipline_index)
    fta = fga * fta_rate
    
    tov = player_poss * tov_pct
    ast = team_possessions * ast_pct * 0.4
    
    trb_opportunities = 50 * (2.0 - opp_ctx.def_reb_strength)
    trb = (minutes / 48.0) * trb_pct * trb_opportunities
    
    stl = team_possessions * stl_pct
    blk = team_possessions * blk_pct
    
    base_foul_rate = 0.15
    foul_multiplier = 1.0
    
    total_minutes = sum(minutes_split)
    if total_minutes > 0:
        for state, state_minutes in zip(states, minutes_split):
            if state == GameState.FOUL_RISK:
                weight = state_minutes / total_minutes
                offsets = state_offsets.get('FoulRisk', {})
                foul_multiplier += weight * (offsets.get('foul_rate', 1.30) - 1.0)
    
    pf = minutes * base_foul_rate * foul_multiplier
    
    return {
        'PTS': pts,
        'TRB': trb,
        'AST': ast,
        'STL': stl,
        'BLK': blk,
        'TOV': tov,
        'FGA': fga,
        '3PA': three_pa,
        'FTA': fta,
        'PF': pf
    }


def _compute_risk_metrics_worker(distributions: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Helper function to compute risk metrics for worker."""
    risk_metrics = {}
    key_stats = ['PTS', 'TRB', 'AST', 'TOV']
    
    for stat in key_stats:
        if stat not in distributions:
            continue
        
        samples = distributions[stat]
        
        var_05 = np.percentile(samples, 5)
        var_95 = np.percentile(samples, 95)
        
        risk_metrics[f'{stat}_VaR_05'] = var_05
        risk_metrics[f'{stat}_VaR_95'] = var_95
        
        cvar_05 = np.mean(samples[samples <= var_05])
        risk_metrics[f'{stat}_CVaR_05'] = cvar_05
        
        median = np.median(samples)
        
        prob_high = np.mean(samples >= 1.5 * median)
        risk_metrics[f'{stat}_prob_high'] = prob_high
        
        prob_low = np.mean(samples <= 0.5 * median)
        risk_metrics[f'{stat}_prob_low'] = prob_low
    
    return risk_metrics


class GameState(Enum):
    """
    Game states for Markov simulation.
    
    States represent different performance contexts during a game:
    - Normal: Standard play
    - Hot: Player is performing above their typical level
    - Cold: Player is performing below their typical level
    - FoulRisk: Player is in foul trouble, playing cautiously
    - WindDown: End-of-game situation with different dynamics
    """
    NORMAL = "Normal"
    HOT = "Hot"
    COLD = "Cold"
    FOUL_RISK = "FoulRisk"
    WIND_DOWN = "WindDown"


@dataclass
class GameContext:
    """
    Context information for a game.
    
    Attributes:
        game_id: Unique identifier for the game
        team_id: Team identifier
        opponent_id: Opponent team identifier
        venue: "home" or "away"
        pace: Expected game pace (possessions per 48 minutes)
    """
    game_id: str
    team_id: str
    opponent_id: str
    venue: str
    pace: float


@dataclass
class OpponentContext:
    """
    Opponent defensive characteristics.
    
    Attributes:
        opponent_id: Opponent team identifier
        scheme_drop_rate: Frequency of drop coverage
        scheme_switch_rate: Frequency of switching
        scheme_ice_rate: Frequency of ice coverage
        blitz_rate: Frequency of blitzing pick-and-roll
        rim_deterrence_index: Rim protection strength
        def_reb_strength: Defensive rebounding strength
        foul_discipline_index: Foul discipline (lower = more fouls)
        pace: Team pace
        help_nail_freq: Help defense frequency
    """
    opponent_id: str
    scheme_drop_rate: float
    scheme_switch_rate: float
    scheme_ice_rate: float
    blitz_rate: float
    rim_deterrence_index: float
    def_reb_strength: float
    foul_discipline_index: float
    pace: float
    help_nail_freq: float


@dataclass
class PlayerContext:
    """
    Player-specific context.
    
    Attributes:
        player_id: Player identifier
        role: Player role ("starter", "rotation", "bench")
        exp_minutes: Expected minutes
        exp_usage: Expected usage rate
    """
    player_id: str
    role: str
    exp_minutes: float
    exp_usage: float


@dataclass
class SimulationResult:
    """
    Results from a player game simulation.
    
    Attributes:
        player_id: Player identifier
        distributions: Dictionary mapping stat names to arrays of samples
        risk_metrics: Dictionary of risk metrics (VaR, CVaR, tail probabilities)
        hypervolume_index: Hypervolume above baseline
        metadata: Additional metadata (seed, n_trials, etc.)
    """
    player_id: str
    distributions: Dict[str, np.ndarray]
    risk_metrics: Dict[str, float]
    hypervolume_index: float
    metadata: Dict[str, Any]


class GlobalSimulator:
    """
    Global Markov-Monte Carlo simulator for player performance.
    
    This simulator:
    1. Samples minutes and usage from role-specific distributions
    2. Samples capability vectors from the player's capability region
    3. Simulates game states using a Markov transition matrix
    4. Applies state-specific offsets to capability vectors
    5. Projects capability vectors to box statistics
    6. Runs N trials to generate probabilistic forecasts
    """
    
    # Box statistics to project
    BOX_STATS = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', '3PA', 'FTA', 'PF']
    
    # Mapping from capability attributes to box stats
    # Capability vector: [TS%, USG%, AST%, TOV%, TRB%, STL%, BLK%]
    CAPABILITY_TO_BOX_MAPPING = {
        'TS%': 0,
        'USG%': 1,
        'AST%': 2,
        'TOV%': 3,
        'TRB%': 4,
        'STL%': 5,
        'BLK%': 6
    }
    
    def __init__(
        self,
        n_trials: int = 20000,
        n_stints: int = 5,
        seed: Optional[int] = None,
        config_path: str = "configs/default.yaml",
        n_workers: Optional[int] = None,
        enable_progress: bool = True
    ):
        """
        Initialize the GlobalSimulator.
        
        Args:
            n_trials: Number of simulation trials (default: 20000)
            n_stints: Number of stints per game for state transitions (default: 5)
            seed: Random seed for reproducibility (default: None)
            config_path: Path to configuration file (default: "configs/default.yaml")
            n_workers: Number of parallel workers (None = use config/env, 1 = no parallelization)
            enable_progress: Whether to show progress bars (default: True)
        """
        self.n_trials = n_trials
        self.n_stints = n_stints
        self.seed = seed
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract simulation parameters
        sim_config = self.config.get('simulation', {})
        
        # State transition matrix
        # Rows: current state, Columns: next state
        # Order: Normal, Hot, Cold, FoulRisk, WindDown
        state_transitions = sim_config.get('state_transitions', {})
        self.transition_matrix = np.array([
            state_transitions.get('Normal', [0.70, 0.15, 0.10, 0.03, 0.02]),
            state_transitions.get('Hot', [0.40, 0.50, 0.05, 0.03, 0.02]),
            state_transitions.get('Cold', [0.50, 0.10, 0.35, 0.03, 0.02]),
            state_transitions.get('FoulRisk', [0.60, 0.10, 0.10, 0.15, 0.05]),
            state_transitions.get('WindDown', [0.30, 0.05, 0.05, 0.05, 0.55])
        ])
        
        # State offsets
        self.state_offsets = sim_config.get('state_offsets', {})
        
        # Role parameters
        self.role_params = sim_config.get('role_params', {})
        
        # Initialize region builder
        self.region_builder = RegionBuilder()
        
        # Configure parallelization
        parallel_config = self.config.get('parallelization', {})
        
        # Determine number of workers
        if n_workers is not None:
            self.n_workers = n_workers
        else:
            # Try environment variable first
            env_workers = os.environ.get('NBA_PRED_N_WORKERS')
            if env_workers is not None:
                self.n_workers = int(env_workers)
            else:
                # Use config value
                config_workers = parallel_config.get('n_workers')
                if config_workers is None:
                    # Use all available cores
                    self.n_workers = cpu_count()
                else:
                    self.n_workers = config_workers
        
        # Enable progress bars
        if enable_progress is not None:
            self.enable_progress = enable_progress
        else:
            self.enable_progress = parallel_config.get('enable_progress_bars', True)
    
    def sample_minutes(
        self,
        player_ctx: PlayerContext,
        game_ctx: GameContext,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Sample player minutes from role-specific distribution.
        
        Uses a truncated normal distribution based on player role.
        
        Args:
            player_ctx: Player context with role information
            game_ctx: Game context
            rng: Random number generator (uses self.seed if None)
        
        Returns:
            Sampled minutes (clipped to [0, 48])
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)
        
        # Get role parameters
        role = player_ctx.role
        role_config = self.role_params.get(role, self.role_params.get('rotation', {}))
        
        minutes_mean = role_config.get('minutes_mean', player_ctx.exp_minutes)
        minutes_sigma = role_config.get('minutes_sigma', 5.0)
        
        # Sample from normal distribution
        minutes = rng.normal(minutes_mean, minutes_sigma)
        
        # Clip to valid range [0, 48]
        minutes = np.clip(minutes, 0.0, 48.0)
        
        return minutes
    
    def sample_usage(
        self,
        player_ctx: PlayerContext,
        minutes: float,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Sample player usage rate from Beta distribution.
        
        Usage rate represents the percentage of team possessions used by the player
        while on the court. Uses Beta distribution for bounded [0, 1] support.
        
        Args:
            player_ctx: Player context with role information
            minutes: Sampled minutes for the game
            rng: Random number generator
        
        Returns:
            Sampled usage rate (between 0 and 1)
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)
        
        # Get role parameters
        role = player_ctx.role
        role_config = self.role_params.get(role, self.role_params.get('rotation', {}))
        
        usage_alpha = role_config.get('usage_alpha', 5.0)
        usage_beta = role_config.get('usage_beta', 15.0)
        
        # Sample from Beta distribution
        # Beta(alpha, beta) has mean = alpha / (alpha + beta)
        usage = rng.beta(usage_alpha, usage_beta)
        
        # Adjust for minutes (lower minutes -> slightly lower usage)
        # This is a simple heuristic
        if minutes < 20:
            usage = usage * (0.8 + 0.2 * minutes / 20)
        
        return usage
    
    def sample_stint_states(
        self,
        n_stints: int,
        initial_state: GameState = GameState.NORMAL,
        rng: Optional[np.random.Generator] = None
    ) -> List[GameState]:
        """
        Sample game states for each stint using Markov transition matrix.
        
        Args:
            n_stints: Number of stints to simulate
            initial_state: Starting state (default: Normal)
            rng: Random number generator
        
        Returns:
            List of GameState for each stint
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)
        
        # State index mapping
        state_to_idx = {
            GameState.NORMAL: 0,
            GameState.HOT: 1,
            GameState.COLD: 2,
            GameState.FOUL_RISK: 3,
            GameState.WIND_DOWN: 4
        }
        idx_to_state = {v: k for k, v in state_to_idx.items()}
        
        # Initialize states
        states = []
        current_idx = state_to_idx[initial_state]
        
        for _ in range(n_stints):
            # Record current state
            states.append(idx_to_state[current_idx])
            
            # Sample next state from transition probabilities
            transition_probs = self.transition_matrix[current_idx]
            current_idx = rng.choice(len(transition_probs), p=transition_probs)
        
        return states
    
    def apply_state_offsets(
        self,
        capability_vector: np.ndarray,
        states: List[GameState],
        minutes_split: List[float]
    ) -> np.ndarray:
        """
        Apply state-specific offsets to capability vector.
        
        Each stint has a different state, and we apply weighted offsets based on
        the time spent in each state.
        
        Args:
            capability_vector: Base capability vector [TS%, USG%, AST%, TOV%, TRB%, STL%, BLK%]
            states: List of GameState for each stint
            minutes_split: Minutes played in each stint
        
        Returns:
            Adjusted capability vector
        """
        # Start with base capability
        adjusted = capability_vector.copy()
        
        # Compute total minutes
        total_minutes = sum(minutes_split)
        
        if total_minutes == 0:
            return adjusted
        
        # Apply weighted offsets for each state
        for state, minutes in zip(states, minutes_split):
            weight = minutes / total_minutes
            
            if state == GameState.HOT:
                # Hot: +10% scoring efficiency, +5% usage
                offsets = self.state_offsets.get('Hot', {})
                adjusted[0] *= (1.0 + weight * (offsets.get('scoring_efficiency', 1.10) - 1.0))
                adjusted[1] *= (1.0 + weight * (offsets.get('usage', 1.05) - 1.0))
            
            elif state == GameState.COLD:
                # Cold: -15% scoring efficiency, -5% usage
                offsets = self.state_offsets.get('Cold', {})
                adjusted[0] *= (1.0 + weight * (offsets.get('scoring_efficiency', 0.85) - 1.0))
                adjusted[1] *= (1.0 + weight * (offsets.get('usage', 0.95) - 1.0))
            
            elif state == GameState.FOUL_RISK:
                # FoulRisk: +30% foul rate (affects PF in projection)
                # This is handled in project_to_box
                pass
            
            elif state == GameState.WIND_DOWN:
                # WindDown: -10% usage, +10% assist rate
                offsets = self.state_offsets.get('WindDown', {})
                adjusted[1] *= (1.0 + weight * (offsets.get('usage', 0.90) - 1.0))
                adjusted[2] *= (1.0 + weight * (offsets.get('assist_rate', 1.10) - 1.0))
        
        return adjusted
    
    def project_to_box(
        self,
        capability_vector: np.ndarray,
        minutes: float,
        usage: float,
        opp_ctx: OpponentContext,
        states: List[GameState],
        minutes_split: List[float]
    ) -> Dict[str, float]:
        """
        Project capability vector to box statistics.
        
        Converts capability attributes (TS%, USG%, AST%, etc.) to counting stats
        (PTS, REB, AST, etc.) based on minutes, usage, and opponent context.
        
        Args:
            capability_vector: Capability vector [TS%, USG%, AST%, TOV%, TRB%, STL%, BLK%]
            minutes: Minutes played
            usage: Usage rate
            opp_ctx: Opponent context
            states: List of game states
            minutes_split: Minutes in each state
        
        Returns:
            Dictionary of box statistics
        """
        # Extract capability attributes
        ts_pct = capability_vector[0]
        usg_pct = capability_vector[1]
        ast_pct = capability_vector[2]
        tov_pct = capability_vector[3]
        trb_pct = capability_vector[4]
        stl_pct = capability_vector[5]
        blk_pct = capability_vector[6]
        
        # Estimate team possessions
        # Possessions = (minutes / 48) * pace
        team_possessions = (minutes / 48.0) * opp_ctx.pace
        
        # Player possessions used
        # USG% represents the percentage of team possessions used while on court
        player_poss = team_possessions * usg_pct
        
        # Points calculation
        # TS% = PTS / (2 * TSA), where TSA = FGA + 0.44 * FTA
        # For a typical player: TSA ≈ player_poss * 0.75 (some possessions are turnovers)
        # So: PTS = TS% * 2 * TSA = TS% * 2 * player_poss * 0.75
        tsa = player_poss * 0.75
        pts = ts_pct * 2.0 * tsa
        
        # Field goal attempts
        # Assume 70% of player possessions result in FGA (rest are FTA or TOV)
        fga = player_poss * 0.70
        
        # Three-point attempts (simplified, assume 35% of FGA are 3PA)
        three_pa = fga * 0.35
        
        # Free throw attempts
        # FTA rate varies by opponent foul discipline
        # Typical FTA/FGA ratio is around 0.25-0.30
        fta_rate = 0.28 * (2.0 - opp_ctx.foul_discipline_index)
        fta = fga * fta_rate
        
        # Turnovers
        tov = player_poss * tov_pct
        
        # Assists
        # AST% = percentage of teammate FG made while on court that player assisted
        # Approximate: AST = team_poss * AST% * 0.4 (scaling factor)
        ast = team_possessions * ast_pct * 0.4
        
        # Rebounds
        # TRB% = percentage of available rebounds grabbed while on court
        # Total rebounds per game ≈ 100 (both teams)
        # Player's share ≈ (minutes / 48) * TRB% * 50
        trb_opportunities = 50 * (2.0 - opp_ctx.def_reb_strength)
        trb = (minutes / 48.0) * trb_pct * trb_opportunities
        
        # Steals
        # STL% = percentage of opponent possessions that end in steal while on court
        stl = team_possessions * stl_pct
        
        # Blocks
        # BLK% = percentage of opponent 2PA blocked while on court
        blk = team_possessions * blk_pct
        
        # Personal fouls
        # Base foul rate, adjusted by FoulRisk state
        base_foul_rate = 0.15  # fouls per minute
        foul_multiplier = 1.0
        
        total_minutes = sum(minutes_split)
        if total_minutes > 0:
            for state, state_minutes in zip(states, minutes_split):
                if state == GameState.FOUL_RISK:
                    weight = state_minutes / total_minutes
                    offsets = self.state_offsets.get('FoulRisk', {})
                    foul_multiplier += weight * (offsets.get('foul_rate', 1.30) - 1.0)
        
        pf = minutes * base_foul_rate * foul_multiplier
        
        return {
            'PTS': pts,
            'TRB': trb,
            'AST': ast,
            'STL': stl,
            'BLK': blk,
            'TOV': tov,
            'FGA': fga,
            '3PA': three_pa,
            'FTA': fta,
            'PF': pf
        }

    
    def simulate_player_game(
        self,
        region: CapabilityRegion,
        player_ctx: PlayerContext,
        game_ctx: GameContext,
        opp_ctx: OpponentContext,
        N: Optional[int] = None,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Simulate a player's game performance over N trials.
        
        This is the main simulation method that:
        1. Samples minutes and usage for each trial
        2. Samples capability vectors from the region
        3. Simulates game states
        4. Applies state offsets
        5. Projects to box statistics
        6. Computes distributions and risk metrics
        
        Args:
            region: Player's capability region
            player_ctx: Player context
            game_ctx: Game context
            opp_ctx: Opponent context
            N: Number of trials (uses self.n_trials if None)
            seed: Random seed (uses self.seed if None)
        
        Returns:
            SimulationResult with distributions and metrics
        """
        if N is None:
            N = self.n_trials
        
        if seed is None:
            seed = self.seed
        
        # Initialize random number generator
        rng = np.random.default_rng(seed)
        
        # Sample capability vectors from region
        # Use a different seed for region sampling to ensure independence
        region_seed = rng.integers(0, 2**31) if seed is not None else None
        capability_samples = self.region_builder.sample_region(
            region,
            n=N,
            seed=region_seed
        )
        
        # Initialize storage for box stats
        box_stats_samples = {stat: np.zeros(N) for stat in self.BOX_STATS}
        
        # Run N trials
        for trial in range(N):
            # Sample minutes and usage
            minutes = self.sample_minutes(player_ctx, game_ctx, rng)
            usage = self.sample_usage(player_ctx, minutes, rng)
            
            # Split minutes across stints (equal split for simplicity)
            minutes_split = [minutes / self.n_stints] * self.n_stints
            
            # Sample game states
            states = self.sample_stint_states(self.n_stints, rng=rng)
            
            # Get capability vector for this trial
            capability_vector = capability_samples[trial]
            
            # Apply state offsets
            adjusted_capability = self.apply_state_offsets(
                capability_vector,
                states,
                minutes_split
            )
            
            # Project to box statistics
            box_stats = self.project_to_box(
                adjusted_capability,
                minutes,
                usage,
                opp_ctx,
                states,
                minutes_split
            )
            
            # Store results
            for stat, value in box_stats.items():
                box_stats_samples[stat][trial] = value
        
        # Compute risk metrics
        risk_metrics = self._compute_risk_metrics(box_stats_samples)
        
        # Compute hypervolume index
        hypervolume_index = region.hypervolume_above_baseline or 0.0
        
        # Prepare metadata
        metadata = {
            'n_trials': N,
            'n_stints': self.n_stints,
            'seed': seed,
            'player_role': player_ctx.role,
            'game_id': game_ctx.game_id,
            'opponent_id': opp_ctx.opponent_id
        }
        
        return SimulationResult(
            player_id=player_ctx.player_id,
            distributions=box_stats_samples,
            risk_metrics=risk_metrics,
            hypervolume_index=hypervolume_index,
            metadata=metadata
        )
    
    def _compute_risk_metrics(
        self,
        distributions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute risk metrics from simulation distributions.
        
        Computes:
        - VaR (Value at Risk) at 5th and 95th percentiles
        - CVaR (Conditional Value at Risk) - expected value in tail
        - Tail probabilities for key stats
        
        Args:
            distributions: Dictionary of stat distributions
        
        Returns:
            Dictionary of risk metrics
        """
        risk_metrics = {}
        
        # Focus on key stats for risk metrics
        key_stats = ['PTS', 'TRB', 'AST', 'TOV']
        
        for stat in key_stats:
            if stat not in distributions:
                continue
            
            samples = distributions[stat]
            
            # VaR (Value at Risk)
            var_05 = np.percentile(samples, 5)
            var_95 = np.percentile(samples, 95)
            
            risk_metrics[f'{stat}_VaR_05'] = var_05
            risk_metrics[f'{stat}_VaR_95'] = var_95
            
            # CVaR (Conditional Value at Risk) - expected value below 5th percentile
            cvar_05 = np.mean(samples[samples <= var_05])
            risk_metrics[f'{stat}_CVaR_05'] = cvar_05
            
            # Tail probabilities
            median = np.median(samples)
            
            # Probability of exceeding 1.5x median
            prob_high = np.mean(samples >= 1.5 * median)
            risk_metrics[f'{stat}_prob_high'] = prob_high
            
            # Probability of falling below 0.5x median
            prob_low = np.mean(samples <= 0.5 * median)
            risk_metrics[f'{stat}_prob_low'] = prob_low
        
        return risk_metrics
    
    def simulate_multiple_players(
        self,
        players: List[Tuple[CapabilityRegion, PlayerContext]],
        game_ctx: GameContext,
        opp_ctx: OpponentContext,
        N: Optional[int] = None,
        seed: Optional[int] = None,
        parallel: bool = True
    ) -> List[SimulationResult]:
        """
        Simulate multiple players for the same game.
        
        Supports parallel execution across players for improved performance.
        
        Args:
            players: List of (region, player_ctx) tuples
            game_ctx: Game context
            opp_ctx: Opponent context
            N: Number of trials per player
            seed: Random seed
            parallel: Whether to use parallel processing (default: True)
        
        Returns:
            List of SimulationResult, one per player
        """
        # Use different seeds for each player to ensure independence
        rng = np.random.default_rng(seed)
        player_seeds = [
            rng.integers(0, 2**31) if seed is not None else None
            for _ in range(len(players))
        ]
        
        # Decide whether to parallelize
        use_parallel = parallel and self.n_workers > 1 and len(players) > 1
        
        if use_parallel:
            # Parallel execution
            results = self._simulate_players_parallel(
                players=players,
                game_ctx=game_ctx,
                opp_ctx=opp_ctx,
                N=N,
                seeds=player_seeds
            )
        else:
            # Sequential execution
            results = []
            
            # Create progress bar if enabled
            iterator = enumerate(players)
            if self.enable_progress:
                iterator = tqdm(
                    iterator,
                    total=len(players),
                    desc="Simulating players",
                    unit="player"
                )
            
            for i, (region, player_ctx) in iterator:
                result = self.simulate_player_game(
                    region=region,
                    player_ctx=player_ctx,
                    game_ctx=game_ctx,
                    opp_ctx=opp_ctx,
                    N=N,
                    seed=player_seeds[i]
                )
                
                results.append(result)
        
        return results
    
    def _simulate_players_parallel(
        self,
        players: List[Tuple[CapabilityRegion, PlayerContext]],
        game_ctx: GameContext,
        opp_ctx: OpponentContext,
        N: Optional[int],
        seeds: List[Optional[int]]
    ) -> List[SimulationResult]:
        """
        Simulate multiple players in parallel using multiprocessing.
        
        Args:
            players: List of (region, player_ctx) tuples
            game_ctx: Game context
            opp_ctx: Opponent context
            N: Number of trials per player
            seeds: List of random seeds for each player
        
        Returns:
            List of SimulationResult, one per player
        """
        # Prepare arguments for parallel execution
        args_list = [
            (
                region,
                player_ctx,
                game_ctx,
                opp_ctx,
                N if N is not None else self.n_trials,
                seed,
                self.n_stints,
                self.transition_matrix,
                self.state_offsets,
                self.role_params
            )
            for (region, player_ctx), seed in zip(players, seeds)
        ]
        
        # Use multiprocessing pool
        with Pool(processes=self.n_workers) as pool:
            if self.enable_progress:
                # Use imap with tqdm for progress tracking
                results = list(tqdm(
                    pool.imap(_simulate_player_worker, args_list),
                    total=len(args_list),
                    desc="Simulating players (parallel)",
                    unit="player"
                ))
            else:
                # Use map without progress tracking
                results = pool.map(_simulate_player_worker, args_list)
        
        return results
    
    def get_summary_statistics(
        self,
        result: SimulationResult
    ) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics from simulation result.
        
        Computes mean, median, std, and percentiles for each stat.
        
        Args:
            result: SimulationResult
        
        Returns:
            Dictionary mapping stat names to summary statistics
        """
        summary = {}
        
        for stat, samples in result.distributions.items():
            summary[stat] = {
                'mean': np.mean(samples),
                'median': np.median(samples),
                'std': np.std(samples),
                'min': np.min(samples),
                'max': np.max(samples),
                'p10': np.percentile(samples, 10),
                'p25': np.percentile(samples, 25),
                'p75': np.percentile(samples, 75),
                'p90': np.percentile(samples, 90)
            }
        
        return summary
