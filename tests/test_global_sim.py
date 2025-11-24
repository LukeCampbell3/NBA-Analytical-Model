"""
Unit tests for global simulation module.
"""

import pytest
import numpy as np
import yaml
import tempfile
import os

from src.simulation.global_sim import (
    GameState,
    GameContext,
    OpponentContext,
    PlayerContext,
    SimulationResult,
    GlobalSimulator
)
from src.regions.build import (
    Ellipsoid,
    HPolytope,
    CapabilityRegion,
    RegionBuilder
)
from src.frontiers.fit import Halfspace


class TestGameState:
    """Tests for GameState enum."""
    
    def test_game_state_values(self):
        """Test that all game states are defined."""
        assert GameState.NORMAL.value == "Normal"
        assert GameState.HOT.value == "Hot"
        assert GameState.COLD.value == "Cold"
        assert GameState.FOUL_RISK.value == "FoulRisk"
        assert GameState.WIND_DOWN.value == "WindDown"
    
    def test_game_state_count(self):
        """Test that we have exactly 5 states."""
        assert len(GameState) == 5


class TestDataClasses:
    """Tests for data classes."""
    
    def test_game_context_creation(self):
        """Test GameContext creation."""
        ctx = GameContext(
            game_id="TEST_001",
            team_id="GSW",
            opponent_id="LAL",
            venue="home",
            pace=100.5
        )
        
        assert ctx.game_id == "TEST_001"
        assert ctx.team_id == "GSW"
        assert ctx.opponent_id == "LAL"
        assert ctx.venue == "home"
        assert ctx.pace == 100.5
    
    def test_opponent_context_creation(self):
        """Test OpponentContext creation."""
        ctx = OpponentContext(
            opponent_id="LAL",
            scheme_drop_rate=0.4,
            scheme_switch_rate=0.3,
            scheme_ice_rate=0.2,
            blitz_rate=0.15,
            rim_deterrence_index=1.2,
            def_reb_strength=1.1,
            foul_discipline_index=1.0,
            pace=100.0,
            help_nail_freq=0.5
        )
        
        assert ctx.opponent_id == "LAL"
        assert ctx.blitz_rate == 0.15
        assert ctx.pace == 100.0
    
    def test_player_context_creation(self):
        """Test PlayerContext creation."""
        ctx = PlayerContext(
            player_id="curry_stephen",
            role="starter",
            exp_minutes=33.0,
            exp_usage=0.30
        )
        
        assert ctx.player_id == "curry_stephen"
        assert ctx.role == "starter"
        assert ctx.exp_minutes == 33.0
        assert ctx.exp_usage == 0.30


class TestGlobalSimulator:
    """Tests for GlobalSimulator class."""
    
    @pytest.fixture
    def config_file(self):
        """Create temporary config file for testing."""
        config = {
            'simulation': {
                'n_trials': 1000,
                'n_stints': 5,
                'seed': None,
                'state_transitions': {
                    'Normal': [0.70, 0.15, 0.10, 0.03, 0.02],
                    'Hot': [0.40, 0.50, 0.05, 0.03, 0.02],
                    'Cold': [0.50, 0.10, 0.35, 0.03, 0.02],
                    'FoulRisk': [0.60, 0.10, 0.10, 0.15, 0.05],
                    'WindDown': [0.30, 0.05, 0.05, 0.05, 0.55]
                },
                'state_offsets': {
                    'Hot': {
                        'scoring_efficiency': 1.10,
                        'usage': 1.05
                    },
                    'Cold': {
                        'scoring_efficiency': 0.85,
                        'usage': 0.95
                    },
                    'FoulRisk': {
                        'minutes': 0.80,
                        'foul_rate': 1.30
                    },
                    'WindDown': {
                        'usage': 0.90,
                        'assist_rate': 1.10
                    }
                },
                'role_params': {
                    'starter': {
                        'minutes_mean': 33.0,
                        'minutes_sigma': 3.0,
                        'usage_alpha': 5.0,
                        'usage_beta': 15.0
                    },
                    'rotation': {
                        'minutes_mean': 22.0,
                        'minutes_sigma': 5.0,
                        'usage_alpha': 4.0,
                        'usage_beta': 20.0
                    },
                    'bench': {
                        'minutes_mean': 12.0,
                        'minutes_sigma': 6.0,
                        'usage_alpha': 3.0,
                        'usage_beta': 25.0
                    }
                }
            }
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    def simulator(self, config_file):
        """Create GlobalSimulator for testing."""
        return GlobalSimulator(
            n_trials=100,
            n_stints=5,
            seed=42,
            config_path=config_file
        )
    
    @pytest.fixture
    def simple_region(self):
        """Create simple capability region for testing."""
        builder = RegionBuilder()
        
        # 7D capability vector: [TS%, USG%, AST%, TOV%, TRB%, STL%, BLK%]
        mu = np.array([0.55, 0.25, 0.20, 0.12, 0.10, 0.015, 0.015])
        Sigma = np.eye(7) * 0.01  # Small variance
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
        polytope = HPolytope(halfspaces=[], dimension=7)
        
        return builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
    
    @pytest.fixture
    def player_ctx(self):
        """Create player context for testing."""
        return PlayerContext(
            player_id="test_player",
            role="starter",
            exp_minutes=33.0,
            exp_usage=0.25
        )
    
    @pytest.fixture
    def game_ctx(self):
        """Create game context for testing."""
        return GameContext(
            game_id="TEST_001",
            team_id="GSW",
            opponent_id="LAL",
            venue="home",
            pace=100.0
        )
    
    @pytest.fixture
    def opp_ctx(self):
        """Create opponent context for testing."""
        return OpponentContext(
            opponent_id="LAL",
            scheme_drop_rate=0.4,
            scheme_switch_rate=0.3,
            scheme_ice_rate=0.2,
            blitz_rate=0.15,
            rim_deterrence_index=1.2,
            def_reb_strength=1.1,
            foul_discipline_index=1.0,
            pace=100.0,
            help_nail_freq=0.5
        )
    
    def test_simulator_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.n_trials == 100
        assert simulator.n_stints == 5
        assert simulator.seed == 42
        assert simulator.transition_matrix.shape == (5, 5)
    
    def test_transition_matrix_valid(self, simulator):
        """Test that transition matrix rows sum to 1."""
        for row in simulator.transition_matrix:
            assert np.isclose(np.sum(row), 1.0)
    
    def test_sample_minutes_basic(self, simulator, player_ctx, game_ctx):
        """Test basic minutes sampling."""
        rng = np.random.default_rng(42)
        
        minutes = simulator.sample_minutes(player_ctx, game_ctx, rng)
        
        assert 0 <= minutes <= 48
        assert isinstance(minutes, (float, np.floating))
    
    def test_sample_minutes_role_specific(self, simulator, game_ctx):
        """Test that different roles get different minutes."""
        rng = np.random.default_rng(42)
        
        starter = PlayerContext("p1", "starter", 33.0, 0.25)
        bench = PlayerContext("p2", "bench", 12.0, 0.15)
        
        starter_minutes = []
        bench_minutes = []
        
        for _ in range(50):
            starter_minutes.append(simulator.sample_minutes(starter, game_ctx, rng))
            bench_minutes.append(simulator.sample_minutes(bench, game_ctx, rng))
        
        # Starters should average more minutes
        assert np.mean(starter_minutes) > np.mean(bench_minutes)
    
    def test_sample_minutes_reproducibility(self, simulator, player_ctx, game_ctx):
        """Test that sampling is reproducible with same seed."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        
        minutes1 = simulator.sample_minutes(player_ctx, game_ctx, rng1)
        minutes2 = simulator.sample_minutes(player_ctx, game_ctx, rng2)
        
        assert minutes1 == minutes2
    
    def test_sample_usage_basic(self, simulator, player_ctx):
        """Test basic usage sampling."""
        rng = np.random.default_rng(42)
        
        usage = simulator.sample_usage(player_ctx, 33.0, rng)
        
        assert 0 <= usage <= 1
        assert isinstance(usage, (float, np.floating))
    
    def test_sample_usage_role_specific(self, simulator):
        """Test that different roles get different usage."""
        rng = np.random.default_rng(42)
        
        starter = PlayerContext("p1", "starter", 33.0, 0.25)
        bench = PlayerContext("p2", "bench", 12.0, 0.15)
        
        starter_usage = []
        bench_usage = []
        
        for _ in range(50):
            starter_usage.append(simulator.sample_usage(starter, 33.0, rng))
            bench_usage.append(simulator.sample_usage(bench, 12.0, rng))
        
        # Starters should average higher usage
        assert np.mean(starter_usage) > np.mean(bench_usage)
    
    def test_sample_stint_states_basic(self, simulator):
        """Test basic stint state sampling."""
        rng = np.random.default_rng(42)
        
        states = simulator.sample_stint_states(5, rng=rng)
        
        assert len(states) == 5
        assert all(isinstance(s, GameState) for s in states)
    
    def test_sample_stint_states_initial_state(self, simulator):
        """Test that initial state is respected."""
        rng = np.random.default_rng(42)
        
        states = simulator.sample_stint_states(
            5,
            initial_state=GameState.HOT,
            rng=rng
        )
        
        assert states[0] == GameState.HOT
    
    def test_sample_stint_states_reproducibility(self, simulator):
        """Test that state sampling is reproducible."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        
        states1 = simulator.sample_stint_states(5, rng=rng1)
        states2 = simulator.sample_stint_states(5, rng=rng2)
        
        assert states1 == states2
    
    def test_apply_state_offsets_normal(self, simulator):
        """Test state offsets with all normal states."""
        capability = np.array([0.55, 0.25, 0.20, 0.12, 0.10, 0.015, 0.015])
        states = [GameState.NORMAL] * 5
        minutes_split = [6.0] * 5
        
        adjusted = simulator.apply_state_offsets(capability, states, minutes_split)
        
        # Should be unchanged for all normal states
        np.testing.assert_array_almost_equal(adjusted, capability)
    
    def test_apply_state_offsets_hot(self, simulator):
        """Test state offsets with hot state."""
        capability = np.array([0.55, 0.25, 0.20, 0.12, 0.10, 0.015, 0.015])
        states = [GameState.HOT] * 5
        minutes_split = [6.0] * 5
        
        adjusted = simulator.apply_state_offsets(capability, states, minutes_split)
        
        # TS% and USG% should increase
        assert adjusted[0] > capability[0]  # TS%
        assert adjusted[1] > capability[1]  # USG%
    
    def test_apply_state_offsets_cold(self, simulator):
        """Test state offsets with cold state."""
        capability = np.array([0.55, 0.25, 0.20, 0.12, 0.10, 0.015, 0.015])
        states = [GameState.COLD] * 5
        minutes_split = [6.0] * 5
        
        adjusted = simulator.apply_state_offsets(capability, states, minutes_split)
        
        # TS% and USG% should decrease
        assert adjusted[0] < capability[0]  # TS%
        assert adjusted[1] < capability[1]  # USG%
    
    def test_apply_state_offsets_wind_down(self, simulator):
        """Test state offsets with wind down state."""
        capability = np.array([0.55, 0.25, 0.20, 0.12, 0.10, 0.015, 0.015])
        states = [GameState.WIND_DOWN] * 5
        minutes_split = [6.0] * 5
        
        adjusted = simulator.apply_state_offsets(capability, states, minutes_split)
        
        # USG% should decrease, AST% should increase
        assert adjusted[1] < capability[1]  # USG%
        assert adjusted[2] > capability[2]  # AST%
    
    def test_project_to_box_basic(self, simulator, opp_ctx):
        """Test basic box stat projection."""
        capability = np.array([0.55, 0.25, 0.20, 0.12, 0.10, 0.015, 0.015])
        minutes = 30.0
        usage = 0.25
        states = [GameState.NORMAL] * 5
        minutes_split = [6.0] * 5
        
        box_stats = simulator.project_to_box(
            capability,
            minutes,
            usage,
            opp_ctx,
            states,
            minutes_split
        )
        
        # Check that all expected stats are present
        for stat in simulator.BOX_STATS:
            assert stat in box_stats
            assert box_stats[stat] >= 0
    
    def test_project_to_box_reasonable_values(self, simulator, opp_ctx):
        """Test that projected values are reasonable."""
        capability = np.array([0.55, 0.25, 0.20, 0.12, 0.10, 0.015, 0.015])
        minutes = 36.0
        usage = 0.30
        states = [GameState.NORMAL] * 5
        minutes_split = [7.2] * 5
        
        box_stats = simulator.project_to_box(
            capability,
            minutes,
            usage,
            opp_ctx,
            states,
            minutes_split
        )
        
        # Points should be reasonable for a high-usage player
        assert 10 <= box_stats['PTS'] <= 50
        
        # Rebounds should be reasonable
        assert 0 <= box_stats['TRB'] <= 20
        
        # Assists should be reasonable
        assert 0 <= box_stats['AST'] <= 15
        
        # Fouls should be reasonable
        assert 0 <= box_stats['PF'] <= 6
    
    def test_project_to_box_foul_risk_state(self, simulator, opp_ctx):
        """Test that foul risk state increases fouls."""
        capability = np.array([0.55, 0.25, 0.20, 0.12, 0.10, 0.015, 0.015])
        minutes = 30.0
        usage = 0.25
        
        # Normal states
        states_normal = [GameState.NORMAL] * 5
        minutes_split = [6.0] * 5
        
        box_normal = simulator.project_to_box(
            capability, minutes, usage, opp_ctx, states_normal, minutes_split
        )
        
        # Foul risk states
        states_foul = [GameState.FOUL_RISK] * 5
        
        box_foul = simulator.project_to_box(
            capability, minutes, usage, opp_ctx, states_foul, minutes_split
        )
        
        # Fouls should be higher in foul risk state
        assert box_foul['PF'] > box_normal['PF']
    
    def test_simulate_player_game_basic(
        self, simulator, simple_region, player_ctx, game_ctx, opp_ctx
    ):
        """Test basic player game simulation."""
        result = simulator.simulate_player_game(
            region=simple_region,
            player_ctx=player_ctx,
            game_ctx=game_ctx,
            opp_ctx=opp_ctx,
            N=50,
            seed=42
        )
        
        assert isinstance(result, SimulationResult)
        assert result.player_id == "test_player"
        assert len(result.distributions) == len(simulator.BOX_STATS)
        
        # Check that each distribution has correct number of samples
        for stat, samples in result.distributions.items():
            assert len(samples) == 50
    
    def test_simulate_player_game_reproducibility(
        self, simulator, simple_region, player_ctx, game_ctx, opp_ctx
    ):
        """Test that simulation is reproducible with same seed."""
        result1 = simulator.simulate_player_game(
            region=simple_region,
            player_ctx=player_ctx,
            game_ctx=game_ctx,
            opp_ctx=opp_ctx,
            N=50,
            seed=123
        )
        
        result2 = simulator.simulate_player_game(
            region=simple_region,
            player_ctx=player_ctx,
            game_ctx=game_ctx,
            opp_ctx=opp_ctx,
            N=50,
            seed=123
        )
        
        # Distributions should be identical
        for stat in simulator.BOX_STATS:
            np.testing.assert_array_almost_equal(
                result1.distributions[stat],
                result2.distributions[stat]
            )
    
    def test_simulate_player_game_risk_metrics(
        self, simulator, simple_region, player_ctx, game_ctx, opp_ctx
    ):
        """Test that risk metrics are computed."""
        result = simulator.simulate_player_game(
            region=simple_region,
            player_ctx=player_ctx,
            game_ctx=game_ctx,
            opp_ctx=opp_ctx,
            N=50,
            seed=42
        )
        
        # Should have risk metrics
        assert len(result.risk_metrics) > 0
        
        # Check for expected metrics
        assert 'PTS_VaR_05' in result.risk_metrics
        assert 'PTS_VaR_95' in result.risk_metrics
        assert 'PTS_CVaR_05' in result.risk_metrics
    
    def test_simulate_player_game_metadata(
        self, simulator, simple_region, player_ctx, game_ctx, opp_ctx
    ):
        """Test that metadata is included."""
        result = simulator.simulate_player_game(
            region=simple_region,
            player_ctx=player_ctx,
            game_ctx=game_ctx,
            opp_ctx=opp_ctx,
            N=50,
            seed=42
        )
        
        assert 'n_trials' in result.metadata
        assert result.metadata['n_trials'] == 50
        assert 'seed' in result.metadata
        assert result.metadata['seed'] == 42
        assert 'game_id' in result.metadata
        assert result.metadata['game_id'] == "TEST_001"
    
    def test_get_summary_statistics(
        self, simulator, simple_region, player_ctx, game_ctx, opp_ctx
    ):
        """Test summary statistics computation."""
        result = simulator.simulate_player_game(
            region=simple_region,
            player_ctx=player_ctx,
            game_ctx=game_ctx,
            opp_ctx=opp_ctx,
            N=100,
            seed=42
        )
        
        summary = simulator.get_summary_statistics(result)
        
        # Should have summary for each stat
        assert len(summary) == len(simulator.BOX_STATS)
        
        # Check PTS summary
        pts_summary = summary['PTS']
        assert 'mean' in pts_summary
        assert 'median' in pts_summary
        assert 'std' in pts_summary
        assert 'p10' in pts_summary
        assert 'p90' in pts_summary
        
        # Mean should be positive
        assert pts_summary['mean'] > 0
        
        # Percentiles should be ordered
        assert pts_summary['p10'] <= pts_summary['p25']
        assert pts_summary['p25'] <= pts_summary['median']
        assert pts_summary['median'] <= pts_summary['p75']
        assert pts_summary['p75'] <= pts_summary['p90']
