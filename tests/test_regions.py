"""
Unit tests for capability region construction module.
"""

import pytest
import numpy as np
from scipy.stats import chi2

from src.regions.build import (
    Ellipsoid, HPolytope, CapabilityRegion, RegionBuilder
)
from src.frontiers.fit import Halfspace, FrontierModel


class TestEllipsoid:
    """Tests for Ellipsoid class."""
    
    def test_ellipsoid_creation(self):
        """Test basic ellipsoid creation."""
        center = np.array([1.0, 2.0])
        shape_matrix = np.eye(2)
        alpha = 0.80
        
        ellipsoid = Ellipsoid(
            center=center,
            shape_matrix=shape_matrix,
            alpha=alpha,
            dimension=2
        )
        
        assert ellipsoid.dimension == 2
        assert ellipsoid.alpha == 0.80
        np.testing.assert_array_equal(ellipsoid.center, center)
        np.testing.assert_array_equal(ellipsoid.shape_matrix, shape_matrix)


class TestHPolytope:
    """Tests for HPolytope class."""
    
    def test_polytope_creation(self):
        """Test basic polytope creation."""
        halfspaces = [
            Halfspace(normal=np.array([1.0, 0.0]), offset=1.0),
            Halfspace(normal=np.array([0.0, 1.0]), offset=1.0)
        ]
        
        polytope = HPolytope(halfspaces=halfspaces, dimension=2)
        
        assert polytope.dimension == 2
        assert len(polytope.halfspaces) == 2
    
    def test_get_Ab(self):
        """Test matrix representation extraction."""
        halfspaces = [
            Halfspace(normal=np.array([1.0, 0.0]), offset=2.0),
            Halfspace(normal=np.array([0.0, 1.0]), offset=3.0),
            Halfspace(normal=np.array([-1.0, -1.0]), offset=1.0)
        ]
        
        polytope = HPolytope(halfspaces=halfspaces, dimension=2)
        A, b = polytope.get_Ab()
        
        assert A.shape == (3, 2)
        assert b.shape == (3,)
        
        expected_A = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0]
        ])
        expected_b = np.array([2.0, 3.0, 1.0])
        
        np.testing.assert_array_almost_equal(A, expected_A)
        np.testing.assert_array_almost_equal(b, expected_b)
    
    def test_empty_polytope(self):
        """Test empty polytope."""
        polytope = HPolytope(halfspaces=[], dimension=2)
        A, b = polytope.get_Ab()
        
        assert A.shape == (0, 2)
        assert b.shape == (0,)


class TestRegionBuilder:
    """Tests for RegionBuilder class."""
    
    @pytest.fixture
    def simple_posterior(self):
        """Create simple 2D posterior for testing."""
        mu = np.array([0.5, 0.5])
        Sigma = np.array([
            [0.1, 0.02],
            [0.02, 0.1]
        ])
        return mu, Sigma
    
    def test_credible_ellipsoid_basic(self, simple_posterior):
        """Test basic ellipsoid construction."""
        mu, Sigma = simple_posterior
        builder = RegionBuilder()
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
        
        assert ellipsoid.dimension == 2
        assert ellipsoid.alpha == 0.80
        np.testing.assert_array_almost_equal(ellipsoid.center, mu)
        
        # Check that shape matrix is positive definite
        eigenvalues = np.linalg.eigvals(ellipsoid.shape_matrix)
        assert np.all(eigenvalues > 0)
    
    def test_credible_ellipsoid_chi2_scaling(self, simple_posterior):
        """Test that ellipsoid is scaled by chi-squared quantile."""
        mu, Sigma = simple_posterior
        builder = RegionBuilder()
        alpha = 0.80
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=alpha)
        
        # The shape matrix should be Sigma^{-1} / chi2_quantile
        chi2_quantile = chi2.ppf(alpha, df=2)
        Sigma_inv = np.linalg.inv(Sigma + builder.regularization * np.eye(2))
        expected_shape = Sigma_inv / chi2_quantile
        
        np.testing.assert_array_almost_equal(
            ellipsoid.shape_matrix,
            expected_shape,
            decimal=5
        )
    
    def test_credible_ellipsoid_invalid_dimensions(self):
        """Test that dimension mismatch raises error."""
        builder = RegionBuilder()
        
        mu = np.array([1.0, 2.0])
        Sigma = np.eye(3)  # Wrong dimension
        
        with pytest.raises(ValueError, match="doesn't match"):
            builder.credible_ellipsoid(mu, Sigma)
    
    def test_credible_ellipsoid_non_positive_definite(self):
        """Test that non-positive definite covariance raises error."""
        builder = RegionBuilder(regularization=0.0)
        
        mu = np.array([1.0, 2.0])
        Sigma = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])  # Not positive definite (det < 0)
        
        with pytest.raises(ValueError, match="not positive definite"):
            builder.credible_ellipsoid(mu, Sigma)
    
    def test_assemble_halfspaces_from_bounds(self):
        """Test assembling halfspaces from attribute bounds."""
        builder = RegionBuilder()
        
        attribute_bounds = {
            0: (0.0, 1.0),  # 0 <= x[0] <= 1
            1: (0.2, 0.8)   # 0.2 <= x[1] <= 0.8
        }
        
        polytope = builder.assemble_halfspaces(
            attribute_bounds=attribute_bounds,
            dimension=2
        )
        
        assert polytope.dimension == 2
        assert len(polytope.halfspaces) == 4  # 2 bounds per attribute
        
        # Verify constraints
        A, b = polytope.get_Ab()
        
        # Check that bounds are represented correctly
        # Should have: -x[0] <= 0, x[0] <= 1, -x[1] <= -0.2, x[1] <= 0.8
        assert A.shape == (4, 2)
    
    def test_assemble_halfspaces_from_scheme_constraints(self):
        """Test assembling halfspaces from scheme constraints."""
        builder = RegionBuilder()
        
        scheme_constraints = [
            Halfspace(normal=np.array([1.0, 0.0]), offset=0.5),
            Halfspace(normal=np.array([0.0, 1.0]), offset=0.7)
        ]
        
        polytope = builder.assemble_halfspaces(
            scheme_constraints=scheme_constraints
        )
        
        assert polytope.dimension == 2
        assert len(polytope.halfspaces) == 2
    
    def test_assemble_halfspaces_no_dimension(self):
        """Test that missing dimension raises error."""
        builder = RegionBuilder()
        
        with pytest.raises(ValueError, match="Cannot infer dimension"):
            builder.assemble_halfspaces()
    
    def test_intersect_ellipsoid_polytope(self, simple_posterior):
        """Test intersection of ellipsoid and polytope."""
        mu, Sigma = simple_posterior
        builder = RegionBuilder()
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
        
        # Create simple box constraints: 0 <= x[i] <= 1
        halfspaces = [
            Halfspace(normal=np.array([-1.0, 0.0]), offset=0.0),
            Halfspace(normal=np.array([1.0, 0.0]), offset=1.0),
            Halfspace(normal=np.array([0.0, -1.0]), offset=0.0),
            Halfspace(normal=np.array([0.0, 1.0]), offset=1.0)
        ]
        polytope = HPolytope(halfspaces=halfspaces, dimension=2)
        
        region = builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
        
        assert isinstance(region, CapabilityRegion)
        assert region.ellipsoid == ellipsoid
        assert region.polytope == polytope
    
    def test_intersect_dimension_mismatch(self, simple_posterior):
        """Test that dimension mismatch raises error."""
        mu, Sigma = simple_posterior
        builder = RegionBuilder()
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
        
        # Create 3D polytope
        halfspaces = [
            Halfspace(normal=np.array([1.0, 0.0, 0.0]), offset=1.0)
        ]
        polytope = HPolytope(halfspaces=halfspaces, dimension=3)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
    
    def test_sample_region_basic(self, simple_posterior):
        """Test basic region sampling."""
        mu, Sigma = simple_posterior
        builder = RegionBuilder()
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
        
        # Create box constraints
        halfspaces = [
            Halfspace(normal=np.array([-1.0, 0.0]), offset=0.0),
            Halfspace(normal=np.array([1.0, 0.0]), offset=1.0),
            Halfspace(normal=np.array([0.0, -1.0]), offset=0.0),
            Halfspace(normal=np.array([0.0, 1.0]), offset=1.0)
        ]
        polytope = HPolytope(halfspaces=halfspaces, dimension=2)
        
        region = builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
        
        # Sample from region
        n_samples = 100
        samples = builder.sample_region(
            region,
            n=n_samples,
            seed=42,
            burn_in=100,
            thin=5
        )
        
        assert samples.shape == (n_samples, 2)
        
        # Verify all samples satisfy constraints
        for sample in samples:
            # Check ellipsoid constraint
            diff = sample - mu
            ellipsoid_val = diff @ ellipsoid.shape_matrix @ diff
            assert ellipsoid_val <= 1.0 + 1e-4
            
            # Check box constraints
            assert np.all(sample >= -1e-4)
            assert np.all(sample <= 1.0 + 1e-4)
    
    def test_sample_region_reproducibility(self, simple_posterior):
        """Test that sampling is reproducible with same seed."""
        mu, Sigma = simple_posterior
        builder = RegionBuilder()
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
        polytope = HPolytope(halfspaces=[], dimension=2)
        region = builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
        
        samples1 = builder.sample_region(region, n=50, seed=123, burn_in=50)
        samples2 = builder.sample_region(region, n=50, seed=123, burn_in=50)
        
        np.testing.assert_array_almost_equal(samples1, samples2)
    
    def test_estimate_volume(self, simple_posterior):
        """Test volume estimation."""
        mu, Sigma = simple_posterior
        builder = RegionBuilder()
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
        polytope = HPolytope(halfspaces=[], dimension=2)
        region = builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
        
        volume = builder.estimate_volume(region, n_samples=1000, seed=42)
        
        assert volume > 0
        assert region.volume_estimate == volume
    
    def test_hypervolume_above_baseline(self, simple_posterior):
        """Test hypervolume above baseline computation."""
        mu, Sigma = simple_posterior
        builder = RegionBuilder()
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
        polytope = HPolytope(halfspaces=[], dimension=2)
        region = builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
        
        baseline = {'attr1': 0.3, 'attr2': 0.3}
        feature_names = ['attr1', 'attr2']
        
        hypervolume = builder.hypervolume_above_baseline(
            region,
            baseline,
            feature_names,
            n_samples=500,
            seed=42
        )
        
        assert hypervolume >= 0
        assert region.hypervolume_above_baseline == hypervolume
    
    def test_is_in_region(self, simple_posterior):
        """Test point-in-region check."""
        mu, Sigma = simple_posterior
        builder = RegionBuilder()
        
        ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
        
        # Create box constraints: 0 <= x[i] <= 1
        halfspaces = [
            Halfspace(normal=np.array([-1.0, 0.0]), offset=0.0),
            Halfspace(normal=np.array([1.0, 0.0]), offset=1.0),
            Halfspace(normal=np.array([0.0, -1.0]), offset=0.0),
            Halfspace(normal=np.array([0.0, 1.0]), offset=1.0)
        ]
        polytope = HPolytope(halfspaces=halfspaces, dimension=2)
        
        region = builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
        
        # Center should be in region
        assert builder._is_in_region(mu, region)
        
        # Point outside box should not be in region
        outside_point = np.array([2.0, 2.0])
        assert not builder._is_in_region(outside_point, region)
        
        # Point far from ellipsoid should not be in region
        far_point = np.array([0.5, 10.0])
        assert not builder._is_in_region(far_point, region)


class TestHitAndRunMCMC:
    """Tests for hit-and-run MCMC sampler."""
    
    def test_hit_and_run_basic(self):
        """Test basic hit-and-run sampling."""
        from src.regions.build import hit_and_run_mcmc
        
        # Simple 2D case: unit circle (ellipsoid) intersected with box [0,1]^2
        x0 = np.array([0.5, 0.5])
        ellipsoid_center = np.array([0.5, 0.5])
        ellipsoid_shape = np.eye(2) * 4.0  # Radius 0.5
        
        # Box constraints: 0 <= x[i] <= 1
        polytope_A = np.array([
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0]
        ])
        polytope_b = np.array([0.0, 1.0, 0.0, 1.0])
        
        samples = hit_and_run_mcmc(
            x0=x0,
            n_samples=50,
            burn_in=50,
            thin=5,
            ellipsoid_center=ellipsoid_center,
            ellipsoid_shape=ellipsoid_shape,
            polytope_A=polytope_A,
            polytope_b=polytope_b,
            seed=42
        )
        
        assert samples.shape == (50, 2)
        
        # Verify samples are in valid region
        for sample in samples:
            # Check box constraints
            assert np.all(sample >= -1e-4)
            assert np.all(sample <= 1.0 + 1e-4)
            
            # Check ellipsoid constraint
            diff = sample - ellipsoid_center
            ellipsoid_val = diff @ ellipsoid_shape @ diff
            assert ellipsoid_val <= 1.0 + 1e-4



class TestMatchupConstraintBuilder:
    """Tests for MatchupConstraintBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create MatchupConstraintBuilder for testing."""
        from src.regions.matchup import MatchupConstraintBuilder
        return MatchupConstraintBuilder()
    
    @pytest.fixture
    def opponent_row(self):
        """Create sample opponent data."""
        import pandas as pd
        return pd.Series({
            'blitz_rate': 0.4,
            'rim_deterrence_index': 1.5,
            'def_reb_strength': 1.3,
            'foul_discipline_index': 1.2,
            'scheme_switch_rate': 0.6,
            'help_nail_freq': 0.5
        })
    
    def test_builder_initialization(self, builder):
        """Test basic initialization."""
        assert builder.dimension == 11
        assert len(builder.attribute_names) == 11
        assert 'usage' in builder.attr_to_idx
        assert 'ts_pct' in builder.attr_to_idx
    
    def test_scheme_to_constraints_basic(self, builder, opponent_row):
        """Test basic scheme constraint generation."""
        constraints = builder.scheme_to_constraints(opponent_row)
        
        # Should generate multiple constraints
        assert len(constraints) > 0
        
        # All constraints should be Halfspace objects
        for constraint in constraints:
            assert isinstance(constraint, Halfspace)
            assert len(constraint.normal) == builder.dimension
            assert isinstance(constraint.offset, (float, np.floating))
    
    def test_scheme_to_constraints_blitz(self, builder):
        """Test blitz rate constraint."""
        import pandas as pd
        opponent_row = pd.Series({
            'blitz_rate': 0.5,  # High blitz rate
            'rim_deterrence_index': 1.0,
            'def_reb_strength': 1.0,
            'foul_discipline_index': 1.0,
            'scheme_switch_rate': 0.3,
            'help_nail_freq': 0.3
        })
        
        constraints = builder.scheme_to_constraints(opponent_row)
        
        # Should have blitz constraint
        assert len(constraints) > 0
        
        # Check that usage and tov_pct are involved
        usage_idx = builder.attr_to_idx['usage']
        tov_idx = builder.attr_to_idx['tov_pct']
        
        blitz_constraint_found = False
        for constraint in constraints:
            if constraint.normal[usage_idx] != 0 and constraint.normal[tov_idx] != 0:
                blitz_constraint_found = True
                break
        
        assert blitz_constraint_found
    
    def test_scheme_to_constraints_rim_deterrence(self, builder):
        """Test rim deterrence constraint."""
        import pandas as pd
        opponent_row = pd.Series({
            'blitz_rate': 0.2,
            'rim_deterrence_index': 1.8,  # High rim deterrence
            'def_reb_strength': 1.0,
            'foul_discipline_index': 1.0,
            'scheme_switch_rate': 0.3,
            'help_nail_freq': 0.3
        })
        
        constraints = builder.scheme_to_constraints(opponent_row)
        
        # Should have rim deterrence constraint
        rim_idx = builder.attr_to_idx['rim_attempt_rate']
        
        rim_constraint_found = False
        for constraint in constraints:
            if constraint.normal[rim_idx] == 1.0:
                rim_constraint_found = True
                # Offset should be reduced due to high deterrence
                assert constraint.offset < 0.30
                break
        
        assert rim_constraint_found
    
    def test_scheme_to_constraints_toggles(self, builder, opponent_row):
        """Test constraint toggles."""
        # Disable all constraints
        toggles = {
            'blitz_constraint': False,
            'rim_deterrence_constraint': False,
            'def_reb_constraint': False,
            'foul_discipline_constraint': False,
            'switch_constraint': False,
            'help_defense_constraint': False
        }
        
        constraints = builder.scheme_to_constraints(opponent_row, toggles=toggles)
        
        # Should have no constraints
        assert len(constraints) == 0
    
    def test_role_bounds_starter(self, builder):
        """Test role bounds for starter."""
        constraints = builder.role_bounds('starter')
        
        # Should have bounds for multiple attributes
        assert len(constraints) > 0
        
        # Each attribute should have lower and upper bound (2 constraints)
        # So total should be even
        assert len(constraints) % 2 == 0
        
        # Check usage bounds exist
        usage_idx = builder.attr_to_idx['usage']
        
        usage_constraints = [
            c for c in constraints
            if c.normal[usage_idx] != 0
        ]
        
        # Should have 2 constraints (lower and upper)
        assert len(usage_constraints) == 2
    
    def test_role_bounds_rotation(self, builder):
        """Test role bounds for rotation player."""
        constraints = builder.role_bounds('rotation')
        
        assert len(constraints) > 0
        
        # Check that rotation has different bounds than starter
        usage_idx = builder.attr_to_idx['usage']
        
        usage_constraints = [
            c for c in constraints
            if c.normal[usage_idx] != 0
        ]
        
        assert len(usage_constraints) == 2
    
    def test_role_bounds_bench(self, builder):
        """Test role bounds for bench player."""
        constraints = builder.role_bounds('bench')
        
        assert len(constraints) > 0
        
        # Bench should have lower usage ceiling
        usage_idx = builder.attr_to_idx['usage']
        
        usage_upper_bounds = [
            c.offset for c in constraints
            if c.normal[usage_idx] == 1.0
        ]
        
        # Should have upper bound
        assert len(usage_upper_bounds) > 0
        # Bench upper bound should be lower than typical starter
        assert usage_upper_bounds[0] < 0.35
    
    def test_role_bounds_custom(self, builder):
        """Test custom attribute bounds."""
        custom_bounds = {
            'usage': (0.20, 0.30),
            'ts_pct': (0.50, 0.65)
        }
        
        constraints = builder.role_bounds('starter', attribute_bounds=custom_bounds)
        
        # Should have exactly 4 constraints (2 per attribute)
        assert len(constraints) == 4
        
        # Check usage bounds
        usage_idx = builder.attr_to_idx['usage']
        usage_constraints = [
            c for c in constraints
            if c.normal[usage_idx] != 0
        ]
        
        assert len(usage_constraints) == 2
    
    def test_pairwise_frontiers_for_missing_files(self, builder, tmp_path):
        """Test frontier loading with missing files."""
        # Use temporary directory that doesn't have frontier files
        builder.frontier_dir = tmp_path
        
        frontiers = builder.pairwise_frontiers_for(
            player_role='starter',
            opponent_scheme_bin='balanced'
        )
        
        # Should return empty list when files don't exist
        assert isinstance(frontiers, list)
        assert len(frontiers) == 0
    
    def test_pairwise_frontiers_for_custom_pairs(self, builder, tmp_path):
        """Test frontier loading with custom pairs."""
        builder.frontier_dir = tmp_path
        
        custom_pairs = [
            ('usage', 'ts_pct'),
            ('ast_pct', 'tov_pct')
        ]
        
        frontiers = builder.pairwise_frontiers_for(
            player_role='starter',
            opponent_scheme_bin='balanced',
            frontier_pairs=custom_pairs
        )
        
        # Should return empty list (no files exist)
        assert isinstance(frontiers, list)
    
    def test_build_all_constraints(self, builder, opponent_row):
        """Test building all constraints together."""
        frontiers, scheme_constraints, role_constraints = builder.build_all_constraints(
            player_role='starter',
            opponent_row=opponent_row,
            opponent_scheme_bin='balanced'
        )
        
        # Should return three lists
        assert isinstance(frontiers, list)
        assert isinstance(scheme_constraints, list)
        assert isinstance(role_constraints, list)
        
        # Should have scheme and role constraints
        assert len(scheme_constraints) > 0
        assert len(role_constraints) > 0
    
    def test_build_all_constraints_with_toggles(self, builder, opponent_row):
        """Test building constraints with toggles."""
        toggles = {
            'blitz_constraint': False,
            'rim_deterrence_constraint': True
        }
        
        frontiers, scheme_constraints, role_constraints = builder.build_all_constraints(
            player_role='rotation',
            opponent_row=opponent_row,
            opponent_scheme_bin='high_blitz',
            toggles=toggles
        )
        
        # Should still return valid lists
        assert isinstance(scheme_constraints, list)
        assert isinstance(role_constraints, list)
    
    def test_constraint_dimensions_match(self, builder, opponent_row):
        """Test that all constraints have matching dimensions."""
        constraints = builder.scheme_to_constraints(opponent_row)
        role_constraints = builder.role_bounds('starter')
        
        all_constraints = constraints + role_constraints
        
        # All should have same dimension
        for constraint in all_constraints:
            assert len(constraint.normal) == builder.dimension
    
    def test_scheme_constraints_are_valid_halfspaces(self, builder, opponent_row):
        """Test that scheme constraints are valid halfspaces."""
        constraints = builder.scheme_to_constraints(opponent_row)
        
        for constraint in constraints:
            # Normal vector should not be all zeros
            assert np.any(constraint.normal != 0)
            
            # Offset should be finite
            assert np.isfinite(constraint.offset)
            
            # Normal vector should have correct dimension
            assert len(constraint.normal) == builder.dimension
