"""
Capability region construction module for NBA player performance prediction.

This module implements geometric capability regions as the intersection of
credible ellipsoids (from player posteriors) and halfspace polytopes (from
frontiers, schemes, and bounds). Includes efficient sampling using hit-and-run
MCMC with Numba optimization.

Integrated with:
- Graceful degradation for singular matrices
- Fallbacks for missing frontiers
- Wider regions for missing opponent features
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from numba import jit
import warnings
import time

from src.frontiers.fit import Halfspace, FrontierModel
from src.features.transform import PosteriorParams
from src.utils.errors import (
    RegionConstructionError,
    SingularMatrixError,
    EmptyRegionError,
    ValidationError
)
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@dataclass
class Ellipsoid:
    """
    Represents a credible ellipsoid from player posterior.
    
    The ellipsoid is defined as: (x - center)^T @ shape_matrix @ (x - center) <= 1
    
    Attributes:
        center: Center point of the ellipsoid (mu from posterior)
        shape_matrix: Shape matrix A (inverse of scaled covariance)
        alpha: Credibility level (e.g., 0.80 for 80% credible region)
        dimension: Dimensionality of the space
    """
    center: np.ndarray
    shape_matrix: np.ndarray
    alpha: float
    dimension: int


@dataclass
class HPolytope:
    """
    Represents a halfspace polytope: {x : A @ x <= b}
    
    Attributes:
        halfspaces: List of Halfspace constraints
        dimension: Dimensionality of the space
    """
    halfspaces: List[Halfspace]
    dimension: int
    
    def get_Ab(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get matrix representation A @ x <= b.
        
        Returns:
            Tuple of (A matrix, b vector)
        """
        if not self.halfspaces:
            return np.empty((0, self.dimension)), np.empty(0)
        
        A = np.array([h.normal for h in self.halfspaces])
        b = np.array([h.offset for h in self.halfspaces])
        return A, b


@dataclass
class CapabilityRegion:
    """
    Represents a capability region as ellipsoid ∩ polytope.
    
    Attributes:
        ellipsoid: Credible ellipsoid from posterior
        polytope: Halfspace polytope from constraints
        volume_estimate: Estimated volume of the region
        hypervolume_above_baseline: Hypervolume above baseline performance
        samples: Cached samples from the region (optional)
    """
    ellipsoid: Ellipsoid
    polytope: HPolytope
    volume_estimate: Optional[float] = None
    hypervolume_above_baseline: Optional[float] = None
    samples: Optional[np.ndarray] = None


class RegionBuilder:
    """
    Constructs capability regions from posteriors, frontiers, and constraints.
    
    The capability region represents the feasible performance space for a player
    given their historical posterior, efficiency frontiers, opponent schemes,
    and role boundaries.
    """
    
    def __init__(self, regularization: float = 1e-6):
        """
        Initialize RegionBuilder.
        
        Args:
            regularization: Ridge regularization for covariance matrix
        """
        self.regularization = regularization
        logger.info(
            "RegionBuilder initialized",
            context={"regularization": regularization}
        )
    
    def credible_ellipsoid(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        alpha: float = 0.80
    ) -> Ellipsoid:
        """
        Construct credible ellipsoid from player posterior.
        
        The ellipsoid is defined as the alpha-credible region of a multivariate
        normal distribution: {x : (x-mu)^T @ Sigma^{-1} @ (x-mu) <= chi2_alpha}
        
        Args:
            mu: Mean vector (center of ellipsoid)
            Sigma: Covariance matrix
            alpha: Credibility level (default: 0.80 for 80% region)
        
        Returns:
            Ellipsoid object
        
        Raises:
            ValidationError: If inputs have invalid dimensions
            SingularMatrixError: If Sigma is not positive definite
        """
        logger.log_operation_start(
            "credible_ellipsoid",
            details={
                "dimension": len(mu) if mu.ndim == 1 else None,
                "alpha": alpha
            }
        )
        
        try:
            # Validate inputs
            if mu.ndim != 1:
                logger.error(
                    "Invalid mu dimension",
                    context={"mu_shape": mu.shape}
                )
                raise ValidationError(
                    f"mu must be 1-dimensional, got shape {mu.shape}",
                    field_name="mu",
                    invalid_value=mu.shape
                )
            
            dimension = len(mu)
            
            if Sigma.shape != (dimension, dimension):
                logger.error(
                    "Sigma shape mismatch",
                    context={
                        "sigma_shape": Sigma.shape,
                        "mu_dimension": dimension
                    }
                )
                raise ValidationError(
                    f"Sigma shape {Sigma.shape} doesn't match mu dimension {dimension}",
                    field_name="Sigma",
                    invalid_value=Sigma.shape
                )
            
            # Add regularization to ensure positive definite
            Sigma_reg = Sigma + self.regularization * np.eye(dimension)
            
            # Check if positive definite by attempting Cholesky decomposition
            try:
                np.linalg.cholesky(Sigma_reg)
            except np.linalg.LinAlgError as e:
                # FALLBACK: Try stronger regularization
                logger.warning(
                    "Initial regularization insufficient, trying stronger ridge",
                    context={"initial_ridge": self.regularization}
                )
                
                # Try progressively stronger regularization
                for ridge_multiplier in [10, 100, 1000]:
                    try:
                        ridge = self.regularization * ridge_multiplier
                        Sigma_reg = Sigma + ridge * np.eye(dimension)
                        np.linalg.cholesky(Sigma_reg)
                        logger.info(f"Successfully regularized with ridge={ridge}")
                        break
                    except np.linalg.LinAlgError:
                        continue
                else:
                    # All regularization attempts failed
                    # Compute condition number for diagnostics
                    try:
                        cond_num = np.linalg.cond(Sigma)
                    except:
                        cond_num = None
                    
                    logger.error(
                        "Covariance matrix is not positive definite even with strong regularization",
                        context={
                            "dimension": dimension,
                            "condition_number": cond_num,
                            "max_regularization_tried": self.regularization * 1000
                        }
                    )
                    raise SingularMatrixError(
                    "Covariance matrix is not positive definite even after regularization",
                    matrix_shape=Sigma.shape,
                    condition_number=cond_num
                )
            
            # Compute chi-squared quantile for the credibility level
            # For multivariate normal, the alpha-credible region satisfies:
            # (x-mu)^T @ Sigma^{-1} @ (x-mu) <= chi2_{alpha, dimension}
            from scipy.stats import chi2
            chi2_quantile = chi2.ppf(alpha, df=dimension)
            
            # Shape matrix: A = Sigma^{-1} / chi2_quantile
            # So that (x-mu)^T @ A @ (x-mu) <= 1 defines the ellipsoid
            Sigma_inv = np.linalg.inv(Sigma_reg)
            shape_matrix = Sigma_inv / chi2_quantile
            
            ellipsoid = Ellipsoid(
                center=mu.copy(),
                shape_matrix=shape_matrix,
                alpha=alpha,
                dimension=dimension
            )
            
            logger.log_operation_complete(
                "credible_ellipsoid",
                details={
                    "dimension": dimension,
                    "alpha": alpha
                }
            )
            
            return ellipsoid
            
        except (ValidationError, SingularMatrixError):
            raise
        except Exception as e:
            logger.log_operation_failed(
                "credible_ellipsoid",
                error=e,
                details={"dimension": len(mu) if mu.ndim == 1 else None}
            )
            raise RegionConstructionError(
                f"Failed to construct credible ellipsoid: {e}",
                reason=str(e),
                dimension=len(mu) if mu.ndim == 1 else None
            )
    
    def assemble_halfspaces(
        self,
        frontiers: Optional[List[FrontierModel]] = None,
        scheme_constraints: Optional[List[Halfspace]] = None,
        role_bounds: Optional[List[Halfspace]] = None,
        attribute_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        dimension: Optional[int] = None
    ) -> HPolytope:
        """
        Assemble halfspace polytope from various constraint sources.
        
        Args:
            frontiers: List of frontier models to linearize
            scheme_constraints: Halfspaces from opponent scheme
            role_bounds: Halfspaces from player role
            attribute_bounds: Dictionary of {attr_idx: (min, max)} bounds
            dimension: Dimensionality of the space (required if no other constraints)
        
        Returns:
            HPolytope with all constraints
        
        Raises:
            ValueError: If dimension cannot be inferred
        """
        all_halfspaces = []
        inferred_dimension = dimension
        
        # Add frontier constraints (linearized)
        if frontiers:
            from src.frontiers.fit import FrontierFitter
            fitter = FrontierFitter()
            
            for frontier in frontiers:
                linearized = fitter.linearize_frontier(frontier)
                all_halfspaces.extend(linearized)
                
                # Infer dimension from frontier halfspaces
                if inferred_dimension is None and linearized:
                    inferred_dimension = len(linearized[0].normal)
        
        # Add scheme constraints
        if scheme_constraints:
            all_halfspaces.extend(scheme_constraints)
            
            # Infer dimension
            if inferred_dimension is None and scheme_constraints:
                inferred_dimension = len(scheme_constraints[0].normal)
        
        # Add role bounds
        if role_bounds:
            all_halfspaces.extend(role_bounds)
            
            # Infer dimension
            if inferred_dimension is None and role_bounds:
                inferred_dimension = len(role_bounds[0].normal)
        
        # Add attribute bounds
        if attribute_bounds and inferred_dimension:
            for attr_idx, (min_val, max_val) in attribute_bounds.items():
                # Lower bound: x[attr_idx] >= min_val => -x[attr_idx] <= -min_val
                normal_lower = np.zeros(inferred_dimension)
                normal_lower[attr_idx] = -1.0
                all_halfspaces.append(Halfspace(
                    normal=normal_lower,
                    offset=-min_val
                ))
                
                # Upper bound: x[attr_idx] <= max_val
                normal_upper = np.zeros(inferred_dimension)
                normal_upper[attr_idx] = 1.0
                all_halfspaces.append(Halfspace(
                    normal=normal_upper,
                    offset=max_val
                ))
        
        # Validate dimension
        if inferred_dimension is None:
            raise ValueError(
                "Cannot infer dimension from constraints. "
                "Please provide dimension parameter."
            )
        
        return HPolytope(
            halfspaces=all_halfspaces,
            dimension=inferred_dimension
        )

    
    def intersect_ellipsoid_polytope(
        self,
        E: Ellipsoid,
        H: HPolytope
    ) -> CapabilityRegion:
        """
        Construct capability region as ellipsoid ∩ polytope.
        
        Validates that the intersection is non-empty by checking if the
        ellipsoid center satisfies polytope constraints.
        
        Args:
            E: Ellipsoid from posterior
            H: Halfspace polytope from constraints
        
        Returns:
            CapabilityRegion
        
        Raises:
            ValueError: If dimensions don't match or intersection is empty
        """
        # Validate dimensions
        if E.dimension != H.dimension:
            raise ValueError(
                f"Dimension mismatch: ellipsoid has dimension {E.dimension}, "
                f"polytope has dimension {H.dimension}"
            )
        
        # Check if ellipsoid center satisfies polytope constraints
        A, b = H.get_Ab()
        
        if len(A) > 0:
            violations = A @ E.center - b
            max_violation = np.max(violations)
            
            if max_violation > 1e-6:
                warnings.warn(
                    f"Ellipsoid center violates polytope constraints by {max_violation}. "
                    "The intersection may be empty or very small."
                )
        
        return CapabilityRegion(
            ellipsoid=E,
            polytope=H
        )
    
    def sample_region(
        self,
        region: CapabilityRegion,
        n: int,
        seed: Optional[int] = None,
        burn_in: int = 1000,
        thin: int = 10
    ) -> np.ndarray:
        """
        Sample points from capability region using hit-and-run MCMC.
        
        Uses Numba-optimized hit-and-run algorithm for efficient sampling
        from the intersection of ellipsoid and polytope.
        
        Args:
            region: CapabilityRegion to sample from
            n: Number of samples to generate
            seed: Random seed for reproducibility
            burn_in: Number of burn-in iterations (default: 1000)
            thin: Thinning factor - keep every thin-th sample (default: 10)
        
        Returns:
            Array of samples, shape (n, dimension)
        
        Raises:
            ValueError: If unable to find valid starting point
        """
        dimension = region.ellipsoid.dimension
        
        # Get polytope constraints
        A, b = region.polytope.get_Ab()
        
        # Find starting point (use ellipsoid center if valid, else search)
        # Set seed before finding starting point for reproducibility
        if seed is not None:
            np.random.seed(seed)
        x0 = self._find_starting_point(region)
        
        # Run hit-and-run MCMC (pass seed to ensure reproducibility)
        samples = hit_and_run_mcmc(
            x0=x0,
            n_samples=n,
            burn_in=burn_in,
            thin=thin,
            ellipsoid_center=region.ellipsoid.center,
            ellipsoid_shape=region.ellipsoid.shape_matrix,
            polytope_A=A,
            polytope_b=b,
            seed=seed if seed is not None else 0
        )
        
        # Cache samples in region
        region.samples = samples
        
        return samples
    
    def _find_starting_point(
        self,
        region: CapabilityRegion,
        max_attempts: int = 1000
    ) -> np.ndarray:
        """
        Find a valid starting point inside the capability region.
        
        Args:
            region: CapabilityRegion
            max_attempts: Maximum number of random attempts
        
        Returns:
            Valid starting point
        
        Raises:
            EmptyRegionError: If no valid point found
        """
        logger.debug(
            "Finding starting point for region sampling",
            context={"dimension": region.ellipsoid.dimension}
        )
        
        # Try ellipsoid center first
        center = region.ellipsoid.center
        if self._is_in_region(center, region):
            logger.debug("Using ellipsoid center as starting point")
            return center.copy()
        
        # Try random points from ellipsoid
        dimension = region.ellipsoid.dimension
        
        try:
            Sigma_sqrt = np.linalg.cholesky(
                np.linalg.inv(region.ellipsoid.shape_matrix)
            )
        except np.linalg.LinAlgError as e:
            logger.error(
                "Failed to compute Cholesky decomposition for sampling",
                context={"dimension": dimension}
            )
            raise EmptyRegionError(
                "Cannot sample from region: matrix decomposition failed",
                attempts=0,
                constraints_count=len(region.polytope.halfspaces) if region.polytope else 0
            )
        
        for attempt in range(max_attempts):
            # Sample from standard normal and transform to ellipsoid
            z = np.random.randn(dimension)
            x = center + Sigma_sqrt @ z
            
            if self._is_in_region(x, region):
                logger.debug(
                    "Found valid starting point",
                    context={"attempts": attempt + 1}
                )
                return x
        
        logger.error(
            "Could not find valid starting point",
            context={
                "max_attempts": max_attempts,
                "dimension": dimension,
                "constraints_count": len(region.polytope.halfspaces) if region.polytope else 0
            }
        )
        raise EmptyRegionError(
            f"Could not find valid starting point after {max_attempts} attempts. "
            "The region may be empty or very small.",
            attempts=max_attempts,
            constraints_count=len(region.polytope.halfspaces) if region.polytope else 0
        )
    
    def _is_in_region(self, x: np.ndarray, region: CapabilityRegion) -> bool:
        """
        Check if point x is inside the capability region.
        
        Args:
            x: Point to check
            region: CapabilityRegion
        
        Returns:
            True if x is in region, False otherwise
        """
        # Check ellipsoid constraint
        diff = x - region.ellipsoid.center
        ellipsoid_val = diff @ region.ellipsoid.shape_matrix @ diff
        
        if ellipsoid_val > 1.0 + 1e-6:
            return False
        
        # Check polytope constraints
        A, b = region.polytope.get_Ab()
        
        if len(A) > 0:
            violations = A @ x - b
            if np.any(violations > 1e-6):
                return False
        
        return True
    
    def estimate_volume(
        self,
        region: CapabilityRegion,
        n_samples: int = 10000,
        seed: Optional[int] = None
    ) -> float:
        """
        Estimate volume of capability region using Monte Carlo sampling.
        
        Uses rejection sampling from the bounding ellipsoid to estimate
        the volume of the intersection.
        
        Args:
            region: CapabilityRegion
            n_samples: Number of Monte Carlo samples
            seed: Random seed
        
        Returns:
            Estimated volume
        """
        if seed is not None:
            np.random.seed(seed)
        
        dimension = region.ellipsoid.dimension
        
        # Compute ellipsoid volume
        # Volume = (pi^(d/2) / Gamma(d/2 + 1)) * sqrt(det(Sigma))
        # where Sigma = inv(shape_matrix)
        from scipy.special import gamma
        
        Sigma = np.linalg.inv(region.ellipsoid.shape_matrix)
        det_Sigma = np.linalg.det(Sigma)
        
        ellipsoid_volume = (
            (np.pi ** (dimension / 2)) / gamma(dimension / 2 + 1)
        ) * np.sqrt(det_Sigma)
        
        # Sample from ellipsoid and count how many satisfy polytope constraints
        Sigma_sqrt = np.linalg.cholesky(Sigma)
        center = region.ellipsoid.center
        
        A, b = region.polytope.get_Ab()
        
        n_inside = 0
        for _ in range(n_samples):
            # Sample from standard normal and transform to ellipsoid
            z = np.random.randn(dimension)
            x = center + Sigma_sqrt @ z
            
            # Check if inside unit ball (ellipsoid constraint)
            if np.dot(z, z) > 1.0:
                continue
            
            # Check polytope constraints
            if len(A) > 0:
                violations = A @ x - b
                if np.any(violations > 1e-6):
                    continue
            
            n_inside += 1
        
        # Estimate volume as fraction of ellipsoid volume
        volume_estimate = ellipsoid_volume * (n_inside / n_samples)
        
        # Cache in region
        region.volume_estimate = volume_estimate
        
        return volume_estimate
    
    def hypervolume_above_baseline(
        self,
        region: CapabilityRegion,
        baseline: Dict[str, float],
        feature_names: List[str],
        n_samples: int = 5000,
        seed: Optional[int] = None
    ) -> float:
        """
        Compute hypervolume of region above baseline performance.
        
        This measures the "volume" of capability space where the player
        exceeds baseline performance levels.
        
        Args:
            region: CapabilityRegion
            baseline: Dictionary mapping feature names to baseline values
            feature_names: Ordered list of feature names matching region dimensions
            n_samples: Number of samples for estimation
            seed: Random seed
        
        Returns:
            Hypervolume above baseline (as fraction of total region volume)
        """
        # Sample from region
        if region.samples is None or len(region.samples) < n_samples:
            samples = self.sample_region(region, n_samples, seed=seed)
        else:
            samples = region.samples[:n_samples]
        
        # Create baseline vector
        baseline_vector = np.array([
            baseline.get(name, 0.0) for name in feature_names
        ])
        
        # Count samples above baseline (all dimensions)
        above_baseline = np.all(samples >= baseline_vector, axis=1)
        fraction_above = np.mean(above_baseline)
        
        # Compute hypervolume as fraction * total volume
        if region.volume_estimate is None:
            total_volume = self.estimate_volume(region, n_samples, seed)
        else:
            total_volume = region.volume_estimate
        
        hypervolume = fraction_above * total_volume
        
        # Cache in region
        region.hypervolume_above_baseline = hypervolume
        
        return hypervolume


# Numba-optimized hit-and-run MCMC sampler
@jit(nopython=True)
def hit_and_run_mcmc(
    x0: np.ndarray,
    n_samples: int,
    burn_in: int,
    thin: int,
    ellipsoid_center: np.ndarray,
    ellipsoid_shape: np.ndarray,
    polytope_A: np.ndarray,
    polytope_b: np.ndarray,
    seed: int
) -> np.ndarray:
    """
    Hit-and-run MCMC sampler for ellipsoid ∩ polytope (Numba-optimized).
    
    Args:
        x0: Starting point
        n_samples: Number of samples to collect
        burn_in: Number of burn-in iterations
        thin: Thinning factor
        ellipsoid_center: Center of ellipsoid
        ellipsoid_shape: Shape matrix of ellipsoid
        polytope_A: Constraint matrix A
        polytope_b: Constraint vector b
        seed: Random seed for reproducibility
    
    Returns:
        Array of samples, shape (n_samples, dimension)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    dimension = len(x0)
    total_iterations = burn_in + n_samples * thin
    
    # Initialize
    x_current = x0.copy()
    samples = np.zeros((n_samples, dimension))
    sample_idx = 0
    
    for iteration in range(total_iterations):
        # Sample random direction
        direction = np.random.randn(dimension)
        direction = direction / np.linalg.norm(direction)
        
        # Find valid interval along direction
        t_min, t_max = find_valid_interval(
            x_current,
            direction,
            ellipsoid_center,
            ellipsoid_shape,
            polytope_A,
            polytope_b
        )
        
        # Sample uniformly from valid interval
        if t_max > t_min:
            t = np.random.uniform(t_min, t_max)
            x_current = x_current + t * direction
        
        # Collect sample after burn-in and thinning
        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            samples[sample_idx] = x_current
            sample_idx += 1
    
    return samples


@jit(nopython=True)
def find_valid_interval(
    x: np.ndarray,
    direction: np.ndarray,
    ellipsoid_center: np.ndarray,
    ellipsoid_shape: np.ndarray,
    polytope_A: np.ndarray,
    polytope_b: np.ndarray
) -> Tuple[float, float]:
    """
    Find valid interval [t_min, t_max] along direction from x.
    
    The interval is constrained by both ellipsoid and polytope.
    
    Args:
        x: Current point
        direction: Direction vector (normalized)
        ellipsoid_center: Center of ellipsoid
        ellipsoid_shape: Shape matrix of ellipsoid
        polytope_A: Constraint matrix A
        polytope_b: Constraint vector b
    
    Returns:
        Tuple of (t_min, t_max)
    """
    # Initialize with large interval
    t_min = -1e10
    t_max = 1e10
    
    # Ellipsoid constraint: (x + t*d - c)^T @ A @ (x + t*d - c) <= 1
    # This is a quadratic in t: a*t^2 + b*t + c <= 1
    diff = x - ellipsoid_center
    
    a = direction @ ellipsoid_shape @ direction
    b_coef = 2.0 * diff @ ellipsoid_shape @ direction
    c = diff @ ellipsoid_shape @ diff - 1.0
    
    # Solve quadratic: a*t^2 + b*t + c = 0
    if a > 1e-10:
        discriminant = b_coef ** 2 - 4 * a * c
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b_coef - sqrt_disc) / (2 * a)
            t2 = (-b_coef + sqrt_disc) / (2 * a)
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
    
    # Polytope constraints: A @ (x + t*d) <= b
    # For each row i: a_i^T @ x + t * a_i^T @ d <= b_i
    # If a_i^T @ d > 0: t <= (b_i - a_i^T @ x) / (a_i^T @ d)
    # If a_i^T @ d < 0: t >= (b_i - a_i^T @ x) / (a_i^T @ d)
    
    for i in range(len(polytope_A)):
        a_i = polytope_A[i]
        b_i = polytope_b[i]
        
        a_dot_d = np.dot(a_i, direction)
        a_dot_x = np.dot(a_i, x)
        
        if abs(a_dot_d) > 1e-10:
            t_bound = (b_i - a_dot_x) / a_dot_d
            
            if a_dot_d > 0:
                t_max = min(t_max, t_bound)
            else:
                t_min = max(t_min, t_bound)
    
    return t_min, t_max
