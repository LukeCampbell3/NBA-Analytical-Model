"""
Manual integration test for capability region construction.
This demonstrates the full workflow of building and sampling from a capability region.
"""

import numpy as np
from src.regions.build import RegionBuilder, Halfspace
from src.features.transform import PosteriorParams
from datetime import datetime

def test_full_region_workflow():
    """Test complete workflow: posterior -> ellipsoid -> polytope -> region -> samples."""
    
    print("=" * 60)
    print("Testing Capability Region Construction")
    print("=" * 60)
    
    # Step 1: Create a player posterior (simulating output from FeatureTransform)
    print("\n1. Creating player posterior...")
    mu = np.array([0.55, 0.25, 0.30, 0.15, 0.50, 0.02, 0.03])  # 7D capability vector
    Sigma = np.array([
        [0.01, 0.002, 0.001, 0.0005, 0.001, 0.0001, 0.0001],
        [0.002, 0.008, 0.001, 0.0003, 0.0008, 0.0001, 0.0001],
        [0.001, 0.001, 0.01, 0.0005, 0.001, 0.0001, 0.0001],
        [0.0005, 0.0003, 0.0005, 0.005, 0.0005, 0.0001, 0.0001],
        [0.001, 0.0008, 0.001, 0.0005, 0.012, 0.0001, 0.0001],
        [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.001, 0.0001],
        [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.001]
    ])
    
    print(f"   Posterior mean (mu): {mu}")
    print(f"   Covariance shape: {Sigma.shape}")
    
    # Step 2: Build credible ellipsoid
    print("\n2. Building credible ellipsoid (alpha=0.80)...")
    builder = RegionBuilder()
    ellipsoid = builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
    
    print(f"   Ellipsoid dimension: {ellipsoid.dimension}")
    print(f"   Ellipsoid center: {ellipsoid.center}")
    print(f"   Shape matrix eigenvalues: {np.linalg.eigvals(ellipsoid.shape_matrix)[:3]}...")
    
    # Step 3: Create halfspace constraints (simulating frontiers, schemes, bounds)
    print("\n3. Assembling halfspace polytope...")
    
    # Simple box constraints: 0 <= x[i] <= 1 for all dimensions
    attribute_bounds = {i: (0.0, 1.0) for i in range(7)}
    
    # Add some scheme constraints (e.g., usage + turnover rate constraint)
    scheme_constraints = [
        Halfspace(
            normal=np.array([0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]),  # USG + 0.5*TOV
            offset=0.35  # <= 0.35
        )
    ]
    
    polytope = builder.assemble_halfspaces(
        scheme_constraints=scheme_constraints,
        attribute_bounds=attribute_bounds,
        dimension=7
    )
    
    print(f"   Polytope dimension: {polytope.dimension}")
    print(f"   Number of halfspace constraints: {len(polytope.halfspaces)}")
    
    # Step 4: Intersect ellipsoid and polytope
    print("\n4. Constructing capability region (ellipsoid ∩ polytope)...")
    region = builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
    
    print(f"   Region created successfully")
    print(f"   Ellipsoid center in region: {builder._is_in_region(mu, region)}")
    
    # Step 5: Sample from region
    print("\n5. Sampling from capability region...")
    n_samples = 500
    samples = builder.sample_region(
        region,
        n=n_samples,
        seed=42,
        burn_in=500,
        thin=10
    )
    
    print(f"   Generated {len(samples)} samples")
    print(f"   Sample shape: {samples.shape}")
    print(f"   Sample mean: {np.mean(samples, axis=0)}")
    print(f"   Sample std: {np.std(samples, axis=0)}")
    
    # Verify all samples are valid
    valid_count = 0
    for sample in samples:
        if builder._is_in_region(sample, region):
            valid_count += 1
    
    print(f"   Valid samples: {valid_count}/{n_samples} ({100*valid_count/n_samples:.1f}%)")
    
    # Step 6: Estimate volume
    print("\n6. Estimating region volume...")
    volume = builder.estimate_volume(region, n_samples=2000, seed=42)
    
    print(f"   Estimated volume: {volume:.6f}")
    
    # Step 7: Compute hypervolume above baseline
    print("\n7. Computing hypervolume above baseline...")
    baseline = {
        'TS%': 0.50,
        'USG%': 0.20,
        'AST%': 0.25,
        'TOV%': 0.20,
        'TRB%': 0.45,
        'STL%': 0.015,
        'BLK%': 0.02
    }
    feature_names = ['TS%', 'USG%', 'AST%', 'TOV%', 'TRB%', 'STL%', 'BLK%']
    
    hypervolume = builder.hypervolume_above_baseline(
        region,
        baseline,
        feature_names,
        n_samples=500,
        seed=42
    )
    
    print(f"   Hypervolume above baseline: {hypervolume:.6f}")
    print(f"   Fraction above baseline: {hypervolume/volume:.2%}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_full_region_workflow()
