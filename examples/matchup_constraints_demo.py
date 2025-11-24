"""
Demo script showing how to use MatchupConstraintBuilder with RegionBuilder.

This example demonstrates the full workflow of:
1. Creating opponent scheme constraints
2. Creating role-specific bounds
3. Loading frontier models (if available)
4. Building a capability region with all constraints
5. Sampling from the constrained region
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.regions.matchup import MatchupConstraintBuilder
from src.regions.build import RegionBuilder


def main():
    """Run matchup constraints demo."""
    
    print("=" * 70)
    print("Matchup Constraints Demo")
    print("=" * 70)
    
    # Step 1: Create opponent data
    print("\n1. Creating opponent defensive scheme data...")
    opponent_data = pd.Series({
        'blitz_rate': 0.35,              # Moderate blitz rate
        'rim_deterrence_index': 1.3,     # Moderate rim protection
        'def_reb_strength': 1.2,         # Moderate defensive rebounding
        'foul_discipline_index': 1.1,    # Moderate foul discipline
        'scheme_switch_rate': 0.45,      # Moderate switching
        'help_nail_freq': 0.35           # Moderate help defense
    })
    
    print(f"   Opponent blitz rate: {opponent_data['blitz_rate']:.2f}")
    print(f"   Rim deterrence: {opponent_data['rim_deterrence_index']:.2f}")
    print(f"   Defensive rebounding: {opponent_data['def_reb_strength']:.2f}")
    
    # Step 2: Initialize MatchupConstraintBuilder
    print("\n2. Initializing MatchupConstraintBuilder...")
    matchup_builder = MatchupConstraintBuilder()
    print(f"   Dimension: {matchup_builder.dimension}")
    print(f"   Attributes: {', '.join(matchup_builder.attribute_names[:5])}...")
    
    # Step 3: Generate scheme constraints
    print("\n3. Generating scheme constraints...")
    scheme_constraints = matchup_builder.scheme_to_constraints(opponent_data)
    print(f"   Generated {len(scheme_constraints)} scheme constraints")
    
    for i, constraint in enumerate(scheme_constraints[:3]):
        # Find which attributes are involved
        involved_attrs = [
            matchup_builder.attribute_names[j]
            for j in range(len(constraint.normal))
            if constraint.normal[j] != 0
        ]
        print(f"   Constraint {i+1}: {', '.join(involved_attrs)} <= {constraint.offset:.3f}")
    
    # Step 4: Generate role bounds
    print("\n4. Generating role bounds for 'starter'...")
    role_constraints = matchup_builder.role_bounds('starter')
    print(f"   Generated {len(role_constraints)} role bound constraints")
    
    # Show usage bounds
    usage_idx = matchup_builder.attr_to_idx['usage']
    usage_bounds = [
        (constraint.normal[usage_idx], constraint.offset)
        for constraint in role_constraints
        if constraint.normal[usage_idx] != 0
    ]
    print(f"   Usage bounds: {usage_bounds}")
    
    # Step 5: Try to load frontiers (will be empty if not available)
    print("\n5. Loading frontier models...")
    frontiers = matchup_builder.pairwise_frontiers_for(
        player_role='starter',
        opponent_scheme_bin='high_blitz'
    )
    print(f"   Loaded {len(frontiers)} frontier models")
    
    # Step 6: Build complete constraint set
    print("\n6. Building complete constraint set...")
    all_frontiers, all_scheme, all_role = matchup_builder.build_all_constraints(
        player_role='starter',
        opponent_row=opponent_data,
        opponent_scheme_bin='high_blitz'
    )
    
    total_constraints = len(all_scheme) + len(all_role)
    print(f"   Total constraints: {total_constraints}")
    print(f"   - Scheme constraints: {len(all_scheme)}")
    print(f"   - Role constraints: {len(all_role)}")
    print(f"   - Frontier models: {len(all_frontiers)}")
    
    # Step 7: Create a capability region with constraints
    print("\n7. Creating capability region...")
    
    # Create a realistic posterior for a starter that satisfies constraints
    # Order: ts_pct, usage, ast_pct, tov_pct, orb_pct, drb_pct, stl_pct, blk_pct, 
    #        three_pa_rate, rim_attempt_rate, ft_rate
    dimension = matchup_builder.dimension
    
    # Blitz constraint: usage + 2.0 * tov_pct <= 0.502
    # So if usage=0.22, then tov_pct <= (0.502 - 0.22) / 2.0 = 0.141
    mu = np.array([
        0.56,   # ts_pct - good efficiency
        0.22,   # usage - moderate usage (within starter bounds)
        0.20,   # ast_pct - moderate assists
        0.10,   # tov_pct - low turnovers to satisfy blitz constraint
        0.03,   # orb_pct - low (constrained by def_reb_strength)
        0.12,   # drb_pct - moderate defensive rebounds
        0.015,  # stl_pct - low steals
        0.015,  # blk_pct - low blocks
        0.35,   # three_pa_rate - moderate 3PT attempts
        0.15,   # rim_attempt_rate - moderate (constrained by rim_deterrence)
        0.25    # ft_rate - moderate free throws
    ])
    
    # Smaller covariance to stay within constraints
    Sigma = np.eye(dimension) * 0.005
    for i in range(dimension - 1):
        Sigma[i, i+1] = 0.0005
        Sigma[i+1, i] = 0.0005
    
    # Build region
    region_builder = RegionBuilder()
    
    # Create ellipsoid from posterior
    ellipsoid = region_builder.credible_ellipsoid(mu, Sigma, alpha=0.80)
    print(f"   Created ellipsoid with alpha={ellipsoid.alpha}")
    
    # Assemble polytope from constraints
    polytope = region_builder.assemble_halfspaces(
        frontiers=all_frontiers,
        scheme_constraints=all_scheme,
        role_bounds=all_role
    )
    print(f"   Assembled polytope with {len(polytope.halfspaces)} halfspaces")
    
    # Intersect to create capability region
    region = region_builder.intersect_ellipsoid_polytope(ellipsoid, polytope)
    print(f"   Created capability region")
    
    # Step 8: Verify constraints are satisfied
    print("\n8. Verifying posterior satisfies constraints...")
    
    # Check scheme constraints
    scheme_violations = 0
    for constraint in all_scheme:
        value = np.dot(constraint.normal, mu)
        if value > constraint.offset + 1e-6:
            scheme_violations += 1
            print(f"   WARNING: Constraint violated by {value - constraint.offset:.4f}")
    
    if scheme_violations == 0:
        print(f"   ✓ All {len(all_scheme)} scheme constraints satisfied")
    else:
        print(f"   ✗ {scheme_violations} scheme constraints violated")
    
    # Check role constraints
    role_violations = 0
    for constraint in all_role:
        value = np.dot(constraint.normal, mu)
        if value > constraint.offset + 1e-6:
            role_violations += 1
    
    if role_violations == 0:
        print(f"   ✓ All {len(all_role)} role constraints satisfied")
    else:
        print(f"   ✗ {role_violations} role constraints violated")
    
    # Step 9: Show how constraints affect the region
    print("\n9. Constraint impact analysis:")
    print(f"   Posterior mean (selected attributes):")
    for attr_name in ['usage', 'tov_pct', 'rim_attempt_rate', 'orb_pct']:
        idx = matchup_builder.attr_to_idx[attr_name]
        print(f"     {attr_name:20s}: {mu[idx]:.3f}")
    
    print(f"\n   Key constraints:")
    print(f"     Blitz constraint: usage + 2*tov_pct <= 0.502")
    print(f"       Actual value: {mu[1] + 2*mu[3]:.3f}")
    print(f"     Rim deterrence: rim_attempt_rate <= 0.210")
    print(f"       Actual value: {mu[9]:.3f}")
    print(f"     Def rebounding: orb_pct <= 0.040")
    print(f"       Actual value: {mu[4]:.3f}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
