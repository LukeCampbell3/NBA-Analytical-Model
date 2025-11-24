"""
Matchup constraint builder for NBA player performance prediction.

This module converts opponent defensive schemes and player roles into
geometric halfspace constraints for capability region construction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import joblib

from src.frontiers.fit import Halfspace, FrontierModel


class MatchupConstraintBuilder:
    """
    Builds geometric constraints from opponent schemes and player roles.
    
    Converts defensive strategy parameters (drop rate, switch rate, etc.)
    and player role boundaries into halfspace constraints that restrict
    the capability region.
    """
    
    def __init__(
        self,
        frontier_dir: str = "artifacts/frontiers",
        attribute_names: Optional[List[str]] = None
    ):
        """
        Initialize MatchupConstraintBuilder.
        
        Args:
            frontier_dir: Directory containing saved frontier models
            attribute_names: Ordered list of attribute names for indexing
                           (e.g., ['ts_pct', 'usage', 'ast_pct', ...])
        """
        self.frontier_dir = Path(frontier_dir)
        
        # Default attribute ordering if not provided
        if attribute_names is None:
            self.attribute_names = [
                'ts_pct',           # 0: True shooting percentage
                'usage',            # 1: Usage rate
                'ast_pct',          # 2: Assist percentage
                'tov_pct',          # 3: Turnover percentage
                'orb_pct',          # 4: Offensive rebound percentage
                'drb_pct',          # 5: Defensive rebound percentage
                'stl_pct',          # 6: Steal percentage
                'blk_pct',          # 7: Block percentage
                'three_pa_rate',    # 8: Three-point attempt rate
                'rim_attempt_rate', # 9: Rim attempt rate
                'ft_rate'           # 10: Free throw rate
            ]
        else:
            self.attribute_names = attribute_names
        
        self.dimension = len(self.attribute_names)
        
        # Create attribute name to index mapping
        self.attr_to_idx = {name: idx for idx, name in enumerate(self.attribute_names)}
    
    def scheme_to_constraints(
        self,
        opponent_row: pd.Series,
        toggles: Optional[Dict[str, bool]] = None
    ) -> List[Halfspace]:
        """
        Convert opponent defensive scheme to halfspace constraints.
        
        Defensive schemes constrain player capabilities based on how the
        opponent defends. For example:
        - High blitz_rate constrains usage and increases turnover risk
        - High rim_deterrence constrains rim_attempt_rate and efficiency
        - High switch_rate affects isolation opportunities
        
        Args:
            opponent_row: Series with opponent defensive features:
                - scheme_drop_rate: Frequency of drop coverage (0-1)
                - scheme_switch_rate: Frequency of switching (0-1)
                - scheme_ice_rate: Frequency of ice/down coverage (0-1)
                - blitz_rate: Frequency of blitzing pick-and-roll (0-1)
                - rim_deterrence_index: Rim protection strength (0-2+)
                - def_reb_strength: Defensive rebounding strength (0-2+)
                - foul_discipline_index: Foul discipline (0-2+)
                - help_nail_freq: Help defense frequency (0-1)
            toggles: Optional dict to enable/disable specific constraints
        
        Returns:
            List of Halfspace constraints
        """
        constraints = []
        
        # Default: all constraints enabled
        if toggles is None:
            toggles = {}
        
        # Extract scheme parameters with defaults
        blitz_rate = opponent_row.get('blitz_rate', 0.0)
        rim_deterrence = opponent_row.get('rim_deterrence_index', 1.0)
        def_reb_strength = opponent_row.get('def_reb_strength', 1.0)
        foul_discipline = opponent_row.get('foul_discipline_index', 1.0)
        switch_rate = opponent_row.get('scheme_switch_rate', 0.0)
        help_nail_freq = opponent_row.get('help_nail_freq', 0.0)
        
        # Constraint 1: High blitz rate constrains usage and increases turnover risk
        # Constraint: usage + 2.0 * tov_pct <= baseline_sum * (1 + blitz_penalty)
        if toggles.get('blitz_constraint', True) and blitz_rate > 0.3:
            blitz_penalty = (blitz_rate - 0.3) * 0.5  # Scale penalty
            
            normal = np.zeros(self.dimension)
            normal[self.attr_to_idx['usage']] = 1.0
            normal[self.attr_to_idx['tov_pct']] = 2.0
            
            # Baseline: typical usage=0.25, tov_pct=0.12 => sum=0.49
            baseline_sum = 0.49
            offset = baseline_sum * (1.0 + blitz_penalty)
            
            constraints.append(Halfspace(normal=normal, offset=offset))
        
        # Constraint 2: High rim deterrence constrains rim attempts and efficiency
        # Constraint: rim_attempt_rate <= baseline * (2.0 - rim_deterrence)
        if toggles.get('rim_deterrence_constraint', True) and rim_deterrence > 1.0:
            normal = np.zeros(self.dimension)
            normal[self.attr_to_idx['rim_attempt_rate']] = 1.0
            
            # Baseline rim attempt rate: 0.30
            baseline_rim_rate = 0.30
            offset = baseline_rim_rate * (2.0 - rim_deterrence)
            
            constraints.append(Halfspace(normal=normal, offset=offset))
        
        # Constraint 3: Strong defensive rebounding constrains offensive rebounds
        # Constraint: orb_pct <= baseline * (2.0 - def_reb_strength)
        if toggles.get('def_reb_constraint', True) and def_reb_strength > 1.0:
            normal = np.zeros(self.dimension)
            normal[self.attr_to_idx['orb_pct']] = 1.0
            
            # Baseline offensive rebound percentage: 0.05
            baseline_orb = 0.05
            offset = baseline_orb * (2.0 - def_reb_strength)
            
            constraints.append(Halfspace(normal=normal, offset=offset))
        
        # Constraint 4: High foul discipline reduces free throw opportunities
        # Constraint: ft_rate <= baseline * (2.0 - foul_discipline)
        if toggles.get('foul_discipline_constraint', True) and foul_discipline > 1.0:
            normal = np.zeros(self.dimension)
            normal[self.attr_to_idx['ft_rate']] = 1.0
            
            # Baseline free throw rate: 0.25
            baseline_ft = 0.25
            offset = baseline_ft * (2.0 - foul_discipline)
            
            constraints.append(Halfspace(normal=normal, offset=offset))
        
        # Constraint 5: High switch rate affects isolation and assist opportunities
        # Constraint: usage - ast_pct <= baseline_diff * (1 + switch_penalty)
        if toggles.get('switch_constraint', True) and switch_rate > 0.5:
            switch_penalty = (switch_rate - 0.5) * 0.3
            
            normal = np.zeros(self.dimension)
            normal[self.attr_to_idx['usage']] = 1.0
            normal[self.attr_to_idx['ast_pct']] = -1.0
            
            # Baseline: usage=0.25, ast_pct=0.20 => diff=0.05
            baseline_diff = 0.05
            offset = baseline_diff * (1.0 + switch_penalty)
            
            constraints.append(Halfspace(normal=normal, offset=offset))
        
        # Constraint 6: High help defense frequency constrains assist opportunities
        # Constraint: ast_pct <= baseline * (1.5 - help_nail_freq)
        if toggles.get('help_defense_constraint', True) and help_nail_freq > 0.4:
            normal = np.zeros(self.dimension)
            normal[self.attr_to_idx['ast_pct']] = 1.0
            
            # Baseline assist percentage: 0.25
            baseline_ast = 0.25
            offset = baseline_ast * (1.5 - help_nail_freq)
            
            constraints.append(Halfspace(normal=normal, offset=offset))
        
        return constraints
    
    def role_bounds(
        self,
        role: str,
        attribute_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> List[Halfspace]:
        """
        Generate role-specific attribute bounds as halfspace constraints.
        
        Different roles have different typical ranges for attributes:
        - Starters: Higher usage floor, tighter minutes range
        - Rotation: Moderate ranges
        - Bench: Lower usage ceiling, wider variance
        
        Args:
            role: Player role ('starter', 'rotation', 'bench')
            attribute_bounds: Optional custom bounds dict
                            {attr_name: (min_val, max_val)}
        
        Returns:
            List of Halfspace constraints for role bounds
        """
        constraints = []
        
        # Default role-specific bounds
        if attribute_bounds is None:
            if role == 'starter':
                attribute_bounds = {
                    'usage': (0.18, 0.40),        # Higher floor, high ceiling
                    'ts_pct': (0.45, 0.75),       # Reasonable efficiency range
                    'ast_pct': (0.05, 0.50),      # Wide range for different positions
                    'tov_pct': (0.05, 0.25),      # Turnover range
                    'three_pa_rate': (0.0, 0.60), # 3PT attempt rate
                    'rim_attempt_rate': (0.0, 0.50), # Rim attempt rate
                    'ft_rate': (0.10, 0.60),      # Free throw rate
                }
            elif role == 'rotation':
                attribute_bounds = {
                    'usage': (0.12, 0.35),        # Moderate floor and ceiling
                    'ts_pct': (0.40, 0.75),       # Slightly wider efficiency range
                    'ast_pct': (0.03, 0.45),      # Moderate assist range
                    'tov_pct': (0.05, 0.28),      # Slightly higher turnover ceiling
                    'three_pa_rate': (0.0, 0.65), # 3PT attempt rate
                    'rim_attempt_rate': (0.0, 0.55), # Rim attempt rate
                    'ft_rate': (0.05, 0.55),      # Free throw rate
                }
            else:  # bench
                attribute_bounds = {
                    'usage': (0.08, 0.30),        # Lower floor and ceiling
                    'ts_pct': (0.35, 0.75),       # Widest efficiency range
                    'ast_pct': (0.02, 0.40),      # Lower assist range
                    'tov_pct': (0.05, 0.30),      # Higher turnover ceiling
                    'three_pa_rate': (0.0, 0.70), # 3PT attempt rate
                    'rim_attempt_rate': (0.0, 0.60), # Rim attempt rate
                    'ft_rate': (0.03, 0.50),      # Free throw rate
                }
        
        # Convert bounds to halfspace constraints
        for attr_name, (min_val, max_val) in attribute_bounds.items():
            if attr_name not in self.attr_to_idx:
                continue
            
            attr_idx = self.attr_to_idx[attr_name]
            
            # Lower bound: x[attr_idx] >= min_val => -x[attr_idx] <= -min_val
            normal_lower = np.zeros(self.dimension)
            normal_lower[attr_idx] = -1.0
            constraints.append(Halfspace(
                normal=normal_lower,
                offset=-min_val
            ))
            
            # Upper bound: x[attr_idx] <= max_val
            normal_upper = np.zeros(self.dimension)
            normal_upper[attr_idx] = 1.0
            constraints.append(Halfspace(
                normal=normal_upper,
                offset=max_val
            ))
        
        return constraints
    
    def pairwise_frontiers_for(
        self,
        player_role: str,
        opponent_scheme_bin: str,
        frontier_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> List[FrontierModel]:
        """
        Retrieve pairwise frontier constraints for role and opponent scheme.
        
        Loads pre-fitted frontier models that define trade-off boundaries
        between attribute pairs (e.g., usage vs. efficiency, volume vs. assists).
        
        Args:
            player_role: Player role ('starter', 'rotation', 'bench')
            opponent_scheme_bin: Opponent scheme bin identifier
                               (e.g., 'high_blitz', 'rim_protection', 'balanced')
            frontier_pairs: Optional list of (x_attr, y_attr) pairs to load.
                          If None, loads default important pairs.
        
        Returns:
            List of FrontierModel objects
        
        Raises:
            FileNotFoundError: If frontier file doesn't exist (returns empty list)
        """
        frontiers = []
        
        # Default frontier pairs if not specified
        if frontier_pairs is None:
            frontier_pairs = [
                ('usage', 'ts_pct'),           # Volume-efficiency tradeoff
                ('usage', 'tov_pct'),          # Volume-turnover tradeoff
                ('ast_pct', 'tov_pct'),        # Playmaking-turnover tradeoff
                ('three_pa_rate', 'ts_pct'),   # 3PT volume-efficiency tradeoff
                ('rim_attempt_rate', 'ts_pct'), # Rim volume-efficiency tradeoff
            ]
        
        # Load each frontier model
        for x_attr, y_attr in frontier_pairs:
            # Construct filename based on strata
            filename = (
                f"frontier_{x_attr}_{y_attr}_"
                f"role_{player_role}_"
                f"scheme_{opponent_scheme_bin}.pkl"
            )
            filepath = self.frontier_dir / filename
            
            # Try to load frontier
            try:
                frontier = joblib.load(filepath)
                frontiers.append(frontier)
            except FileNotFoundError:
                # Frontier doesn't exist - skip silently
                # This is expected for some role/scheme combinations
                continue
            except Exception as e:
                # Log other errors but continue
                print(f"Warning: Failed to load frontier {filename}: {e}")
                continue
        
        return frontiers
    
    def build_all_constraints(
        self,
        player_role: str,
        opponent_row: pd.Series,
        opponent_scheme_bin: str,
        toggles: Optional[Dict[str, bool]] = None,
        custom_attribute_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        frontier_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[List[FrontierModel], List[Halfspace], List[Halfspace]]:
        """
        Build all constraints for a matchup scenario.
        
        Convenience method that combines scheme constraints, role bounds,
        and frontier retrieval into a single call.
        
        Args:
            player_role: Player role ('starter', 'rotation', 'bench')
            opponent_row: Series with opponent defensive features
            opponent_scheme_bin: Opponent scheme bin identifier
            toggles: Optional dict to enable/disable specific constraints
            custom_attribute_bounds: Optional custom role bounds
            frontier_pairs: Optional list of frontier pairs to load
        
        Returns:
            Tuple of (frontiers, scheme_constraints, role_bounds)
        """
        # Get scheme constraints
        scheme_constraints = self.scheme_to_constraints(
            opponent_row=opponent_row,
            toggles=toggles
        )
        
        # Get role bounds
        role_constraints = self.role_bounds(
            role=player_role,
            attribute_bounds=custom_attribute_bounds
        )
        
        # Get frontier models
        frontiers = self.pairwise_frontiers_for(
            player_role=player_role,
            opponent_scheme_bin=opponent_scheme_bin,
            frontier_pairs=frontier_pairs
        )
        
        return frontiers, scheme_constraints, role_constraints
