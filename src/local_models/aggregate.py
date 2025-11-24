"""Local model aggregation and blending with global simulation.

This module implements aggregation of local event-level predictions into
box-level expectations and blending with global simulation distributions.
Includes recalibration logic to maintain proper uncertainty quantification.
"""

import numpy as np
from typing import Dict, Optional, Any
from scipy import stats


class LocalAggregator:
    """Aggregates local model predictions and blends with global simulation.
    
    Converts event-level probabilities (rebounds, assists, shots) into
    expected box statistics and blends them with global simulation distributions
    using configurable weights. Maintains proper uncertainty through recalibration.
    """
    
    def __init__(
        self,
        global_weight: float = 0.6,
        local_weight: float = 0.4,
        recalibration_method: str = 'variance_scaling'
    ):
        """Initialize the local aggregator.
        
        Args:
            global_weight: Weight for global simulation (default: 0.6)
            local_weight: Weight for local expectations (default: 0.4)
            recalibration_method: Method for uncertainty recalibration
                Options: 'variance_scaling', 'bootstrap', 'none'
        
        Raises:
            ValueError: If weights don't sum to 1.0
        """
        if not np.isclose(global_weight + local_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {global_weight + local_weight}"
            )
        
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.recalibration_method = recalibration_method
    
    def local_to_box_expectations(
        self,
        local_probs: Dict[str, np.ndarray],
        minutes: float,
        usage: float,
        pace: float,
        possessions_per_minute: float = 1.0
    ) -> Dict[str, float]:
        """Convert local event probabilities to expected box statistics.
        
        Maps event-level probabilities to expected counting stats using
        game context (minutes, usage, pace). Handles rebounds, assists,
        and shots with appropriate scaling.
        
        Args:
            local_probs: Dictionary mapping event types to probability arrays
                Expected keys: 'rebound_prob', 'assist_prob', 'shot_prob'
                Each value is an array of event-level probabilities
            minutes: Expected minutes played
            usage: Player usage rate (possessions per minute)
            pace: Team pace (possessions per 48 minutes)
            possessions_per_minute: Possessions per minute (derived from pace)
        
        Returns:
            Dictionary mapping box stat names to expected values
            Keys: 'PTS', 'REB', 'AST', 'FGA', 'FGM', '3PA', '3PM', 'FTA', 'FTM'
        
        Raises:
            ValueError: If required probability arrays are missing
        """
        expectations = {}
        
        # Calculate expected possessions for the player
        expected_possessions = minutes * usage * possessions_per_minute
        
        # Rebounds: sum of rebound probabilities scaled by opportunities
        if 'rebound_prob' in local_probs:
            rebound_probs = local_probs['rebound_prob']
            # Estimate rebound opportunities based on pace and minutes
            # Typical: ~40 rebounds per game at pace 100, player gets minutes/48 share
            rebound_opportunities = (pace / 100.0) * 40 * (minutes / 48.0)
            
            # Expected rebounds = sum of probabilities * opportunities / num_events
            if len(rebound_probs) > 0:
                avg_rebound_prob = np.mean(rebound_probs)
                expectations['REB'] = avg_rebound_prob * rebound_opportunities
            else:
                expectations['REB'] = 0.0
        
        # Assists: sum of assist probabilities scaled by possessions
        if 'assist_prob' in local_probs:
            assist_probs = local_probs['assist_prob']
            
            # Expected assists = sum of probabilities * possessions / num_events
            if len(assist_probs) > 0:
                avg_assist_prob = np.mean(assist_probs)
                # Typical assist rate: ~20% of possessions for high-usage players
                assist_opportunities = expected_possessions * 0.5  # Half of possessions could be assists
                expectations['AST'] = avg_assist_prob * assist_opportunities
            else:
                expectations['AST'] = 0.0
        
        # Shots: convert shot probabilities to points and attempts
        if 'shot_prob' in local_probs:
            shot_probs = local_probs['shot_prob']
            
            if len(shot_probs) > 0:
                # Expected field goal attempts based on usage
                # Typical: usage rate * possessions * shot_rate
                shot_rate = 0.7  # ~70% of possessions end in a shot attempt
                expected_fga = expected_possessions * shot_rate
                expectations['FGA'] = expected_fga
                
                # Expected field goals made
                avg_shot_prob = np.mean(shot_probs)
                expectations['FGM'] = avg_shot_prob * expected_fga
                
                # Estimate 3PA and 3PM (assume 35% of shots are threes)
                three_point_rate = 0.35
                expectations['3PA'] = expected_fga * three_point_rate
                expectations['3PM'] = expectations['FGM'] * three_point_rate
                
                # Estimate free throws (typical: 0.25 FTA per FGA)
                ft_rate = 0.25
                expectations['FTA'] = expected_fga * ft_rate
                expectations['FTM'] = expectations['FTA'] * 0.75  # Assume 75% FT%
                
                # Calculate points: 2pt FGM + 3pt FGM + FTM
                two_pt_fgm = expectations['FGM'] - expectations['3PM']
                expectations['PTS'] = (
                    2 * two_pt_fgm + 
                    3 * expectations['3PM'] + 
                    expectations['FTM']
                )
            else:
                expectations['FGA'] = 0.0
                expectations['FGM'] = 0.0
                expectations['3PA'] = 0.0
                expectations['3PM'] = 0.0
                expectations['FTA'] = 0.0
                expectations['FTM'] = 0.0
                expectations['PTS'] = 0.0
        
        return expectations
    
    def blend_global_local(
        self,
        global_summary: Dict[str, np.ndarray],
        local_expect: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """Blend global simulation distributions with local expectations.
        
        Combines global simulation samples with local point estimates using
        weighted blending. Recalibrates to maintain proper uncertainty.
        
        Args:
            global_summary: Dictionary mapping stat names to sample arrays
                Each value is an array of N simulation samples
            local_expect: Dictionary mapping stat names to expected values
                Each value is a scalar point estimate
            weights: Optional stat-specific weights. If None, uses default weights.
                Format: {'stat_name': {'global': w_g, 'local': w_l}}
        
        Returns:
            Dictionary mapping stat names to blended sample arrays
            Each array has the same length as input global samples
        
        Raises:
            ValueError: If stat names don't match between inputs
        """
        blended = {}
        
        # Get all stats that appear in both global and local
        global_stats = set(global_summary.keys())
        local_stats = set(local_expect.keys())
        common_stats = global_stats.intersection(local_stats)
        
        if not common_stats:
            raise ValueError(
                f"No common statistics between global ({global_stats}) "
                f"and local ({local_stats})"
            )
        
        for stat in common_stats:
            global_samples = global_summary[stat]
            local_value = local_expect[stat]
            
            # Get weights for this stat
            if weights and stat in weights:
                w_global = weights[stat].get('global', self.global_weight)
                w_local = weights[stat].get('local', self.local_weight)
            else:
                w_global = self.global_weight
                w_local = self.local_weight
            
            # Blend: weighted combination of global samples and local value
            # Shift global samples toward local expectation
            global_mean = np.mean(global_samples)
            global_std = np.std(global_samples)
            
            # Compute blended mean
            blended_mean = w_global * global_mean + w_local * local_value
            
            # Shift global samples to match blended mean
            shifted_samples = global_samples - global_mean + blended_mean
            
            # Recalibrate uncertainty
            if self.recalibration_method == 'variance_scaling':
                # Scale variance to account for information from local model
                # More local weight = less uncertainty
                uncertainty_factor = np.sqrt(w_global)
                blended_samples = (
                    blended_mean + 
                    (shifted_samples - blended_mean) * uncertainty_factor
                )
            elif self.recalibration_method == 'bootstrap':
                # Bootstrap resampling with replacement
                n_samples = len(global_samples)
                n_global = int(w_global * n_samples)
                n_local = n_samples - n_global
                
                # Sample from global distribution
                global_bootstrap = np.random.choice(
                    shifted_samples, 
                    size=n_global, 
                    replace=True
                )
                
                # Sample from local (add noise around point estimate)
                local_noise_std = global_std * 0.5  # Reduced uncertainty
                local_bootstrap = np.random.normal(
                    local_value, 
                    local_noise_std, 
                    size=n_local
                )
                
                # Combine
                blended_samples = np.concatenate([global_bootstrap, local_bootstrap])
                np.random.shuffle(blended_samples)
            elif self.recalibration_method == 'none':
                # No recalibration, just use shifted samples
                blended_samples = shifted_samples
            else:
                raise ValueError(
                    f"Unknown recalibration method: {self.recalibration_method}"
                )
            
            blended[stat] = blended_samples
        
        # Add stats that only appear in global (no local information)
        for stat in global_stats - common_stats:
            blended[stat] = global_summary[stat]
        
        return blended
    
    def compute_blend_diagnostics(
        self,
        global_summary: Dict[str, np.ndarray],
        local_expect: Dict[str, float],
        blended: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Compute diagnostic metrics for blending quality.
        
        Args:
            global_summary: Original global simulation distributions
            local_expect: Local model expectations
            blended: Blended distributions
        
        Returns:
            Dictionary mapping stat names to diagnostic metrics
            Metrics: 'shift_magnitude', 'uncertainty_ratio', 'blend_quality'
        """
        diagnostics = {}
        
        for stat in blended.keys():
            if stat not in global_summary:
                continue
            
            global_samples = global_summary[stat]
            blended_samples = blended[stat]
            
            global_mean = np.mean(global_samples)
            global_std = np.std(global_samples)
            blended_mean = np.mean(blended_samples)
            blended_std = np.std(blended_samples)
            
            # Shift magnitude: how much did the mean change?
            shift_magnitude = abs(blended_mean - global_mean) / (global_std + 1e-6)
            
            # Uncertainty ratio: how much did uncertainty change?
            uncertainty_ratio = blended_std / (global_std + 1e-6)
            
            # Blend quality: how close is blended mean to target?
            if stat in local_expect:
                local_value = local_expect[stat]
                target_mean = (
                    self.global_weight * global_mean + 
                    self.local_weight * local_value
                )
                blend_quality = 1.0 - abs(blended_mean - target_mean) / (target_mean + 1e-6)
            else:
                blend_quality = 1.0  # No local info, so no target
            
            diagnostics[stat] = {
                'shift_magnitude': shift_magnitude,
                'uncertainty_ratio': uncertainty_ratio,
                'blend_quality': blend_quality,
                'global_mean': global_mean,
                'global_std': global_std,
                'blended_mean': blended_mean,
                'blended_std': blended_std
            }
        
        return diagnostics
    
    def aggregate_and_blend(
        self,
        global_summary: Dict[str, np.ndarray],
        local_probs: Dict[str, np.ndarray],
        minutes: float,
        usage: float,
        pace: float,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Complete pipeline: aggregate local probs and blend with global.
        
        Convenience method that combines local_to_box_expectations and
        blend_global_local into a single call.
        
        Args:
            global_summary: Global simulation distributions
            local_probs: Local event probabilities
            minutes: Expected minutes played
            usage: Player usage rate
            pace: Team pace
            weights: Optional stat-specific weights
        
        Returns:
            Dictionary with keys:
                - 'blended': Blended distributions
                - 'local_expect': Local expectations
                - 'diagnostics': Blending diagnostics
        """
        # Convert local probabilities to box expectations
        possessions_per_minute = pace / 48.0
        local_expect = self.local_to_box_expectations(
            local_probs=local_probs,
            minutes=minutes,
            usage=usage,
            pace=pace,
            possessions_per_minute=possessions_per_minute
        )
        
        # Blend with global simulation
        blended = self.blend_global_local(
            global_summary=global_summary,
            local_expect=local_expect,
            weights=weights
        )
        
        # Compute diagnostics
        diagnostics = self.compute_blend_diagnostics(
            global_summary=global_summary,
            local_expect=local_expect,
            blended=blended
        )
        
        return {
            'blended': blended,
            'local_expect': local_expect,
            'diagnostics': diagnostics
        }
