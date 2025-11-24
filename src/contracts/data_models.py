"""
Pydantic data models for all data contracts.

These models define the schema for all data tables with validation,
defaults, and graceful handling of missing/extra columns.
"""

from datetime import date as date_type, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PlayersPerGameContract(BaseModel):
    """
    Data contract for player per-game statistics.
    
    Required fields: player_id, game_id, date, team_id, opponent_id, minutes
    Optional fields: All performance metrics with sensible defaults
    """
    
    # Required fields
    player_id: str = Field(..., description="Unique player identifier")
    game_id: str = Field(..., description="Unique game identifier")
    date: date_type = Field(..., description="Game date")
    team_id: str = Field(..., description="Player's team ID")
    opponent_id: str = Field(..., description="Opponent team ID")
    minutes: float = Field(..., ge=0.0, le=48.0, description="Minutes played")
    
    # Optional fields with defaults
    usage: Optional[float] = Field(0.18, ge=0.0, le=1.0, description="Usage rate")
    ts_pct: Optional[float] = Field(None, ge=0.0, le=1.5, description="True shooting percentage")
    three_pa_rate: Optional[float] = Field(0.35, ge=0.0, le=1.0, description="3PA rate")
    rim_attempt_rate: Optional[float] = Field(0.25, ge=0.0, le=1.0, description="Rim attempt rate")
    mid_attempt_rate: Optional[float] = Field(0.20, ge=0.0, le=1.0, description="Mid-range attempt rate")
    ast_pct: Optional[float] = Field(None, ge=0.0, le=1.0, description="Assist percentage")
    tov_pct: Optional[float] = Field(None, ge=0.0, le=1.0, description="Turnover percentage")
    orb_pct: Optional[float] = Field(None, ge=0.0, le=1.0, description="Offensive rebound percentage")
    drb_pct: Optional[float] = Field(None, ge=0.0, le=1.0, description="Defensive rebound percentage")
    stl_pct: Optional[float] = Field(None, ge=0.0, le=1.0, description="Steal percentage")
    blk_pct: Optional[float] = Field(None, ge=0.0, le=1.0, description="Block percentage")
    ft_rate: Optional[float] = Field(None, ge=0.0, le=2.0, description="Free throw rate")
    pf: Optional[float] = Field(0.0, ge=0.0, le=6.0, description="Personal fouls")
    role: Optional[str] = Field("unknown", description="Player role (starter, rotation, bench, unknown)")
    
    model_config = ConfigDict(extra="allow")
        
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        """Validate player role."""
        valid_roles = ['starter', 'rotation', 'bench', 'unknown']
        if v not in valid_roles:
            logger.warning(f"Invalid role '{v}', defaulting to 'unknown'")
            return 'unknown'
        return v


class OpponentFeaturesContract(BaseModel):
    """
    Data contract for opponent defensive features.
    
    Required: opponent_id, date, and at least one scheme rate or pace
    """
    
    # Required fields
    opponent_id: str = Field(..., description="Opponent team ID")
    date: date_type = Field(..., description="As-of date")
    
    # Scheme rates (at least one required)
    scheme_drop_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Drop coverage rate")
    scheme_switch_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Switch rate")
    scheme_ice_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Ice coverage rate")
    blitz_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Blitz rate")
    
    # Defensive metrics
    rim_deterrence_index: Optional[float] = Field(0.0, ge=0.0, description="Rim protection strength")
    def_reb_strength: Optional[float] = Field(0.0, ge=0.0, description="Defensive rebounding strength")
    foul_discipline_index: Optional[float] = Field(0.0, ge=0.0, description="Foul discipline")
    help_nail_freq: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Help defense frequency")
    
    # Pace
    pace: Optional[float] = Field(None, gt=0.0, description="Team pace")
    
    model_config = ConfigDict(extra="allow")
    
    @model_validator(mode='after')
    def check_required_any_of(self):
        """Ensure at least one scheme rate or pace is provided."""
        scheme_rates = [
            self.scheme_drop_rate,
            self.scheme_switch_rate,
            self.scheme_ice_rate
        ]
        
        if not any(scheme_rates) and self.pace is None:
            logger.warning("No scheme rates or pace provided, will use league medians")
        
        return self


class RotationPriorsContract(BaseModel):
    """Data contract for rotation priors."""
    
    game_id: str = Field(..., description="Game ID")
    player_id: str = Field(..., description="Player ID")
    exp_minutes: float = Field(..., ge=0.0, le=48.0, description="Expected minutes")
    exp_usage: float = Field(..., ge=0.0, le=1.0, description="Expected usage rate")
    on_off_synergy_key: Optional[str] = Field(None, description="On-off synergy key")
    
    model_config = ConfigDict(extra="allow")


class TeamTendenciesContract(BaseModel):
    """Data contract for team tendencies (for team fit module)."""
    
    team_id: str = Field(..., description="Team ID")
    date: date_type = Field(..., description="As-of date")
    
    # Offensive tendencies
    pace: float = Field(..., gt=0.0, description="Team pace")
    three_pa_rate: Optional[float] = Field(0.35, ge=0.0, le=1.0, description="3PA rate")
    rim_rate: Optional[float] = Field(0.25, ge=0.0, le=1.0, description="Rim attempt rate")
    mid_rate: Optional[float] = Field(0.20, ge=0.0, le=1.0, description="Mid-range rate")
    ast_tempo: Optional[float] = Field(None, description="Assist tempo")
    tov_tolerance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Turnover tolerance")
    
    # Defensive tendencies
    foul_discipline: Optional[float] = Field(None, ge=0.0, description="Foul discipline")
    def_coverage_drop: Optional[float] = Field(None, ge=0.0, le=1.0, description="Drop coverage rate")
    def_coverage_switch: Optional[float] = Field(None, ge=0.0, le=1.0, description="Switch rate")
    def_coverage_ice: Optional[float] = Field(None, ge=0.0, le=1.0, description="Ice coverage rate")
    blitz_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Blitz rate")
    
    # Rebounding and transition
    oreb_emphasis: Optional[float] = Field(None, ge=0.0, le=1.0, description="Offensive rebound emphasis")
    transition_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Transition rate")
    
    model_config = ConfigDict(extra="allow")


class CoachSlidersContract(BaseModel):
    """Data contract for coach manual adjustments (for team fit module)."""
    
    team_id: str = Field(..., description="Team ID")
    date: date_type = Field(..., description="As-of date")
    
    target_usage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target usage rate")
    target_pace: Optional[float] = Field(None, gt=0.0, description="Target pace")
    spacing_need: Optional[float] = Field(None, description="Spacing requirement")
    turnover_risk_cap: Optional[float] = Field(None, ge=0.0, le=1.0, description="Turnover risk cap")
    foul_cap: Optional[float] = Field(None, ge=0.0, description="Foul cap")
    
    model_config = ConfigDict(extra="allow")


class OnOffLineupsContract(BaseModel):
    """Data contract for on-off lineup data (for synergy graph)."""
    
    lineup_id: str = Field(..., description="Lineup ID")
    game_id: Optional[str] = Field(None, description="Game ID")
    players: List[str] = Field(..., min_items=2, max_items=5, description="Player IDs in lineup")
    minutes: float = Field(..., ge=0.0, description="Minutes played together")
    net_rating: Optional[float] = Field(None, description="Net rating")
    
    # Synergy metrics
    assist_chain_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Assist chain rate")
    spacing_proxy: Optional[float] = Field(None, description="Spacing proxy metric")
    oreb_proxy: Optional[float] = Field(None, description="Offensive rebound proxy")
    turnover_chain_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Turnover chain rate")
    
    model_config = ConfigDict(extra="allow")
    
    @field_validator('players')
    @classmethod
    def validate_players(cls, v):
        """Ensure unique players."""
        if len(v) != len(set(v)):
            raise ValueError("Players must be unique in lineup")
        return v


# Alias mappings for backward compatibility
SCHEMA_ALIASES = {
    'usage_rate': 'usage',
    'ts': 'ts_pct',
    'threepr': 'three_pa_rate',
    'rim_rate': 'rim_attempt_rate',
    'mid_rate': 'mid_attempt_rate',
    '3PA_rate': 'three_pa_rate',
    'TS%': 'ts_pct',
    'USG%': 'usage',
    'AST%': 'ast_pct',
    'TOV%': 'tov_pct',
    'ORB%': 'orb_pct',
    'DRB%': 'drb_pct',
    'STL%': 'stl_pct',
    'BLK%': 'blk_pct'
}
