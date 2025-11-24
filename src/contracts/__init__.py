"""
Data contracts for schema validation and evolution.

This module provides Pydantic models for all data tables with:
- Schema validation
- Type checking
- Default values
- Alias mapping
- Graceful handling of extra/missing columns
"""

from src.contracts.data_models import (
    PlayersPerGameContract,
    OpponentFeaturesContract,
    RotationPriorsContract,
    TeamTendenciesContract,
    CoachSlidersContract,
    OnOffLineupsContract
)
from src.contracts.schema_registry import SchemaRegistry
from src.contracts.validators import ContractValidator

__all__ = [
    'PlayersPerGameContract',
    'OpponentFeaturesContract',
    'RotationPriorsContract',
    'TeamTendenciesContract',
    'CoachSlidersContract',
    'OnOffLineupsContract',
    'SchemaRegistry',
    'ContractValidator'
]
