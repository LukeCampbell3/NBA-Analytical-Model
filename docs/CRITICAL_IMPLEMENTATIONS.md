# Critical Implementations for Production

Quick reference guide for implementing critical features before production deployment.

## Priority 1: Data Contracts (Week 1-2)

### Implementation

Create `src/contracts/data_models.py`:

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date

class PlayersPerGameContract(BaseModel):
    """Data contract for player per-game statistics."""
    
    # Required fields
    player_id: str
    game_id: str
    date: date
    team_id: str
    opponent_id: str
    minutes: float = Field(ge=0.0, le=48.0)
    
    # Optional with defaults
    usage: Optional[float] = 0.18
    ts_pct: Optional[float] = None
    three_pa_rate: Optional[float] = 0.35
    # ... other fields
    
    class Config:
        extra = "allow"  # Allow extra columns
```

### Integration

Update `src/utils/data_loader.py` to use contracts.

## Priority 2: Cold-Start Priors (Week 3)

### Implementation

Create `src/priors/cold_start.py`:

```python
class ColdStartPriors:
    """Manages priors for new players and teams."""
    
    def get_player_prior(self, player_id: str, 
                        role: Optional[str] = None,
                        n_games: int = 0) -> PosteriorParams:
        """Get prior for new player."""
        # Implementation
        pass
```

## Priority 3: Graceful Degradation (Week 3)

### Implementation

Add fallbacks throughout the codebase for missing data.

## Priority 4: Drift Monitoring (Week 4)

### Implementation

Create `src/monitoring/drift.py`:

```python
class DriftMonitor:
    """Monitors data and model drift."""
    
    def check_population_shift(self, current_data, baseline_data):
        """Check for population mean shift using PSI."""
        pass
```

See PRODUCTION_READINESS_ASSESSMENT.md for complete details.
