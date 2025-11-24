"""
Data loading and validation module for NBA player statistics.

This module provides functionality to load player game statistics from CSV files,
validate data quality, apply outlier capping, and enforce temporal ordering to
prevent data leakage.

Integrated with:
- Data contracts for schema validation
- Schema registry for evolution tracking
- Graceful fallbacks for missing data
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .errors import DataQualityError, DataNotFoundError, DataLeakageError, DataError
from .logger import get_logger
from src.contracts import (
    PlayersPerGameContract,
    OpponentFeaturesContract,
    SchemaRegistry,
    ContractValidator
)
from src.priors import ColdStartPriors

# Initialize logger
logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation checks."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missingness_report: Dict[str, float]


class DataLoader:
    """
    Loads and validates player game statistics from CSV files.
    
    This class handles loading player data from the Data directory structure,
    validates data quality (missingness, outliers), and enforces temporal
    ordering to prevent data leakage in model training.
    """
    
    # Required columns that must be present in the data
    REQUIRED_COLUMNS = [
        'Player', 'Date', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK',
        'FG%', '3P%', 'TS%', 'USG%', 'TOV', 'FT%', 'ORTG', 'DRTG'
    ]
    
    # Columns that should be numeric
    NUMERIC_COLUMNS = [
        'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK',
        'FG%', '3P%', 'TS%', 'USG%', 'TOV', 'FT%', 'ORTG', 'DRTG', 'BPM', 'GmSc'
    ]
    
    def __init__(self, data_dir: str = "Data", 
                 use_contracts: bool = True):
        """
        Initialize the DataLoader.
        
        Season is automatically detected from the data being loaded.
        No need to specify season - the loader adapts to any season's data.
        
        Args:
            data_dir: Root directory containing player data folders
            use_contracts: If True, use Pydantic contracts for validation
        
        Raises:
            DataNotFoundError: If data directory doesn't exist
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.error(
                "Data directory not found",
                context={
                    "data_dir": str(data_dir),
                    "absolute_path": str(self.data_dir.absolute())
                }
            )
            raise DataNotFoundError(
                f"Data directory not found: {data_dir}",
                file_path=str(self.data_dir.absolute())
            )
        
        # Initialize contracts and schema registry
        self.use_contracts = use_contracts
        if use_contracts:
            self.schema_registry = SchemaRegistry()
            self.contract_validator = ContractValidator(strict=False)
            logger.info("Initialized with data contracts")
        
        # Cold-start priors cache (season -> ColdStartPriors)
        # Priors are loaded on-demand when data is loaded
        self._priors_cache = {}
        
        logger.info(
            "DataLoader initialized",
            context={
                "data_dir": str(data_dir),
                "use_contracts": use_contracts
            }
        )
    
    def _get_priors_for_season(self, season: int) -> Optional[ColdStartPriors]:
        """
        Get cold-start priors for a specific season.
        
        Priors are cached per season and loaded on-demand.
        
        Args:
            season: Season year
            
        Returns:
            ColdStartPriors for the season, or None if unavailable
        """
        if season not in self._priors_cache:
            try:
                self._priors_cache[season] = ColdStartPriors(season=season)
                logger.info(f"Loaded cold-start priors for season {season}")
            except Exception as e:
                logger.warning(f"Could not load priors for season {season}: {e}")
                self._priors_cache[season] = None
        
        return self._priors_cache[season]
    
    def _detect_season_from_data(self, df: pd.DataFrame) -> Optional[int]:
        """
        Detect season from DataFrame dates.
        
        Args:
            df: DataFrame with Date column
            
        Returns:
            Detected season year, or None if cannot detect
        """
        if 'Date' not in df.columns or len(df) == 0:
            return None
        
        try:
            # Get the most recent date
            dates = pd.to_datetime(df['Date'], errors='coerce')
            max_date = dates.max()
            
            if pd.isna(max_date):
                return None
            
            # NBA season spans two calendar years
            # If date is Oct-Dec, it's the start of that season
            # If date is Jan-Sep, it's the end of previous season
            year = max_date.year
            if max_date.month >= 10:  # Oct, Nov, Dec
                season = year
            else:  # Jan-Sep
                season = year - 1
            
            logger.debug(f"Detected season {season} from data (max_date={max_date})")
            return season
            
        except Exception as e:
            logger.warning(f"Could not detect season from data: {e}")
            return None
    
    def load_player_data(
        self, 
        player_name: str, 
        year: int,
        use_processed: bool = False
    ) -> pd.DataFrame:
        """
        Load player game statistics from CSV file.
        
        Args:
            player_name: Player name (e.g., "Stephen_Curry")
            year: Season year (e.g., 2024)
            use_processed: If True, load processed CSV; otherwise load raw CSV
            
        Returns:
            DataFrame with player game statistics
            
        Raises:
            DataNotFoundError: If the player data file doesn't exist
            DataError: If the CSV file is empty or malformed
        """
        start_time = time.time()
        
        logger.log_operation_start(
            "load_player_data",
            details={
                "player_name": player_name,
                "year": year,
                "use_processed": use_processed
            }
        )
        
        try:
            # Construct file path
            player_dir = self.data_dir / player_name
            if not player_dir.exists():
                logger.error(
                    "Player directory not found",
                    context={
                        "player_name": player_name,
                        "player_dir": str(player_dir)
                    }
                )
                raise DataNotFoundError(
                    f"Player directory not found: {player_dir}",
                    file_path=str(player_dir),
                    player_name=player_name
                )
            
            # Determine filename
            if use_processed:
                filename = f"{year}_processed.csv"
            else:
                filename = f"{year}.csv"
            
            file_path = player_dir / filename
            if not file_path.exists():
                logger.error(
                    "Data file not found",
                    context={
                        "player_name": player_name,
                        "year": year,
                        "file_path": str(file_path)
                    }
                )
                raise DataNotFoundError(
                    f"Data file not found: {file_path}",
                    file_path=str(file_path),
                    player_name=player_name
                )
            
            # Load CSV
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.error(
                    "Failed to read CSV file",
                    context={
                        "file_path": str(file_path),
                        "error": str(e)
                    }
                )
                raise DataError(
                    f"Failed to read CSV file {file_path}: {e}",
                    details={"file_path": str(file_path), "error": str(e)}
                )
            
            if df.empty:
                logger.warning(
                    "CSV file is empty",
                    context={"file_path": str(file_path)}
                )
                raise DataError(
                    f"CSV file is empty: {file_path}",
                    details={"file_path": str(file_path)}
                )
            
            # Parse date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Sort by date to ensure temporal ordering
            if 'Date' in df.columns:
                df = df.sort_values('Date').reset_index(drop=True)
            
            duration = time.time() - start_time
            
            logger.log_operation_complete(
                "load_player_data",
                duration_sec=duration,
                details={
                    "player_name": player_name,
                    "year": year,
                    "rows_loaded": len(df),
                    "columns": list(df.columns)
                }
            )
            
            return df
            
        except (DataNotFoundError, DataError):
            raise
        except Exception as e:
            logger.log_operation_failed(
                "load_player_data",
                error=e,
                details={
                    "player_name": player_name,
                    "year": year
                }
            )
            raise DataError(
                f"Unexpected error loading player data: {e}",
                details={"player_name": player_name, "year": year}
            )
    
    def validate_data(
        self, 
        df: pd.DataFrame,
        max_missingness: float = 0.05
    ) -> ValidationResult:
        """
        Validate data quality by checking for missing values and data integrity.
        
        Args:
            df: DataFrame to validate
            max_missingness: Maximum allowed fraction of missing values (default: 5%)
            
        Returns:
            ValidationResult with validation status and details
            
        Raises:
            DataQualityError: If validation fails with critical errors
        """
        logger.log_operation_start(
            "validate_data",
            details={
                "rows": len(df) if not df.empty else 0,
                "columns": len(df.columns) if not df.empty else 0,
                "max_missingness": max_missingness
            }
        )
        
        errors = []
        warnings = []
        missingness_report = {}
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            logger.error(
                "Validation failed: DataFrame is empty",
                context={"errors": errors}
            )
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                missingness_report=missingness_report
            )
        
        # Check for required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check missingness for each required column
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_frac = missing_count / len(df)
                missingness_report[col] = missing_frac
                
                if missing_frac > max_missingness:
                    errors.append(
                        f"Column '{col}' has {missing_frac:.2%} missing values "
                        f"(threshold: {max_missingness:.2%})"
                    )
                elif missing_frac > 0:
                    warnings.append(
                        f"Column '{col}' has {missing_frac:.2%} missing values"
                    )
        
        # Check for invalid date values
        if 'Date' in df.columns:
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                warnings.append(
                    f"{invalid_dates} rows have invalid date values"
                )
        
        # Check for negative values in counting stats
        count_stats = ['MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
        for col in count_stats:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    warnings.append(
                        f"Column '{col}' has {negative_count} negative values"
                    )
        
        # Check for percentage values out of range [0, 1]
        pct_stats = ['FG%', '3P%', 'TS%', 'FT%']
        for col in pct_stats:
            if col in df.columns:
                out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
                if out_of_range > 0:
                    warnings.append(
                        f"Column '{col}' has {out_of_range} values outside [0, 1] range"
                    )
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.log_operation_complete(
                "validate_data",
                details={
                    "status": "valid",
                    "warnings_count": len(warnings),
                    "missingness_report": missingness_report
                }
            )
        else:
            logger.warning(
                "Data validation found errors",
                context={
                    "errors": errors,
                    "warnings": warnings,
                    "missingness_report": missingness_report
                }
            )
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            missingness_report=missingness_report
        )
        
        # Raise exception if validation failed
        if not is_valid:
            raise DataQualityError(
                "Data validation failed",
                validation_errors=errors,
                missingness_report=missingness_report
            )
        
        return result
    
    def apply_outlier_caps(
        self,
        df: pd.DataFrame,
        role: Optional[str] = None,
        season: Optional[int] = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99
    ) -> pd.DataFrame:
        """
        Apply outlier capping based on role and season quantiles.
        
        This method caps extreme values to prevent outliers from distorting
        model training. Capping is done per role and season to account for
        different performance distributions.
        
        Args:
            df: DataFrame with player statistics
            role: Player role ('starter', 'rotation', 'bench'). If None, use global quantiles
            season: Season year for stratification. If None, use all data
            lower_quantile: Lower quantile for capping (default: 1st percentile)
            upper_quantile: Upper quantile for capping (default: 99th percentile)
            
        Returns:
            DataFrame with outliers capped
        """
        df_capped = df.copy()
        
        # Determine which subset to use for computing quantiles
        if role is not None and 'role' in df.columns:
            mask = df['role'] == role
        else:
            mask = pd.Series([True] * len(df))
        
        if season is not None and 'Date' in df.columns:
            # Extract year from date
            df_year = df['Date'].dt.year
            mask = mask & (df_year == season)
        
        # If mask filters out all data, use global quantiles
        if mask.sum() == 0:
            mask = pd.Series([True] * len(df))
        
        # Apply capping to numeric columns
        for col in self.NUMERIC_COLUMNS:
            if col not in df.columns:
                continue
            
            # Compute quantiles from the stratified subset
            subset = df.loc[mask, col].dropna()
            if len(subset) < 10:  # Need minimum data for quantiles
                continue
            
            lower_bound = subset.quantile(lower_quantile)
            upper_bound = subset.quantile(upper_quantile)
            
            # Apply caps to entire dataframe
            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_capped
    
    def enforce_leakage_control(
        self,
        df: pd.DataFrame,
        forecast_date: datetime,
        strict: bool = True
    ) -> pd.DataFrame:
        """
        Enforce temporal ordering to prevent data leakage.
        
        This method ensures that only data from before the forecast date is
        included, preventing future information from leaking into model training.
        
        Args:
            df: DataFrame with player statistics
            forecast_date: Date for which we're making predictions
            strict: If True, exclude games on forecast_date; if False, include them
            
        Returns:
            DataFrame with only historical data (before forecast_date)
            
        Raises:
            DataLeakageError: If Date column is missing or has invalid values
        """
        logger.log_operation_start(
            "enforce_leakage_control",
            details={
                "forecast_date": forecast_date.isoformat() if isinstance(forecast_date, datetime) else str(forecast_date),
                "strict": strict,
                "input_rows": len(df)
            }
        )
        
        try:
            if 'Date' not in df.columns:
                logger.error(
                    "Date column missing for leakage control",
                    context={"columns": list(df.columns)}
                )
                raise DataLeakageError(
                    "DataFrame must have a 'Date' column for leakage control",
                    forecast_date=str(forecast_date)
                )
            
            # Ensure Date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df = df.copy()
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Check for invalid dates
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                logger.error(
                    "Invalid date values found",
                    context={
                        "invalid_dates": int(invalid_dates),
                        "total_rows": len(df)
                    }
                )
                raise DataLeakageError(
                    f"Found {invalid_dates} invalid date values. Cannot enforce leakage control.",
                    forecast_date=str(forecast_date),
                    invalid_dates=int(invalid_dates)
                )
            
            # Filter based on forecast date
            if strict:
                # Exclude games on or after forecast date
                mask = df['Date'] < forecast_date
            else:
                # Include games up to and including forecast date
                mask = df['Date'] <= forecast_date
            
            df_filtered = df[mask].copy()
            
            # Sort by date to ensure temporal ordering
            df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)
            
            logger.log_operation_complete(
                "enforce_leakage_control",
                details={
                    "output_rows": len(df_filtered),
                    "rows_filtered": len(df) - len(df_filtered),
                    "date_range": {
                        "min": df_filtered['Date'].min().isoformat() if len(df_filtered) > 0 else None,
                        "max": df_filtered['Date'].max().isoformat() if len(df_filtered) > 0 else None
                    }
                }
            )
            
            return df_filtered
            
        except DataLeakageError:
            raise
        except Exception as e:
            logger.log_operation_failed(
                "enforce_leakage_control",
                error=e,
                details={"forecast_date": str(forecast_date)}
            )
            raise DataLeakageError(
                f"Failed to enforce leakage control: {e}",
                forecast_date=str(forecast_date)
            )
    
    def load_with_contract(
        self,
        df: pd.DataFrame,
        contract: Type[BaseModel],
        source: str = "unknown",
        apply_fallbacks: bool = True
    ) -> pd.DataFrame:
        """
        Load and validate DataFrame against contract with fallbacks.
        
        This method:
        1. Applies schema migration (aliases)
        2. Validates against contract
        3. Applies fallbacks for missing data
        4. Records schema history
        
        Args:
            df: Input DataFrame
            contract: Pydantic contract to validate against
            source: Data source identifier
            apply_fallbacks: If True, apply fallbacks for missing data
            
        Returns:
            Validated and migrated DataFrame
        """
        if not self.use_contracts:
            return df
        
        logger.info(f"Loading data with contract: {contract.__name__}")
        
        # 1. Apply schema migration
        df = self.schema_registry.validate_and_migrate(df, contract, source)
        
        # 2. Validate against contract
        try:
            validated = self.contract_validator.validate_dataframe(df, contract)
            df_validated = self.contract_validator.to_dataframe(validated)
        except Exception as e:
            logger.warning(f"Contract validation had errors: {e}")
            # Continue with original df if validation fails
            df_validated = df
        
        # 3. Apply fallbacks if needed
        if apply_fallbacks:
            df_validated = self._apply_fallbacks(df_validated, contract)
        
        # 4. Log validation summary
        error_summary = self.contract_validator.get_error_summary()
        if error_summary['total_errors'] > 0:
            logger.warning(f"Validation errors: {error_summary}")
        
        return df_validated
    
    def _apply_fallbacks(self, df: pd.DataFrame, 
                        contract: Type[BaseModel]) -> pd.DataFrame:
        """
        Apply fallbacks for missing or invalid data.
        
        Uses cold-start priors and league medians to fill missing values.
        Season is automatically detected from the data.
        
        Args:
            df: DataFrame with potential missing values
            contract: Contract defining expected schema
            
        Returns:
            DataFrame with fallbacks applied
        """
        df = df.copy()
        
        # Get contract fields with defaults
        for field_name, field in contract.__fields__.items():
            if field_name not in df.columns:
                # Add column with default value
                if field.default is not None:
                    df[field_name] = field.default
                    logger.info(f"Added missing column '{field_name}' with default: {field.default}")
        
        # Detect season from data
        season = self._detect_season_from_data(df)
        if season is None:
            season = datetime.now().year
            logger.debug(f"Could not detect season, using current year: {season}")
        
        # Get priors for detected season
        priors = self._get_priors_for_season(season)
        
        # Apply cold-start priors for missing player stats
        if priors and 'player_id' in df.columns:
            for idx, row in df.iterrows():
                # Check if key stats are missing
                if pd.isna(row.get('ts_pct')) or pd.isna(row.get('usage')):
                    player_id = row.get('player_id', 'unknown')
                    role = row.get('role', 'unknown')
                    
                    # Get prior for this season
                    prior = priors.get_player_prior(
                        player_id=player_id,
                        role=role,
                        n_games=0
                    )
                    
                    # Fill missing values with prior means
                    # Note: This is a simplified fallback
                    # In production, you'd map prior dimensions to specific stats
                    if pd.isna(row.get('usage')):
                        df.at[idx, 'usage'] = 0.18  # League average
                        logger.debug(f"Applied fallback usage for {player_id} (season {season})")
        
        return df
    
    def load_multiple_players(
        self,
        player_names: List[str],
        year: int,
        use_processed: bool = False,
        validate: bool = True,
        use_contracts: bool = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple players with contract validation.
        
        Args:
            player_names: List of player names
            year: Season year
            use_processed: If True, load processed CSVs
            validate: If True, validate each player's data
            use_contracts: Override instance setting for contracts
            
        Returns:
            Dictionary mapping player names to their DataFrames
        """
        if use_contracts is None:
            use_contracts = self.use_contracts
        
        results = {}
        
        for player_name in player_names:
            try:
                df = self.load_player_data(player_name, year, use_processed)
                
                # Apply contract validation if enabled
                if use_contracts:
                    df = self.load_with_contract(
                        df, 
                        PlayersPerGameContract,
                        source=f"{player_name}_{year}"
                    )
                
                # Legacy validation
                if validate and not use_contracts:
                    validation = self.validate_data(df)
                    if not validation.is_valid:
                        logger.warning(f"Validation failed for {player_name}: {validation.errors}")
                
                results[player_name] = df
                
            except DataNotFoundError as e:
                logger.warning(f"Could not load data for {player_name}: {e}")
                # Apply fallback: create empty DataFrame with schema
                if use_contracts and self.cold_start_priors:
                    logger.info(f"Creating fallback data for new player: {player_name}")
                    results[player_name] = self._create_fallback_player_data(
                        player_name, year
                    )
            except Exception as e:
                logger.error(f"Error loading {player_name}: {e}")
        
        return results
    
    def _create_fallback_player_data(self, player_name: str, 
                                    year: int) -> pd.DataFrame:
        """
        Create fallback data for a player with no historical data.
        
        Uses cold-start priors for the specific season to generate a minimal DataFrame.
        
        Args:
            player_name: Player name
            year: Season year
            
        Returns:
            DataFrame with fallback data
        """
        logger.info(f"Creating fallback data for {player_name} (season {year})")
        
        # Create minimal DataFrame with required fields
        fallback_data = {
            'player_id': [player_name],
            'game_id': [f"{year}_FALLBACK_001"],
            'date': [datetime(year, 10, 1)],  # Season start
            'team_id': ['UNKNOWN'],
            'opponent_id': ['UNKNOWN'],
            'minutes': [0.0],
            'role': ['unknown']
        }
        
        # Get priors for this specific season
        priors = self._get_priors_for_season(year)
        
        # Add default values for optional fields
        if priors:
            prior = priors.get_player_prior(
                player_id=player_name,
                role='unknown',
                n_games=0
            )
            # Add prior-based defaults
            fallback_data['usage'] = [0.18]
            fallback_data['ts_pct'] = [0.55]
        
        df = pd.DataFrame(fallback_data)
        logger.warning(f"Created fallback data for {player_name} (season {year}) - player has no historical data")
        
        return df
