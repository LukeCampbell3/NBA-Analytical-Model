"""
Feature engineering and transformation module for NBA player statistics.

This module provides functionality to compute rolling statistics with exponential decay,
calculate player posterior distributions, apply robust scaling, and join contextual features.

Integrated with:
- Cold-start priors for new players
- Graceful fallbacks for insufficient data
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

if TYPE_CHECKING:
    from src.priors import ColdStartPriors

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PosteriorParams:
    """
    Player posterior distribution parameters.
    
    Attributes:
        mu: Mean vector of the posterior distribution
        Sigma: Covariance matrix of the posterior distribution
        player_id: Unique identifier for the player
        as_of_date: Date when the posterior was computed
        feature_names: Names of features in the posterior (optional)
    """
    mu: np.ndarray
    Sigma: np.ndarray
    player_id: str
    as_of_date: datetime
    feature_names: Optional[List[str]] = None


@dataclass
class RobustScalerParams:
    """
    Parameters for RobustScaler transformation.
    
    Attributes:
        center: Median values for centering
        scale: IQR values for scaling
        feature_names: Names of features
    """
    center: np.ndarray
    scale: np.ndarray
    feature_names: List[str]


class FeatureTransform:
    """
    Feature engineering and transformation for player statistics.
    
    This class handles:
    - Computing rolling window statistics with exponential decay
    - Calculating player posterior distributions (mu, Sigma)
    - Applying RobustScaler transformations
    - Joining player, opponent, and rotation context
    """
    
    # Core performance attributes for posteriors
    CORE_ATTRIBUTES = [
        'TS%', 'USG%', 'AST%', 'TOV%', 'TRB%', 'STL%', 'BLK%'
    ]
    
    # Box stats for rolling features
    BOX_STATS = [
        'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'MP'
    ]
    
    def __init__(self, window_games: int = 20, decay_half_life: int = 7,
                 use_cold_start: bool = True):
        """
        Initialize the FeatureTransform.
        
        Season is automatically detected from the data being processed.
        No need to specify season - the transformer adapts to any season's data.
        
        Args:
            window_games: Number of games to include in rolling window (default: 20)
            decay_half_life: Half-life for exponential decay weighting in games (default: 7)
            use_cold_start: If True, use cold-start priors for new players
        """
        self.window_games = window_games
        self.decay_half_life = decay_half_life
        self.use_cold_start = use_cold_start
        
        # Compute decay constant: weight = exp(-lambda * t)
        # At half-life, weight = 0.5, so: 0.5 = exp(-lambda * half_life)
        # lambda = -ln(0.5) / half_life
        self.decay_lambda = np.log(2) / decay_half_life
        
        # Cold-start priors cache (season -> ColdStartPriors)
        # Priors are loaded on-demand when data is processed
        if use_cold_start:
            self._priors_cache = {}
            logger.info("FeatureTransform initialized with cold-start support (season auto-detected)")
        else:
            self._priors_cache = None
    
    def _get_priors_for_season(self, season: int) -> Optional['ColdStartPriors']:
        """
        Get cold-start priors for a specific season.
        
        Priors are cached per season and loaded on-demand.
        
        Args:
            season: Season year
            
        Returns:
            ColdStartPriors for the season, or None if unavailable
        """
        if self._priors_cache is None:
            return None
        
        if season not in self._priors_cache:
            try:
                from src.priors import ColdStartPriors
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
    
    def _compute_exponential_weights(self, n: int) -> np.ndarray:
        """
        Compute exponential decay weights for n games.
        
        Most recent game has weight 1.0, older games decay exponentially.
        
        Args:
            n: Number of games
            
        Returns:
            Array of weights, shape (n,), with most recent game last
        """
        # Time indices: 0 (oldest) to n-1 (most recent)
        time_indices = np.arange(n)
        # Reverse so most recent has t=0
        time_from_present = n - 1 - time_indices
        # Compute weights
        weights = np.exp(-self.decay_lambda * time_from_present)
        return weights
    
    def compute_rolling_features(
        self,
        df: pd.DataFrame,
        player_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute rolling window statistics with exponential decay.
        
        For each game, computes weighted mean and variance over the previous
        window_games games, with exponential decay giving more weight to recent games.
        
        Args:
            df: DataFrame with player game statistics, sorted by date
            player_id: Optional player identifier to add to output
            
        Returns:
            DataFrame with rolling features added
        """
        df_rolling = df.copy()
        
        # Ensure data is sorted by date
        if 'Date' in df.columns:
            df_rolling = df_rolling.sort_values('Date').reset_index(drop=True)
        
        # Compute rolling features for each stat
        for stat in self.BOX_STATS:
            if stat not in df.columns:
                continue
            
            # Initialize columns
            df_rolling[f'{stat}_rolling_mean'] = np.nan
            df_rolling[f'{stat}_rolling_std'] = np.nan
            
            # Compute for each row
            for i in range(len(df_rolling)):
                # Get window of previous games
                start_idx = max(0, i - self.window_games)
                window_data = df_rolling[stat].iloc[start_idx:i]
                
                if len(window_data) == 0:
                    continue
                
                # Compute exponential weights
                weights = self._compute_exponential_weights(len(window_data))
                
                # Normalize weights to sum to 1
                weights = weights / weights.sum()
                
                # Compute weighted mean
                weighted_mean = np.average(window_data, weights=weights)
                df_rolling.loc[i, f'{stat}_rolling_mean'] = weighted_mean
                
                # Compute weighted variance
                weighted_var = np.average(
                    (window_data - weighted_mean) ** 2,
                    weights=weights
                )
                df_rolling.loc[i, f'{stat}_rolling_std'] = np.sqrt(weighted_var)
        
        # Add player_id if provided
        if player_id is not None:
            df_rolling['player_id'] = player_id
        
        return df_rolling
    
    def compute_player_posteriors(
        self,
        df: pd.DataFrame,
        player_id: Optional[str] = None,
        as_of_date: Optional[datetime] = None
    ) -> PosteriorParams:
        """
        Compute player posterior distribution (mu, Sigma) from historical data.
        
        Uses the most recent window_games to estimate the player's capability
        distribution. Applies exponential decay weighting to emphasize recent performance.
        
        Args:
            df: DataFrame with player statistics, sorted by date
            player_id: Player identifier
            as_of_date: Date for the posterior (uses last date in df if None)
            
        Returns:
            PosteriorParams with mu (mean vector) and Sigma (covariance matrix)
            
        Raises:
            ValueError: If required attributes are missing or insufficient data
        """
        # Check for required attributes
        missing_attrs = set(self.CORE_ATTRIBUTES) - set(df.columns)
        if missing_attrs:
            raise ValueError(f"Missing required attributes: {missing_attrs}")
        
        # Ensure data is sorted by date
        df_sorted = df.copy()
        if 'Date' in df.columns:
            df_sorted = df_sorted.sort_values('Date').reset_index(drop=True)
        
        # Get the most recent window_games
        window_data = df_sorted[self.CORE_ATTRIBUTES].tail(self.window_games)
        
        if len(window_data) < 3:
            raise ValueError(
                f"Insufficient data for posterior computation: "
                f"need at least 3 games, got {len(window_data)}"
            )
        
        # Drop rows with missing values
        window_data = window_data.dropna()
        
        if len(window_data) < 3:
            raise ValueError(
                "Insufficient valid data after removing missing values"
            )
        
        # Compute exponential weights
        weights = self._compute_exponential_weights(len(window_data))
        weights = weights / weights.sum()
        
        # Convert to numpy array
        X = window_data.values
        
        # Compute weighted mean (mu)
        mu = np.average(X, axis=0, weights=weights)
        
        # Compute weighted covariance (Sigma)
        # Center the data
        X_centered = X - mu
        
        # Weighted covariance: Sigma = (X - mu)^T W (X - mu) / (1 - sum(w^2))
        # where W is diagonal matrix of weights
        W = np.diag(weights)
        Sigma = X_centered.T @ W @ X_centered
        
        # Bias correction factor
        bias_correction = 1.0 / (1.0 - np.sum(weights ** 2))
        Sigma = Sigma * bias_correction
        
        # Add small regularization to ensure positive definite
        ridge = 1e-6
        Sigma = Sigma + ridge * np.eye(len(mu))
        
        # Determine as_of_date
        if as_of_date is None:
            if 'Date' in df_sorted.columns:
                as_of_date = df_sorted['Date'].iloc[-1]
            else:
                as_of_date = datetime.now()
        
        # Determine player_id
        if player_id is None:
            if 'player_id' in df_sorted.columns:
                player_id = df_sorted['player_id'].iloc[0]
            elif 'Player' in df_sorted.columns:
                player_id = df_sorted['Player'].iloc[0]
            else:
                player_id = "unknown"
        
        return PosteriorParams(
            mu=mu,
            Sigma=Sigma,
            player_id=str(player_id),
            as_of_date=as_of_date,
            feature_names=self.CORE_ATTRIBUTES
        )
    
    def compute_player_posteriors_with_fallback(
        self,
        df: pd.DataFrame,
        player_id: Optional[str] = None,
        role: Optional[str] = None,
        as_of_date: Optional[datetime] = None,
        player_info: Optional[Dict] = None
    ) -> PosteriorParams:
        """
        Compute player posterior with cold-start fallback.
        
        This method attempts to compute posteriors from data, but falls back
        to cold-start priors if:
        - Insufficient data (< 3 games)
        - Missing required attributes
        - New player with no history
        
        Args:
            df: DataFrame with player statistics
            player_id: Player identifier
            role: Player role (for cold-start)
            as_of_date: Date for the posterior
            player_info: Additional player info for role inference
            
        Returns:
            PosteriorParams (from data or cold-start prior)
        """
        # Determine player_id if not provided
        if player_id is None:
            if 'player_id' in df.columns and len(df) > 0:
                player_id = df['player_id'].iloc[0]
            elif 'Player' in df.columns and len(df) > 0:
                player_id = df['Player'].iloc[0]
            else:
                player_id = "unknown"
        
        # Determine role if not provided
        if role is None and 'role' in df.columns and len(df) > 0:
            role = df['role'].iloc[0]
        
        # Try to compute from data
        try:
            posterior = self.compute_player_posteriors(df, player_id, as_of_date)
            logger.info(f"Computed posterior from {len(df)} games for {player_id}")
            return posterior
            
        except (ValueError, KeyError) as e:
            # Fall back to cold-start prior
            logger.warning(f"Could not compute posterior for {player_id}: {e}")
            
            # Detect season from data
            season = self._detect_season_from_data(df)
            if season is None:
                season = datetime.now().year
                logger.debug(f"Could not detect season, using current year: {season}")
            
            # Get priors for detected season
            priors = self._get_priors_for_season(season)
            
            if priors:
                logger.info(f"Using cold-start prior for {player_id} (season {season})")
                
                # Get number of games for uncertainty scaling
                n_games = len(df) if df is not None else 0
                
                # Get cold-start prior
                prior = priors.get_player_prior(
                    player_id=player_id,
                    role=role,
                    n_games=n_games,
                    player_info=player_info
                )
                
                # If we have some data, update prior with it
                if n_games > 0 and n_games < 3:
                    try:
                        # Extract recent games data
                        recent_data = df[self.CORE_ATTRIBUTES].dropna().values
                        if len(recent_data) > 0:
                            prior = priors.update_with_data(
                                player_id=player_id,
                                recent_games=recent_data,
                                prior=prior,
                                n_games=n_games
                            )
                            logger.info(f"Updated prior with {n_games} games for {player_id} (season {season})")
                    except Exception as update_error:
                        logger.warning(f"Could not update prior: {update_error}")
                
                return prior
            else:
                # No cold-start priors available, raise original error
                logger.error(f"No cold-start priors available for {player_id}")
                raise
    
    def compute_scalers(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> RobustScalerParams:
        """
        Compute RobustScaler parameters from training data.
        
        RobustScaler uses median and IQR for scaling, making it robust to outliers.
        
        Args:
            X: DataFrame with features to scale
            feature_names: Optional list of feature names (uses X.columns if None)
            
        Returns:
            RobustScalerParams with center and scale parameters
        """
        if feature_names is None:
            feature_names = list(X.columns)
        
        # Fit RobustScaler
        scaler = RobustScaler()
        scaler.fit(X[feature_names])
        
        return RobustScalerParams(
            center=scaler.center_,
            scale=scaler.scale_,
            feature_names=feature_names
        )
    
    def apply_scalers(
        self,
        X: pd.DataFrame,
        params: RobustScalerParams
    ) -> pd.DataFrame:
        """
        Apply RobustScaler transformation to features.
        
        Args:
            X: DataFrame with features to scale
            params: RobustScalerParams with scaling parameters
            
        Returns:
            DataFrame with scaled features
        """
        X_scaled = X.copy()
        
        # Apply scaling: (X - center) / scale
        for i, feature in enumerate(params.feature_names):
            if feature in X_scaled.columns:
                X_scaled[feature] = (
                    (X_scaled[feature] - params.center[i]) / params.scale[i]
                )
        
        return X_scaled
    
    def join_context(
        self,
        df_player: pd.DataFrame,
        df_opponent: Optional[pd.DataFrame] = None,
        df_rotation: Optional[pd.DataFrame] = None,
        on_date: bool = True
    ) -> pd.DataFrame:
        """
        Join player data with opponent features and rotation priors.
        
        This creates a unified feature matrix combining player statistics,
        opponent defensive characteristics, and rotation context.
        
        Args:
            df_player: DataFrame with player statistics
            df_opponent: Optional DataFrame with opponent features
            df_rotation: Optional DataFrame with rotation priors
            on_date: If True, join on Date column; otherwise use index
            
        Returns:
            DataFrame with joined features
        """
        df_joined = df_player.copy()
        
        # Join opponent features
        if df_opponent is not None:
            if on_date and 'Date' in df_player.columns and 'Date' in df_opponent.columns:
                # Merge on date and opponent_id if available
                if 'opponent_id' in df_player.columns and 'opponent_id' in df_opponent.columns:
                    df_joined = df_joined.merge(
                        df_opponent,
                        on=['Date', 'opponent_id'],
                        how='left',
                        suffixes=('', '_opp')
                    )
                else:
                    # Merge on date only
                    df_joined = df_joined.merge(
                        df_opponent,
                        on='Date',
                        how='left',
                        suffixes=('', '_opp')
                    )
            else:
                # Join on index
                df_joined = df_joined.join(
                    df_opponent,
                    how='left',
                    rsuffix='_opp'
                )
        
        # Join rotation priors
        if df_rotation is not None:
            if on_date and 'Date' in df_joined.columns and 'Date' in df_rotation.columns:
                # Merge on date and player_id if available
                if 'player_id' in df_joined.columns and 'player_id' in df_rotation.columns:
                    df_joined = df_joined.merge(
                        df_rotation,
                        on=['Date', 'player_id'],
                        how='left',
                        suffixes=('', '_rot')
                    )
                else:
                    # Merge on date only
                    df_joined = df_joined.merge(
                        df_rotation,
                        on='Date',
                        how='left',
                        suffixes=('', '_rot')
                    )
            else:
                # Join on index
                df_joined = df_joined.join(
                    df_rotation,
                    how='left',
                    rsuffix='_rot'
                )
        
        return df_joined
    
    def compute_all_features(
        self,
        df: pd.DataFrame,
        player_id: Optional[str] = None,
        df_opponent: Optional[pd.DataFrame] = None,
        df_rotation: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, PosteriorParams]:
        """
        Compute all features: rolling stats, posteriors, and context joining.
        
        This is a convenience method that applies the full feature engineering pipeline.
        
        Args:
            df: DataFrame with player game statistics
            player_id: Player identifier
            df_opponent: Optional opponent features
            df_rotation: Optional rotation priors
            
        Returns:
            Tuple of (feature DataFrame, posterior parameters)
        """
        # Compute rolling features
        df_features = self.compute_rolling_features(df, player_id)
        
        # Compute posteriors
        posteriors = self.compute_player_posteriors(df, player_id)
        
        # Join context
        df_features = self.join_context(df_features, df_opponent, df_rotation)
        
        return df_features, posteriors
