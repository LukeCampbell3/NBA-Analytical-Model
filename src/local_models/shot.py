"""Local shot model using logistic regression.

This module implements event-level shot prediction using proxy features
until tracking data becomes available. Features include shooter_ts_context,
distance_bin, pullup_vs_catch_proxy, and opponent_rim_deterrence.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Optional, Dict, Any
import joblib


class ShotModel:
    """Event-level shot prediction model.
    
    Uses logistic regression to predict shot success probability based on
    game-slice features. Designed to work with proxy features until
    tracking data becomes available.
    """
    
    def __init__(self):
        """Initialize the shot model."""
        self.model: Optional[LogisticRegression] = None
        self.feature_columns = [
            'shooter_ts_context',
            'distance_bin',
            'pullup_vs_catch_proxy',
            'opponent_rim_deterrence'
        ]
    
    def featurize_shot(self, game_slice_df: pd.DataFrame) -> pd.DataFrame:
        """Extract shot-specific features from game slice data.
        
        Computes proxy features for shot events:
        - shooter_ts_context: Shooter's recent true shooting percentage
        - distance_bin: Categorical distance from basket (rim/mid/three)
        - pullup_vs_catch_proxy: Whether shot is off-the-dribble or catch-and-shoot
        - opponent_rim_deterrence: Opponent's rim protection strength
        
        Args:
            game_slice_df: DataFrame with game slice data containing:
                - shooter_ts_pct: Shooter's true shooting percentage
                - shooter_recent_ts: Recent TS% (last 5 games)
                - shot_distance: Distance from basket in feet
                - shooter_dribbles_before_shot: Number of dribbles before shot
                - shooter_time_of_possession: Time holding ball before shot
                - shooter_catch_and_shoot_pct: Historical catch-and-shoot percentage
                - opponent_rim_protection: Opponent's rim protection rating
                - opponent_blk_pct: Opponent's block percentage
                - defender_distance: Distance of nearest defender
                
        Returns:
            DataFrame with shot features added
        """
        df = game_slice_df.copy()
        
        # Shooter TS context: recent performance weighted with season average
        if 'shooter_recent_ts' in df.columns and 'shooter_ts_pct' in df.columns:
            # Weight recent performance more heavily (70/30 split)
            df['shooter_ts_context'] = (
                0.7 * df['shooter_recent_ts'] + 
                0.3 * df['shooter_ts_pct']
            )
        elif 'shooter_ts_pct' in df.columns:
            df['shooter_ts_context'] = df['shooter_ts_pct']
        else:
            # Default to league average TS%
            df['shooter_ts_context'] = 0.56
        
        # Distance bin: convert continuous distance to categorical
        # 0 = rim (0-5 ft), 1 = mid (5-23 ft), 2 = three (23+ ft)
        if 'shot_distance' in df.columns:
            df['distance_bin'] = pd.cut(
                df['shot_distance'],
                bins=[-np.inf, 5, 23, np.inf],
                labels=[0, 1, 2]
            ).astype(float)
        else:
            # Default to mid-range
            df['distance_bin'] = 1.0
        
        # Pullup vs catch proxy: higher value = more off-the-dribble
        pullup_indicators = []
        
        if 'shooter_dribbles_before_shot' in df.columns:
            # More dribbles = more pullup (normalize by 5 dribbles)
            pullup_indicators.append(
                np.clip(df['shooter_dribbles_before_shot'] / 5.0, 0, 1)
            )
        
        if 'shooter_time_of_possession' in df.columns:
            # More time = more pullup (normalize by 3 seconds)
            pullup_indicators.append(
                np.clip(df['shooter_time_of_possession'] / 3.0, 0, 1)
            )
        
        if 'shooter_catch_and_shoot_pct' in df.columns:
            # Lower catch-and-shoot % = more pullup
            pullup_indicators.append(1.0 - df['shooter_catch_and_shoot_pct'])
        
        if pullup_indicators:
            df['pullup_vs_catch_proxy'] = np.clip(
                np.mean(pullup_indicators, axis=0), 0, 1
            )
        else:
            # Default to balanced (50/50 pullup vs catch)
            df['pullup_vs_catch_proxy'] = 0.5
        
        # Opponent rim deterrence: combination of rim protection and blocks
        rim_deterrence_components = []
        
        if 'opponent_rim_protection' in df.columns:
            # Normalize rim protection rating (typical range 0.9-1.1)
            rim_deterrence_components.append(
                (df['opponent_rim_protection'] - 0.9) / 0.2
            )
        
        if 'opponent_blk_pct' in df.columns:
            # Normalize block percentage (typical range 0.02-0.08)
            rim_deterrence_components.append(
                (df['opponent_blk_pct'] - 0.02) / 0.06
            )
        
        if 'defender_distance' in df.columns:
            # Closer defender = higher deterrence (normalize by 6 feet)
            rim_deterrence_components.append(
                np.clip(1.0 - df['defender_distance'] / 6.0, 0, 1)
            )
        
        if rim_deterrence_components:
            df['opponent_rim_deterrence'] = np.clip(
                np.mean(rim_deterrence_components, axis=0), 0, 1
            )
        else:
            # Default to moderate rim deterrence
            df['opponent_rim_deterrence'] = 0.5
        
        return df
    
    def fit_shot_logit(
        self, 
        df: pd.DataFrame,
        target_col: str = 'shot_made',
        cv_folds: int = 5,
        random_state: int = 42,
        **logit_kwargs
    ) -> LogisticRegression:
        """Fit logistic regression model for shot prediction.
        
        Args:
            df: DataFrame with featurized shot events
            target_col: Name of binary target column (1 = made, 0 = missed)
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            **logit_kwargs: Additional arguments for LogisticRegression
            
        Returns:
            Fitted LogisticRegression model
            
        Raises:
            ValueError: If required features are missing or target is invalid
        """
        # Validate features
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Validate target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Prepare data
        X = df[self.feature_columns].copy()
        y = df[target_col].values
        
        # Check for valid binary target
        unique_values = np.unique(y)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(f"Target must be binary (0/1), found: {unique_values}")
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Set default logistic regression parameters
        default_params = {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': random_state,
            'class_weight': 'balanced'  # Handle class imbalance
        }
        default_params.update(logit_kwargs)
        
        # Fit model
        self.model = LogisticRegression(**default_params)
        self.model.fit(X, y)
        
        # Perform cross-validation
        if cv_folds > 1:
            cv_scores = cross_val_score(
                self.model, X, y, 
                cv=cv_folds, 
                scoring='roc_auc'
            )
            print(f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return self.model
    
    def predict_shot_prob(
        self, 
        model: LogisticRegression, 
        df: pd.DataFrame
    ) -> np.ndarray:
        """Predict shot success probabilities for new data.
        
        Args:
            model: Fitted LogisticRegression model
            df: DataFrame with featurized shot events
            
        Returns:
            Array of shot success probabilities (values in [0, 1])
            
        Raises:
            ValueError: If required features are missing
        """
        # Validate features
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare data
        X = df[self.feature_columns].copy()
        X = X.fillna(X.mean())
        
        # Predict probabilities (class 1 = shot made)
        probs = model.predict_proba(X)[:, 1]
        
        return probs
    
    def save_model(self, model: LogisticRegression, path: str) -> None:
        """Save trained model to disk.
        
        Args:
            model: Fitted LogisticRegression model
            path: File path to save model
        """
        joblib.dump(model, path)
    
    def load_model(self, path: str) -> LogisticRegression:
        """Load trained model from disk.
        
        Args:
            path: File path to load model from
            
        Returns:
            Loaded LogisticRegression model
        """
        self.model = joblib.load(path)
        return self.model
    
    def get_feature_importance(self, model: LogisticRegression) -> Dict[str, float]:
        """Get feature importance from fitted model.
        
        Args:
            model: Fitted LogisticRegression model
            
        Returns:
            Dictionary mapping feature names to coefficients
        """
        if model is None:
            raise ValueError("Model has not been fitted yet")
        
        return dict(zip(self.feature_columns, model.coef_[0]))
