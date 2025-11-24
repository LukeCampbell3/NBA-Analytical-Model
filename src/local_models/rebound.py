"""Local rebound model using logistic regression.

This module implements event-level rebound prediction using proxy features
until tracking data becomes available. Features include time_to_ball_proxy,
crowd_index, reach_margin, and seal_angle_proxy.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Optional, Dict, Any
import joblib


class ReboundModel:
    """Event-level rebound prediction model.
    
    Uses logistic regression to predict rebound probability based on
    game-slice features. Designed to work with proxy features until
    tracking data becomes available.
    """
    
    def __init__(self):
        """Initialize the rebound model."""
        self.model: Optional[LogisticRegression] = None
        self.feature_columns = [
            'time_to_ball_proxy',
            'crowd_index',
            'reach_margin',
            'seal_angle_proxy'
        ]
    
    def featurize_rebound(self, game_slice_df: pd.DataFrame) -> pd.DataFrame:
        """Extract rebound-specific features from game slice data.
        
        Computes proxy features for rebound events:
        - time_to_ball_proxy: Estimated time to reach rebound position
        - crowd_index: Number of nearby players competing for rebound
        - reach_margin: Player height/wingspan advantage over competitors
        - seal_angle_proxy: Quality of box-out position
        
        Args:
            game_slice_df: DataFrame with game slice data containing:
                - player_height: Player height in inches
                - player_wingspan: Player wingspan in inches (optional)
                - distance_to_basket: Distance from basket at shot time
                - nearby_players: Count of players within 5 feet
                - box_out_quality: Box-out positioning score (0-1)
                - opponent_avg_height: Average height of nearby opponents
                - player_speed: Player speed rating (optional)
                
        Returns:
            DataFrame with rebound features added
        """
        df = game_slice_df.copy()
        
        # Time to ball proxy: based on distance and speed
        # Assumes average speed if not provided
        if 'player_speed' in df.columns:
            df['time_to_ball_proxy'] = df['distance_to_basket'] / (df['player_speed'] + 1e-6)
        else:
            # Use distance as proxy (normalized)
            df['time_to_ball_proxy'] = df['distance_to_basket'] / 15.0  # Normalize by typical distance
        
        # Crowd index: number of nearby players competing
        if 'nearby_players' in df.columns:
            df['crowd_index'] = df['nearby_players']
        else:
            # Default to moderate crowding
            df['crowd_index'] = 2.5
        
        # Reach margin: height/wingspan advantage
        player_reach = df.get('player_wingspan', df.get('player_height', 80) * 1.05)
        opponent_reach = df.get('opponent_avg_height', 80) * 1.05
        df['reach_margin'] = (player_reach - opponent_reach) / 10.0  # Normalize by 10 inches
        
        # Seal angle proxy: box-out quality
        if 'box_out_quality' in df.columns:
            df['seal_angle_proxy'] = df['box_out_quality']
        else:
            # Estimate from position: closer to basket = better seal
            df['seal_angle_proxy'] = np.clip(1.0 - df['distance_to_basket'] / 20.0, 0, 1)
        
        return df
    
    def fit_rebound_logit(
        self, 
        df: pd.DataFrame,
        target_col: str = 'rebound_success',
        cv_folds: int = 5,
        random_state: int = 42,
        **logit_kwargs
    ) -> LogisticRegression:
        """Fit logistic regression model for rebound prediction.
        
        Args:
            df: DataFrame with featurized rebound events
            target_col: Name of binary target column (1 = rebound, 0 = no rebound)
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
    
    def predict_rebound_prob(
        self, 
        model: LogisticRegression, 
        df: pd.DataFrame
    ) -> np.ndarray:
        """Predict rebound probabilities for new data.
        
        Args:
            model: Fitted LogisticRegression model
            df: DataFrame with featurized rebound events
            
        Returns:
            Array of rebound probabilities (values in [0, 1])
            
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
        
        # Predict probabilities (class 1 = rebound)
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
