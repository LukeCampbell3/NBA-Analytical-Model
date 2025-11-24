"""Local assist model using logistic regression.

This module implements event-level assist prediction using proxy features
until tracking data becomes available. Features include passer_usage,
passer_ast_pct, receiver_shot_quality_proxy, opponent_help_nail_freq,
and lane_risk_proxy.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Optional, Dict, Any
import joblib


class AssistModel:
    """Event-level assist prediction model.
    
    Uses logistic regression to predict assist probability based on
    game-slice features. Designed to work with proxy features until
    tracking data becomes available.
    """
    
    def __init__(self):
        """Initialize the assist model."""
        self.model: Optional[LogisticRegression] = None
        self.feature_columns = [
            'passer_usage',
            'passer_ast_pct',
            'receiver_shot_quality_proxy',
            'opponent_help_nail_freq',
            'lane_risk_proxy'
        ]
    
    def featurize_assist(self, game_slice_df: pd.DataFrame) -> pd.DataFrame:
        """Extract assist-specific features from game slice data.
        
        Computes proxy features for assist events:
        - passer_usage: Passer's usage rate (possessions per minute)
        - passer_ast_pct: Passer's historical assist percentage
        - receiver_shot_quality_proxy: Quality of receiver's shot opportunity
        - opponent_help_nail_freq: Opponent's help defense frequency
        - lane_risk_proxy: Risk of turnover in the lane
        
        Args:
            game_slice_df: DataFrame with game slice data containing:
                - passer_possessions: Number of possessions for passer
                - passer_minutes: Minutes played by passer
                - passer_assists: Historical assists by passer
                - passer_fga: Historical field goal attempts by passer
                - receiver_ts_pct: Receiver's true shooting percentage
                - receiver_distance_to_basket: Distance of shot from basket
                - receiver_open_shot: Whether receiver is open (boolean)
                - opponent_help_rate: Opponent's help defense rate
                - opponent_nail_rate: Opponent's nail help frequency
                - passer_tov_rate: Passer's turnover rate
                - pass_distance: Distance of the pass
                - pass_through_lane: Whether pass goes through lane (boolean)
                
        Returns:
            DataFrame with assist features added
        """
        df = game_slice_df.copy()
        
        # Passer usage: possessions per minute
        if 'passer_possessions' in df.columns and 'passer_minutes' in df.columns:
            df['passer_usage'] = df['passer_possessions'] / (df['passer_minutes'] + 1e-6)
        else:
            # Default to moderate usage
            df['passer_usage'] = 0.20  # ~20% usage rate
        
        # Passer assist percentage: assists per FGA
        if 'passer_assists' in df.columns and 'passer_fga' in df.columns:
            df['passer_ast_pct'] = df['passer_assists'] / (df['passer_fga'] + 1e-6)
        else:
            # Default to league average
            df['passer_ast_pct'] = 0.15  # ~15% assist rate
        
        # Receiver shot quality proxy: combination of TS%, distance, and openness
        shot_quality_components = []
        
        if 'receiver_ts_pct' in df.columns:
            # Normalize TS% (typical range 0.45-0.65)
            shot_quality_components.append((df['receiver_ts_pct'] - 0.45) / 0.20)
        
        if 'receiver_distance_to_basket' in df.columns:
            # Closer shots are higher quality (normalize by 25 feet)
            shot_quality_components.append(1.0 - df['receiver_distance_to_basket'] / 25.0)
        
        if 'receiver_open_shot' in df.columns:
            # Open shots are higher quality
            shot_quality_components.append(df['receiver_open_shot'].astype(float) * 0.3)
        
        if shot_quality_components:
            df['receiver_shot_quality_proxy'] = np.clip(
                np.mean(shot_quality_components, axis=0), 0, 1
            )
        else:
            # Default to moderate shot quality
            df['receiver_shot_quality_proxy'] = 0.5
        
        # Opponent help nail frequency: how often opponent helps and nails
        if 'opponent_help_rate' in df.columns and 'opponent_nail_rate' in df.columns:
            df['opponent_help_nail_freq'] = df['opponent_help_rate'] * df['opponent_nail_rate']
        elif 'opponent_help_rate' in df.columns:
            df['opponent_help_nail_freq'] = df['opponent_help_rate'] * 0.5  # Assume 50% nail rate
        else:
            # Default to league average
            df['opponent_help_nail_freq'] = 0.15  # ~15% help-nail frequency
        
        # Lane risk proxy: risk of turnover when passing through lane
        lane_risk_components = []
        
        if 'passer_tov_rate' in df.columns:
            # Higher turnover rate = higher risk
            lane_risk_components.append(df['passer_tov_rate'])
        
        if 'pass_distance' in df.columns:
            # Longer passes = higher risk (normalize by 30 feet)
            lane_risk_components.append(df['pass_distance'] / 30.0)
        
        if 'pass_through_lane' in df.columns:
            # Passes through lane are riskier
            lane_risk_components.append(df['pass_through_lane'].astype(float) * 0.3)
        
        if 'opponent_help_nail_freq' in df.columns:
            # Higher opponent help frequency = higher risk
            lane_risk_components.append(df['opponent_help_nail_freq'])
        
        if lane_risk_components:
            df['lane_risk_proxy'] = np.clip(
                np.mean(lane_risk_components, axis=0), 0, 1
            )
        else:
            # Default to moderate risk
            df['lane_risk_proxy'] = 0.3
        
        return df
    
    def fit_assist_logit(
        self, 
        df: pd.DataFrame,
        target_col: str = 'assist_success',
        cv_folds: int = 5,
        random_state: int = 42,
        **logit_kwargs
    ) -> LogisticRegression:
        """Fit logistic regression model for assist prediction.
        
        Args:
            df: DataFrame with featurized assist events
            target_col: Name of binary target column (1 = assist, 0 = no assist)
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
    
    def predict_assist_prob(
        self, 
        model: LogisticRegression, 
        df: pd.DataFrame
    ) -> np.ndarray:
        """Predict assist probabilities for new data.
        
        Args:
            model: Fitted LogisticRegression model
            df: DataFrame with featurized assist events
            
        Returns:
            Array of assist probabilities (values in [0, 1])
            
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
        
        # Predict probabilities (class 1 = assist)
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
