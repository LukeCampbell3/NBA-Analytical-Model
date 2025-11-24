"""
Baseline machine learning models for benchmarking.

This module provides traditional ML approaches (Ridge, XGBoost, MLP) for
comparison against the capability-region simulation approach. These models
serve as baselines to evaluate the performance gains of the geometric approach.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineModelConfig:
    """Configuration for baseline models."""
    # Ridge parameters
    ridge_alpha: float = 1.0
    
    # XGBoost parameters
    xgb_max_depth: int = 6
    xgb_n_estimators: int = 500
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # MLP parameters
    mlp_hidden_layers: List[int] = None
    mlp_dropout: float = 0.1
    mlp_max_iter: int = 500
    mlp_learning_rate_init: float = 0.001
    
    def __post_init__(self):
        if self.mlp_hidden_layers is None:
            self.mlp_hidden_layers = [128, 64]


class BaselineModels:
    """
    Traditional ML models for NBA player performance prediction.
    
    This class provides implementations of Ridge regression, XGBoost, and MLP
    models that serve as baselines for comparison. All models predict box
    statistics (PTS, REB, AST, etc.) from rolling features, opponent context,
    and player role information.
    """
    
    # Target statistics to predict
    TARGET_STATS = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'MP']
    
    # Rolling window sizes for feature engineering
    ROLLING_WINDOWS = [5, 10, 20]
    
    def __init__(self, config: Optional[BaselineModelConfig] = None):
        """
        Initialize BaselineModels.
        
        Args:
            config: Configuration for model hyperparameters
        """
        self.config = config or BaselineModelConfig()
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        
        # Check if xgboost is available
        self.xgboost_available = False
        try:
            import xgboost as xgb
            self.xgboost_available = True
            self.xgb = xgb
        except ImportError:
            print("Warning: xgboost not installed. XGBoost models will not be available.")
    
    def build_features(
        self,
        df: pd.DataFrame,
        include_opponent: bool = True,
        include_role: bool = True
    ) -> pd.DataFrame:
        """
        Build feature matrix from player data.
        
        Creates features including:
        - Rolling means and variances for box stats
        - Opponent defensive features
        - Player role indicators
        - Pace and context features
        
        Args:
            df: DataFrame with player game statistics
            include_opponent: If True, include opponent features
            include_role: If True, include role indicators
            
        Returns:
            DataFrame with engineered features
        """
        df_features = pd.DataFrame(index=df.index)
        
        # Add rolling means and variances for target stats
        for stat in self.TARGET_STATS:
            if stat not in df.columns:
                continue
            
            for window in self.ROLLING_WINDOWS:
                # Rolling mean
                col_mean = f'{stat}_rolling_{window}_mean'
                df_features[col_mean] = df[stat].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling variance (std)
                col_std = f'{stat}_rolling_{window}_std'
                df_features[col_std] = df[stat].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)
        
        # Add advanced stats if available
        advanced_stats = ['TS%', 'USG%', 'AST%', 'TOV%', 'TRB%', 'STL%', 'BLK%']
        for stat in advanced_stats:
            if stat in df.columns:
                for window in self.ROLLING_WINDOWS:
                    col_mean = f'{stat}_rolling_{window}_mean'
                    df_features[col_mean] = df[stat].rolling(
                        window=window, min_periods=1
                    ).mean()
        
        # Add opponent features if available and requested
        if include_opponent:
            opponent_features = [
                'opponent_def_rating', 'opponent_pace', 
                'opponent_scheme_drop_rate', 'opponent_scheme_switch_rate',
                'opponent_blitz_rate', 'opponent_rim_deterrence'
            ]
            for feat in opponent_features:
                if feat in df.columns:
                    df_features[feat] = df[feat]
        
        # Add role indicators if requested
        if include_role and 'role' in df.columns:
            # One-hot encode role
            role_dummies = pd.get_dummies(df['role'], prefix='role')
            df_features = pd.concat([df_features, role_dummies], axis=1)
        
        # Add pace if available
        if 'pace' in df.columns:
            df_features['pace'] = df['pace']
        
        # Add days rest if available
        if 'Date' in df.columns:
            df_features['days_rest'] = df['Date'].diff().dt.days.fillna(2)
        
        # Add home/away indicator if available
        if 'home_away' in df.columns:
            df_features['is_home'] = (df['home_away'] == 'home').astype(int)
        
        # Add temporal features if Date is available
        if 'Date' in df.columns:
            # Month (cyclical encoding)
            df_features['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
            
            # Day of week (cyclical encoding)
            df_features['dow_sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7)
            df_features['dow_cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7)
        
        # Fill any remaining NaN values with 0
        df_features = df_features.fillna(0)
        
        # Store feature names
        self.feature_names = list(df_features.columns)
        
        return df_features
    
    def train_ridge(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        alpha: Optional[float] = None,
        fit_scaler: bool = True
    ) -> Ridge:
        """
        Train Ridge regression model.
        
        Ridge regression with L2 regularization provides a simple baseline
        that is fast to train and robust to multicollinearity.
        
        Args:
            X: Feature matrix
            y: Target values
            alpha: Regularization strength (uses config default if None)
            fit_scaler: If True, fit and apply StandardScaler
            
        Returns:
            Trained Ridge model
        """
        if alpha is None:
            alpha = self.config.ridge_alpha
        
        # Scale features if requested
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Train Ridge model
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_scaled, y)
        
        return model
    
    def train_xgboost(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Train XGBoost model.
        
        XGBoost is a powerful gradient boosting model that often achieves
        state-of-the-art performance on tabular data.
        
        Args:
            X: Feature matrix
            y: Target values
            params: Optional custom parameters (uses config defaults if None)
            
        Returns:
            Trained XGBoost model
            
        Raises:
            ImportError: If xgboost is not installed
        """
        if not self.xgboost_available:
            raise ImportError(
                "xgboost is not installed. Install with: pip install xgboost"
            )
        
        # Set default parameters
        if params is None:
            params = {
                'max_depth': self.config.xgb_max_depth,
                'n_estimators': self.config.xgb_n_estimators,
                'learning_rate': self.config.xgb_learning_rate,
                'subsample': self.config.xgb_subsample,
                'colsample_bytree': self.config.xgb_colsample_bytree,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Train XGBoost model
        model = self.xgb.XGBRegressor(**params)
        model.fit(X, y)
        
        return model
    
    def train_mlp(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        layers: Optional[List[int]] = None,
        dropout: Optional[float] = None,
        fit_scaler: bool = True
    ) -> MLPRegressor:
        """
        Train Multi-Layer Perceptron (MLP) model.
        
        MLP is a neural network that can capture non-linear relationships.
        Uses ReLU activation and early stopping for regularization.
        
        Args:
            X: Feature matrix
            y: Target values
            layers: Hidden layer sizes (uses config default if None)
            dropout: Dropout rate (uses config default if None)
            fit_scaler: If True, fit and apply StandardScaler
            
        Returns:
            Trained MLP model
        """
        if layers is None:
            layers = self.config.mlp_hidden_layers
        
        if dropout is None:
            dropout = self.config.mlp_dropout
        
        # Scale features if requested
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Train MLP model
        # Note: sklearn's MLPRegressor doesn't have built-in dropout,
        # but we use early stopping and alpha for regularization
        model = MLPRegressor(
            hidden_layer_sizes=tuple(layers),
            activation='relu',
            alpha=dropout,  # L2 regularization (not true dropout)
            learning_rate_init=self.config.mlp_learning_rate_init,
            max_iter=self.config.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        return model
    
    def predict(
        self,
        model: Any,
        X: pd.DataFrame,
        model_type: str = 'ridge'
    ) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model (Ridge, XGBoost, or MLP)
            X: Feature matrix
            model_type: Type of model ('ridge', 'xgboost', 'mlp')
            
        Returns:
            Array of predictions
        """
        # Scale features for Ridge and MLP
        if model_type in ['ridge', 'mlp']:
            X_scaled = self.scaler.transform(X)
            predictions = model.predict(X_scaled)
        else:
            # XGBoost doesn't need scaling
            predictions = model.predict(X)
        
        return predictions
    
    def train_all_models(
        self,
        X: pd.DataFrame,
        y_dict: Dict[str, Union[pd.Series, np.ndarray]],
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all baseline models for multiple target statistics.
        
        Args:
            X: Feature matrix
            y_dict: Dictionary mapping stat names to target arrays
            models_to_train: List of model types to train
                           (default: ['ridge', 'xgboost', 'mlp'])
            
        Returns:
            Dictionary with structure: {stat: {model_type: trained_model}}
        """
        if models_to_train is None:
            models_to_train = ['ridge', 'mlp']
            if self.xgboost_available:
                models_to_train.append('xgboost')
        
        results = {}
        
        for stat, y in y_dict.items():
            results[stat] = {}
            
            # Train Ridge
            if 'ridge' in models_to_train:
                print(f"Training Ridge for {stat}...")
                results[stat]['ridge'] = self.train_ridge(X, y)
            
            # Train XGBoost
            if 'xgboost' in models_to_train and self.xgboost_available:
                print(f"Training XGBoost for {stat}...")
                results[stat]['xgboost'] = self.train_xgboost(X, y)
            
            # Train MLP
            if 'mlp' in models_to_train:
                print(f"Training MLP for {stat}...")
                results[stat]['mlp'] = self.train_mlp(X, y, fit_scaler=False)
        
        return results
    
    def save_model(
        self,
        model: Any,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save
            path: File path for saving
            metadata: Optional metadata to save with model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Package model with metadata
        model_package = {
            'model': model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metadata': metadata or {}
        }
        
        # Save using joblib
        joblib.dump(model_package, path)
    
    def load_model(
        self,
        path: Union[str, Path]
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a trained model from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model package
        model_package = joblib.load(path)
        
        # Restore scaler and feature names
        self.scaler = model_package.get('scaler', StandardScaler())
        self.feature_names = model_package.get('feature_names')
        
        model = model_package['model']
        metadata = model_package.get('metadata', {})
        
        return model, metadata
    
    def save_all_models(
        self,
        models_dict: Dict[str, Dict[str, Any]],
        output_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save all trained models to a directory.
        
        Args:
            models_dict: Dictionary with structure {stat: {model_type: model}}
            output_dir: Directory to save models
            metadata: Optional metadata to save with models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for stat, models in models_dict.items():
            for model_type, model in models.items():
                filename = f"{stat}_{model_type}.joblib"
                filepath = output_dir / filename
                
                # Add stat and model type to metadata
                model_metadata = metadata.copy() if metadata else {}
                model_metadata.update({
                    'stat': stat,
                    'model_type': model_type
                })
                
                self.save_model(model, filepath, model_metadata)
        
        print(f"Saved {len(models_dict)} stats × {len(next(iter(models_dict.values())))} models to {output_dir}")
    
    def load_all_models(
        self,
        input_dir: Union[str, Path],
        stats: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load all models from a directory.
        
        Args:
            input_dir: Directory containing saved models
            stats: List of stats to load (loads all if None)
            model_types: List of model types to load (loads all if None)
            
        Returns:
            Dictionary with structure {stat: {model_type: model}}
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {input_dir}")
        
        if stats is None:
            stats = self.TARGET_STATS
        
        if model_types is None:
            model_types = ['ridge', 'xgboost', 'mlp']
        
        results = {}
        
        for stat in stats:
            results[stat] = {}
            for model_type in model_types:
                filename = f"{stat}_{model_type}.joblib"
                filepath = input_dir / filename
                
                if filepath.exists():
                    model, metadata = self.load_model(filepath)
                    results[stat][model_type] = model
                    print(f"Loaded {stat} {model_type} model")
                else:
                    print(f"Warning: Model file not found: {filepath}")
        
        return results
