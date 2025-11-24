"""
Role inference for new players.

Infers player role (starter, rotation, bench) from available features.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RoleInferenceModel:
    """
    Infers player role from available features.
    
    Uses simple heuristics or trained logistic regression.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize role inference model.
        
        Args:
            model_path: Path to trained model (optional)
        """
        self.model = None
        if model_path and Path(model_path).exists():
            self.model = joblib.load(model_path)
            logger.info(f"Loaded role inference model from {model_path}")
    
    def infer_role(self, player_info: Dict) -> str:
        """
        Infer player role from available information.
        
        Args:
            player_info: Dictionary with player features
                - height: Height in inches (optional)
                - weight: Weight in pounds (optional)
                - usage: Usage rate (optional)
                - three_pa_rate: 3PA rate (optional)
                - rim_attempt_rate: Rim attempt rate (optional)
                - ast_pct: Assist percentage (optional)
                - blk_pct: Block percentage (optional)
                
        Returns:
            Inferred role: 'starter', 'rotation', 'bench', or 'unknown'
        """
        # If model is trained, use it
        if self.model is not None:
            return self._infer_with_model(player_info)
        
        # Otherwise use heuristics
        return self._infer_with_heuristics(player_info)
    
    def _infer_with_heuristics(self, player_info: Dict) -> str:
        """Infer role using simple heuristics."""
        
        # Check usage rate (most reliable indicator)
        usage = player_info.get('usage')
        if usage is not None:
            if usage >= 0.25:
                return 'starter'
            elif usage >= 0.18:
                return 'rotation'
            elif usage >= 0.12:
                return 'bench'
        
        # Check minutes if available
        minutes = player_info.get('minutes')
        if minutes is not None:
            if minutes >= 28:
                return 'starter'
            elif minutes >= 18:
                return 'rotation'
            elif minutes >= 8:
                return 'bench'
        
        # Check position-specific indicators
        three_pa_rate = player_info.get('three_pa_rate', 0)
        rim_attempt_rate = player_info.get('rim_attempt_rate', 0)
        ast_pct = player_info.get('ast_pct', 0)
        blk_pct = player_info.get('blk_pct', 0)
        
        # Guards tend to have high 3PA and AST
        if three_pa_rate > 0.4 and ast_pct > 0.2:
            return 'rotation'  # Conservative estimate for guards
        
        # Bigs tend to have high rim attempts and blocks
        if rim_attempt_rate > 0.4 and blk_pct > 0.03:
            return 'rotation'  # Conservative estimate for bigs
        
        # Default to unknown if not enough info
        logger.warning("Insufficient info for role inference, returning 'unknown'")
        return 'unknown'
    
    def _infer_with_model(self, player_info: Dict) -> str:
        """Infer role using trained model."""
        # Extract features
        features = self._extract_features(player_info)
        
        if features is None:
            logger.warning("Could not extract features, falling back to heuristics")
            return self._infer_with_heuristics(player_info)
        
        # Predict
        prediction = self.model.predict([features])[0]
        role_map = {0: 'bench', 1: 'rotation', 2: 'starter'}
        return role_map.get(prediction, 'unknown')
    
    def _extract_features(self, player_info: Dict) -> Optional[np.ndarray]:
        """Extract feature vector from player info."""
        required_features = ['usage', 'three_pa_rate', 'rim_attempt_rate', 
                           'ast_pct', 'blk_pct']
        
        # Check if we have enough features
        if not all(f in player_info for f in required_features):
            return None
        
        return np.array([player_info[f] for f in required_features])
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             save_path: Optional[str] = None):
        """
        Train role inference model.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Role labels (0=bench, 1=rotation, 2=starter)
            save_path: Path to save trained model
        """
        self.model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
        
        self.model.fit(X, y)
        logger.info(f"Trained role inference model (accuracy: {self.model.score(X, y):.3f})")
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, save_path)
            logger.info(f"Saved model to {save_path}")
