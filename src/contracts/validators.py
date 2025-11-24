"""
Contract validators for DataFrame validation against Pydantic models.
"""

from typing import Type, List, Dict, Any
import pandas as pd
from pydantic import BaseModel, ValidationError

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContractValidator:
    """
    Validates DataFrames against Pydantic contracts.
    
    Handles:
    - Row-by-row validation
    - Batch validation with error collection
    - Graceful degradation with warnings
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize validator.
        
        Args:
            strict: If True, raise on validation errors. If False, log warnings.
        """
        self.strict = strict
        self.errors = []
    
    def validate_row(self, row: Dict[str, Any], 
                    contract: Type[BaseModel]) -> BaseModel:
        """
        Validate a single row against contract.
        
        Args:
            row: Dictionary of row data
            contract: Pydantic contract model
            
        Returns:
            Validated contract instance
            
        Raises:
            ValidationError: If strict=True and validation fails
        """
        try:
            return contract(**row)
        except ValidationError as e:
            if self.strict:
                raise
            else:
                logger.warning(f"Validation error: {e}")
                self.errors.append({
                    'row': row,
                    'errors': e.errors()
                })
                # Return with defaults
                return contract(**{k: v for k, v in row.items() 
                                 if k in contract.__fields__})
    
    def validate_dataframe(self, df: pd.DataFrame,
                          contract: Type[BaseModel]) -> List[BaseModel]:
        """
        Validate entire DataFrame against contract.
        
        Args:
            df: Input DataFrame
            contract: Pydantic contract model
            
        Returns:
            List of validated contract instances
        """
        self.errors = []
        validated = []
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            try:
                validated_row = self.validate_row(row_dict, contract)
                validated.append(validated_row)
            except ValidationError as e:
                if self.strict:
                    logger.error(f"Validation failed at row {idx}: {e}")
                    raise
                else:
                    logger.warning(f"Validation warning at row {idx}: {e}")
        
        if self.errors:
            logger.warning(f"Total validation errors: {len(self.errors)}")
        
        return validated
    
    def to_dataframe(self, validated: List[BaseModel]) -> pd.DataFrame:
        """
        Convert validated contracts back to DataFrame.
        
        Args:
            validated: List of validated contract instances
            
        Returns:
            DataFrame with validated data
        """
        return pd.DataFrame([v.dict() for v in validated])
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation errors.
        
        Returns:
            Dictionary with error statistics
        """
        if not self.errors:
            return {'total_errors': 0}
        
        error_types = {}
        for error in self.errors:
            for err in error['errors']:
                err_type = err['type']
                error_types[err_type] = error_types.get(err_type, 0) + 1
        
        return {
            'total_errors': len(self.errors),
            'error_types': error_types,
            'sample_errors': self.errors[:5]  # First 5 errors
        }
