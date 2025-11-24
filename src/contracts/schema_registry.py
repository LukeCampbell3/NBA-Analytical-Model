"""
Schema registry for tracking schema evolution and migrations.

Maintains history of schemas, detects changes, and applies migrations.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Type
from datetime import datetime
import pandas as pd
from pydantic import BaseModel

from src.contracts.data_models import SCHEMA_ALIASES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SchemaRegistry:
    """
    Manages schema versions, migrations, and history.
    
    Responsibilities:
    - Track schema changes over time
    - Apply column aliases for backward compatibility
    - Detect missing/extra columns
    - Log schema evolution
    """
    
    def __init__(self, history_file: str = "logs/ingest_schema_history.json"):
        self.history_file = Path(history_file)
        self.aliases = SCHEMA_ALIASES
        self.history = self._load_history()
        
    def _load_history(self) -> Dict:
        """Load schema history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load schema history: {e}")
                return {}
        return {}
    
    def _save_history(self):
        """Save schema history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save schema history: {e}")
    
    def hash_schema(self, columns: List[str]) -> str:
        """
        Compute hash of schema (sorted column names).
        
        Args:
            columns: List of column names
            
        Returns:
            SHA256 hash of sorted columns
        """
        sorted_cols = sorted(columns)
        schema_str = ','.join(sorted_cols)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
    
    def apply_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column aliases for backward compatibility.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with aliased columns renamed
        """
        rename_map = {}
        for old_name, new_name in self.aliases.items():
            if old_name in df.columns and new_name not in df.columns:
                rename_map[old_name] = new_name
                logger.info(f"Applying alias: {old_name} -> {new_name}")
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        return df
    
    def check_missing_columns(self, df: pd.DataFrame, 
                             contract: Type[BaseModel]) -> List[str]:
        """
        Check for missing required columns.
        
        Args:
            df: Input DataFrame
            contract: Pydantic contract model
            
        Returns:
            List of missing required column names
        """
        required_fields = []
        for field_name, field in contract.__fields__.items():
            if field.required:
                required_fields.append(field_name)
        
        missing = [f for f in required_fields if f not in df.columns]
        return missing
    
    def check_extra_columns(self, df: pd.DataFrame,
                           contract: Type[BaseModel]) -> List[str]:
        """
        Check for extra columns not in contract.
        
        Args:
            df: Input DataFrame
            contract: Pydantic contract model
            
        Returns:
            List of extra column names
        """
        contract_fields = set(contract.__fields__.keys())
        df_columns = set(df.columns)
        extra = list(df_columns - contract_fields)
        return extra
    
    def record_schema(self, schema_hash: str, columns: List[str],
                     source: str = "unknown"):
        """
        Record schema in history.
        
        Args:
            schema_hash: Hash of schema
            columns: List of column names
            source: Data source identifier
        """
        if schema_hash not in self.history:
            self.history[schema_hash] = {
                'first_seen': datetime.now().isoformat(),
                'columns': sorted(columns),
                'source': source,
                'count': 1
            }
            logger.info(f"New schema detected: {schema_hash}")
        else:
            self.history[schema_hash]['count'] += 1
            self.history[schema_hash]['last_seen'] = datetime.now().isoformat()
        
        self._save_history()
    
    def validate_and_migrate(self, df: pd.DataFrame,
                            contract: Type[BaseModel],
                            source: str = "unknown") -> pd.DataFrame:
        """
        Validate schema and apply migrations.
        
        This is the main entry point for schema validation.
        
        Args:
            df: Input DataFrame
            contract: Pydantic contract model
            source: Data source identifier
            
        Returns:
            Migrated DataFrame
        """
        logger.info(f"Validating schema for {source}")
        
        # 1. Compute schema hash
        schema_hash = self.hash_schema(df.columns.tolist())
        
        # 2. Apply aliases
        df = self.apply_aliases(df)
        
        # 3. Check for missing columns
        missing = self.check_missing_columns(df, contract)
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            # Will be handled by contract validation with defaults
        
        # 4. Check for extra columns
        extra = self.check_extra_columns(df, contract)
        if extra:
            logger.info(f"Extra columns (will be preserved): {extra}")
        
        # 5. Record schema
        self.record_schema(schema_hash, df.columns.tolist(), source)
        
        return df
    
    def get_schema_stats(self) -> Dict:
        """
        Get statistics about schema evolution.
        
        Returns:
            Dictionary with schema statistics
        """
        return {
            'total_schemas': len(self.history),
            'schemas': [
                {
                    'hash': hash_val,
                    'first_seen': info['first_seen'],
                    'last_seen': info.get('last_seen', info['first_seen']),
                    'count': info['count'],
                    'n_columns': len(info['columns'])
                }
                for hash_val, info in self.history.items()
            ]
        }
