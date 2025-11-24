"""
Unit tests for data loading and validation functionality.

Tests cover:
- Loading valid CSV files
- Handling missing files
- Data validation (missingness checks)
- Outlier capping
- Temporal leakage control
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader, ValidationResult


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance for testing."""
        return DataLoader(data_dir="Data")
    
    def test_init_valid_directory(self):
        """Test DataLoader initialization with valid directory."""
        loader = DataLoader(data_dir="Data")
        assert loader.data_dir.exists()
    
    def test_init_invalid_directory(self):
        """Test DataLoader initialization with invalid directory."""
        with pytest.raises(FileNotFoundError):
            DataLoader(data_dir="NonExistentDirectory")
    
    def test_load_player_data_valid(self, data_loader):
        """Test loading valid player data."""
        # Use Stephen Curry 2024 data as test case
        df = data_loader.load_player_data("Stephen_Curry", 2024)
        
        assert not df.empty
        assert 'Date' in df.columns
        assert 'PTS' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['Date'])
        
        # Check data is sorted by date
        assert df['Date'].is_monotonic_increasing
    
    def test_load_player_data_missing_file(self, data_loader):
        """Test loading data for non-existent player."""
        with pytest.raises(FileNotFoundError):
            data_loader.load_player_data("NonExistent_Player", 2024)
    
    def test_load_player_data_missing_year(self, data_loader):
        """Test loading data for non-existent year."""
        with pytest.raises(FileNotFoundError):
            data_loader.load_player_data("Stephen_Curry", 1999)
    
    def test_validate_data_valid(self, data_loader):
        """Test validation on valid data."""
        df = data_loader.load_player_data("Stephen_Curry", 2024)
        result = data_loader.validate_data(df)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_data_empty_dataframe(self, data_loader):
        """Test validation on empty DataFrame."""
        df = pd.DataFrame()
        result = data_loader.validate_data(df)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "empty" in result.errors[0].lower()
    
    def test_validate_data_missing_columns(self, data_loader):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            'Date': ['2024-01-01'],
            'PTS': [25]
        })
        result = data_loader.validate_data(df)
        
        assert not result.is_valid
        assert any('Missing required columns' in err for err in result.errors)
    
    def test_validate_data_high_missingness(self, data_loader):
        """Test validation catches high missingness."""
        # Create DataFrame with >5% missing values
        df = pd.DataFrame({
            'Player': ['Test'] * 100,
            'Date': pd.date_range('2024-01-01', periods=100),
            'MP': [30] * 100,
            'PTS': [np.nan] * 10 + [25] * 90,  # 10% missing
            'TRB': [5] * 100,
            'AST': [3] * 100,
            'STL': [1] * 100,
            'BLK': [0] * 100,
            'FG%': [0.5] * 100,
            '3P%': [0.4] * 100,
            'TS%': [0.6] * 100,
            'USG%': [25] * 100,
            'TOV': [2] * 100,
            'FT%': [0.9] * 100,
            'ORTG': [115] * 100,
            'DRTG': [110] * 100
        })
        
        result = data_loader.validate_data(df, max_missingness=0.05)
        
        assert not result.is_valid
        assert any('PTS' in err for err in result.errors)
    
    def test_apply_outlier_caps(self, data_loader):
        """Test outlier capping functionality."""
        # Create DataFrame with outliers
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'PTS': [25] * 95 + [100, 0, 150, -10, 200],  # Outliers at end
            'MP': [30] * 100
        })
        
        df_capped = data_loader.apply_outlier_caps(df)
        
        # Check that extreme values are capped
        assert df_capped['PTS'].max() < 100
        assert df_capped['PTS'].min() >= 0
    
    def test_enforce_leakage_control_strict(self, data_loader):
        """Test strict temporal leakage control."""
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'PTS': range(10)
        })
        
        forecast_date = datetime(2024, 1, 6)
        df_filtered = data_loader.enforce_leakage_control(df, forecast_date, strict=True)
        
        # Should only include dates before forecast_date
        assert len(df_filtered) == 5
        assert all(df_filtered['Date'] < forecast_date)
    
    def test_enforce_leakage_control_non_strict(self, data_loader):
        """Test non-strict temporal leakage control."""
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'PTS': range(10)
        })
        
        forecast_date = datetime(2024, 1, 6)
        df_filtered = data_loader.enforce_leakage_control(df, forecast_date, strict=False)
        
        # Should include dates up to and including forecast_date
        assert len(df_filtered) == 6
        assert all(df_filtered['Date'] <= forecast_date)
    
    def test_enforce_leakage_control_missing_date(self, data_loader):
        """Test leakage control with missing Date column."""
        df = pd.DataFrame({'PTS': [25, 30, 35]})
        
        with pytest.raises(ValueError, match="Date column"):
            data_loader.enforce_leakage_control(df, datetime(2024, 1, 1))
    
    def test_load_multiple_players(self, data_loader):
        """Test loading data for multiple players."""
        players = ["Stephen_Curry", "LeBron_James"]
        results = data_loader.load_multiple_players(players, 2024)
        
        assert len(results) > 0
        for player_name, df in results.items():
            assert not df.empty
            assert 'PTS' in df.columns


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
