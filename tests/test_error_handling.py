"""
Tests for error handling and logging functionality.

This module tests custom exceptions and structured logging to ensure
proper error handling throughout the system.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.errors import (
    DataQualityError,
    DataLeakageError,
    DataNotFoundError,
    SingularMatrixError,
    EmptyRegionError,
    RegionConstructionError,
    ValidationError
)
from src.utils.logger import get_logger, JSONFormatter
from src.utils.data_loader import DataLoader
from src.regions.build import RegionBuilder


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_data_quality_error(self):
        """Test DataQualityError with validation details."""
        errors = ["Missing column: PTS", "High missingness in AST"]
        missingness = {"PTS": 0.0, "AST": 0.08}
        
        exc = DataQualityError(
            "Data validation failed",
            validation_errors=errors,
            missingness_report=missingness
        )
        
        assert exc.message == "Data validation failed"
        assert exc.validation_errors == errors
        assert exc.missingness_report == missingness
        assert "validation_errors" in exc.details
    
    def test_data_leakage_error(self):
        """Test DataLeakageError with date information."""
        exc = DataLeakageError(
            "Temporal ordering violated",
            forecast_date="2024-01-15",
            invalid_dates=5
        )
        
        assert exc.message == "Temporal ordering violated"
        assert exc.details["forecast_date"] == "2024-01-15"
        assert exc.details["invalid_dates"] == 5
    
    def test_singular_matrix_error(self):
        """Test SingularMatrixError with matrix details."""
        exc = SingularMatrixError(
            "Matrix is singular",
            matrix_shape=(6, 6),
            condition_number=1e15
        )
        
        assert exc.message == "Matrix is singular"
        assert exc.details["matrix_shape"] == (6, 6)
        assert exc.details["condition_number"] == 1e15
    
    def test_empty_region_error(self):
        """Test EmptyRegionError with sampling details."""
        exc = EmptyRegionError(
            "No valid samples found",
            attempts=1000,
            constraints_count=15
        )
        
        assert exc.message == "No valid samples found"
        assert exc.details["attempts"] == 1000
        assert exc.details["constraints_count"] == 15


class TestStructuredLogging:
    """Test structured logging functionality."""
    
    def test_json_formatter(self):
        """Test JSON log formatting."""
        import logging
        
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.context = {"key": "value"}
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry["level"] == "INFO"
        assert log_entry["message"] == "Test message"
        assert log_entry["context"]["key"] == "value"
        assert "timestamp" in log_entry
    
    def test_logger_with_context(self):
        """Test logger with context data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = get_logger("test", log_file=str(log_file), console_output=False)
            
            logger.info(
                "Test operation",
                context={"operation": "test", "status": "success"}
            )
            
            # Close handlers to release file
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
            
            # Read log file
            with open(log_file, 'r') as f:
                log_line = f.readline()
                log_entry = json.loads(log_line)
            
            assert log_entry["message"] == "Test operation"
            assert log_entry["context"]["operation"] == "test"
            assert log_entry["context"]["status"] == "success"
    
    def test_log_event(self):
        """Test structured event logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = get_logger("test", log_file=str(log_file), console_output=False)
            
            logger.log_event(
                "data_loaded",
                details={"rows": 100, "columns": 20}
            )
            
            # Close handlers to release file
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
            
            # Read log file
            with open(log_file, 'r') as f:
                log_line = f.readline()
                log_entry = json.loads(log_line)
            
            assert "Event: data_loaded" in log_entry["message"]
            assert log_entry["context"]["event_type"] == "data_loaded"
            assert log_entry["context"]["rows"] == 100


class TestDataLoaderErrorHandling:
    """Test error handling in DataLoader."""
    
    def test_missing_data_directory(self):
        """Test error when data directory doesn't exist."""
        with pytest.raises(DataNotFoundError) as exc_info:
            DataLoader(data_dir="nonexistent_directory")
        
        assert "Data directory not found" in str(exc_info.value)
        assert "nonexistent_directory" in exc_info.value.details.get("file_path", "")
    
    def test_missing_player_file(self):
        """Test error when player file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=tmpdir)
            
            with pytest.raises(DataNotFoundError) as exc_info:
                loader.load_player_data("NonexistentPlayer", 2024)
            
            assert "not found" in str(exc_info.value).lower()
    
    def test_data_validation_failure(self):
        """Test data validation with high missingness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data with high missingness
            player_dir = Path(tmpdir) / "TestPlayer"
            player_dir.mkdir()
            
            df = pd.DataFrame({
                'Player': ['Test'] * 10,
                'Date': pd.date_range('2024-01-01', periods=10),
                'MP': [30.0] * 10,
                'PTS': [20.0] * 3 + [np.nan] * 7,  # 70% missing
                'TRB': [5.0] * 10,
                'AST': [3.0] * 10,
                'STL': [1.0] * 10,
                'BLK': [0.5] * 10,
                'FG%': [0.45] * 10,
                '3P%': [0.35] * 10,
                'TS%': [0.55] * 10,
                'USG%': [0.25] * 10,
                'TOV': [2.0] * 10,
                'FT%': [0.80] * 10,
                'ORTG': [110.0] * 10,
                'DRTG': [105.0] * 10
            })
            
            df.to_csv(player_dir / "2024.csv", index=False)
            
            loader = DataLoader(data_dir=tmpdir)
            loaded_df = loader.load_player_data("TestPlayer", 2024)
            
            # Validation should raise error due to high missingness
            with pytest.raises(DataQualityError) as exc_info:
                loader.validate_data(loaded_df, max_missingness=0.05)
            
            assert "validation failed" in str(exc_info.value).lower()
            assert exc_info.value.validation_errors
    
    def test_leakage_control_missing_date(self):
        """Test leakage control with missing Date column."""
        df = pd.DataFrame({
            'PTS': [20, 25, 30],
            'TRB': [5, 6, 7]
        })
        
        loader = DataLoader(data_dir="Data")
        
        with pytest.raises(DataLeakageError) as exc_info:
            loader.enforce_leakage_control(df, datetime(2024, 1, 15))
        
        assert "'Date' column" in str(exc_info.value)


class TestRegionBuilderErrorHandling:
    """Test error handling in RegionBuilder."""
    
    def test_singular_covariance_matrix(self):
        """Test error with singular covariance matrix."""
        builder = RegionBuilder(regularization=0.0)  # No regularization
        
        mu = np.array([1.0, 2.0, 3.0])
        # Create singular matrix (rank deficient)
        Sigma = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        
        with pytest.raises(SingularMatrixError) as exc_info:
            builder.credible_ellipsoid(mu, Sigma)
        
        assert "not positive definite" in str(exc_info.value).lower()
    
    def test_dimension_mismatch(self):
        """Test error with dimension mismatch."""
        builder = RegionBuilder()
        
        mu = np.array([1.0, 2.0, 3.0])
        Sigma = np.eye(4)  # Wrong dimension
        
        with pytest.raises(ValidationError) as exc_info:
            builder.credible_ellipsoid(mu, Sigma)
        
        assert "dimension" in str(exc_info.value).lower()
        assert "doesn't match" in str(exc_info.value).lower()
    
    def test_invalid_mu_shape(self):
        """Test error with invalid mu shape."""
        builder = RegionBuilder()
        
        mu = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D instead of 1D
        Sigma = np.eye(2)
        
        with pytest.raises(ValidationError) as exc_info:
            builder.credible_ellipsoid(mu, Sigma)
        
        assert "1-dimensional" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
