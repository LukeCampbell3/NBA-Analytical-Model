"""
Custom exception classes for the NBA prediction system.

This module defines custom exceptions for different error scenarios
to provide clear error messages and enable proper error handling.
"""


class NBASystemError(Exception):
    """Base exception for all NBA prediction system errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize exception with message and optional details.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        """String representation of the error."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataError(NBASystemError):
    """Base exception for data-related errors."""
    pass


class DataQualityError(DataError):
    """Raised when data quality checks fail."""
    
    def __init__(self, message: str, validation_errors: list = None, missingness_report: dict = None):
        """
        Initialize data quality error.
        
        Args:
            message: Error message
            validation_errors: List of validation errors
            missingness_report: Dictionary of missingness by column
        """
        details = {}
        if validation_errors:
            details['validation_errors'] = validation_errors
        if missingness_report:
            details['missingness_report'] = missingness_report
        super().__init__(message, details)
        self.validation_errors = validation_errors or []
        self.missingness_report = missingness_report or {}


class DataLeakageError(DataError):
    """Raised when temporal ordering is violated."""
    
    def __init__(self, message: str, forecast_date: str = None, invalid_dates: int = None):
        """
        Initialize data leakage error.
        
        Args:
            message: Error message
            forecast_date: The forecast date that was violated
            invalid_dates: Number of invalid dates found
        """
        details = {}
        if forecast_date:
            details['forecast_date'] = forecast_date
        if invalid_dates is not None:
            details['invalid_dates'] = invalid_dates
        super().__init__(message, details)


class DataNotFoundError(DataError):
    """Raised when required data files are not found."""
    
    def __init__(self, message: str, file_path: str = None, player_name: str = None):
        """
        Initialize data not found error.
        
        Args:
            message: Error message
            file_path: Path to missing file
            player_name: Player name if applicable
        """
        details = {}
        if file_path:
            details['file_path'] = file_path
        if player_name:
            details['player_name'] = player_name
        super().__init__(message, details)


class ModelError(NBASystemError):
    """Base exception for model-related errors."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    
    def __init__(self, message: str, model_type: str = None, error_details: str = None):
        """
        Initialize model training error.
        
        Args:
            message: Error message
            model_type: Type of model that failed
            error_details: Detailed error information
        """
        details = {}
        if model_type:
            details['model_type'] = model_type
        if error_details:
            details['error_details'] = error_details
        super().__init__(message, details)


class ModelNotFoundError(ModelError):
    """Raised when a required model is not found."""
    
    def __init__(self, message: str, model_path: str = None, model_name: str = None):
        """
        Initialize model not found error.
        
        Args:
            message: Error message
            model_path: Path to missing model
            model_name: Name of missing model
        """
        details = {}
        if model_path:
            details['model_path'] = model_path
        if model_name:
            details['model_name'] = model_name
        super().__init__(message, details)


class RegionError(NBASystemError):
    """Base exception for capability region errors."""
    pass


class RegionConstructionError(RegionError):
    """Raised when capability region construction fails."""
    
    def __init__(self, message: str, reason: str = None, dimension: int = None):
        """
        Initialize region construction error.
        
        Args:
            message: Error message
            reason: Reason for failure
            dimension: Dimension of the region
        """
        details = {}
        if reason:
            details['reason'] = reason
        if dimension is not None:
            details['dimension'] = dimension
        super().__init__(message, details)


class SingularMatrixError(RegionError):
    """Raised when a covariance matrix is singular or not positive definite."""
    
    def __init__(self, message: str, matrix_shape: tuple = None, condition_number: float = None):
        """
        Initialize singular matrix error.
        
        Args:
            message: Error message
            matrix_shape: Shape of the singular matrix
            condition_number: Condition number if available
        """
        details = {}
        if matrix_shape:
            details['matrix_shape'] = matrix_shape
        if condition_number is not None:
            details['condition_number'] = condition_number
        super().__init__(message, details)


class EmptyRegionError(RegionError):
    """Raised when a capability region is empty or has no valid samples."""
    
    def __init__(self, message: str, attempts: int = None, constraints_count: int = None):
        """
        Initialize empty region error.
        
        Args:
            message: Error message
            attempts: Number of sampling attempts made
            constraints_count: Number of constraints applied
        """
        details = {}
        if attempts is not None:
            details['attempts'] = attempts
        if constraints_count is not None:
            details['constraints_count'] = constraints_count
        super().__init__(message, details)


class SimulationError(NBASystemError):
    """Raised when simulation fails."""
    
    def __init__(self, message: str, player_id: str = None, game_id: str = None, error_stage: str = None):
        """
        Initialize simulation error.
        
        Args:
            message: Error message
            player_id: Player ID if applicable
            game_id: Game ID if applicable
            error_stage: Stage where error occurred
        """
        details = {}
        if player_id:
            details['player_id'] = player_id
        if game_id:
            details['game_id'] = game_id
        if error_stage:
            details['error_stage'] = error_stage
        super().__init__(message, details)


class CalibrationError(NBASystemError):
    """Raised when calibration fails."""
    
    def __init__(self, message: str, statistic: str = None, method: str = None):
        """
        Initialize calibration error.
        
        Args:
            message: Error message
            statistic: Statistic being calibrated
            method: Calibration method used
        """
        details = {}
        if statistic:
            details['statistic'] = statistic
        if method:
            details['method'] = method
        super().__init__(message, details)


class ConfigurationError(NBASystemError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: str = None, config_file: str = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that is invalid
            config_file: Configuration file path
        """
        details = {}
        if config_key:
            details['config_key'] = config_key
        if config_file:
            details['config_file'] = config_file
        super().__init__(message, details)


class ValidationError(NBASystemError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_name: str = None, invalid_value: any = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field_name: Name of invalid field
            invalid_value: The invalid value
        """
        details = {}
        if field_name:
            details['field_name'] = field_name
        if invalid_value is not None:
            details['invalid_value'] = str(invalid_value)
        super().__init__(message, details)
