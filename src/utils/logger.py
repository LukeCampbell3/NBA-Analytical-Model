"""
JSON structured logging module for the NBA prediction system.

This module provides structured logging with JSON formatting for better
log aggregation, searching, and analysis. All major operations are logged
with relevant context and metadata.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import traceback


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.
    
    Each log entry includes:
    - timestamp: ISO format timestamp
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - context: Additional context data
    - exception: Exception details if present
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
        
        Returns:
            JSON-formatted log string
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add context data if present
        if hasattr(record, 'context') and record.context:
            log_entry["context"] = record.context
        
        # Add exception information if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info', 'context']:
                try:
                    # Only add JSON-serializable values
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)
        
        return json.dumps(log_entry)


class StructuredLogger:
    """
    Structured logger with JSON formatting and context management.
    
    This class provides a convenient interface for logging with structured
    context data. It wraps the standard Python logging module with additional
    functionality for JSON formatting and context tracking.
    """
    
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: str = "INFO",
        console_output: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            log_file: Path to log file (optional)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Whether to output to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Add file handler if log_file specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _log(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs
    ):
        """
        Internal logging method with context support.
        
        Args:
            level: Log level
            message: Log message
            context: Additional context data
            exc_info: Whether to include exception info
            **kwargs: Additional fields to include in log
        """
        extra = {'context': context or {}}
        extra.update(kwargs)
        
        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra, exc_info=exc_info)
    
    def debug(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log debug message."""
        self._log('DEBUG', message, context, **kwargs)
    
    def info(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log info message."""
        self._log('INFO', message, context, **kwargs)
    
    def warning(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log warning message."""
        self._log('WARNING', message, context, **kwargs)
    
    def error(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
        **kwargs
    ):
        """Log error message with exception info."""
        self._log('ERROR', message, context, exc_info=exc_info, **kwargs)
    
    def critical(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
        **kwargs
    ):
        """Log critical message with exception info."""
        self._log('CRITICAL', message, context, exc_info=exc_info, **kwargs)
    
    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        level: str = "INFO"
    ):
        """
        Log a structured event.
        
        Args:
            event_type: Type of event (e.g., 'data_loaded', 'model_trained')
            details: Event details
            level: Log level
        """
        message = f"Event: {event_type}"
        context = {
            "event_type": event_type,
            **details
        }
        self._log(level, message, context)
    
    def log_operation_start(
        self,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log the start of an operation.
        
        Args:
            operation: Operation name
            details: Operation details
        """
        self.info(
            f"Starting operation: {operation}",
            context={
                "operation": operation,
                "status": "started",
                **(details or {})
            }
        )
    
    def log_operation_complete(
        self,
        operation: str,
        duration_sec: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log the completion of an operation.
        
        Args:
            operation: Operation name
            duration_sec: Operation duration in seconds
            details: Operation details
        """
        context = {
            "operation": operation,
            "status": "completed",
            **(details or {})
        }
        
        if duration_sec is not None:
            context["duration_sec"] = duration_sec
        
        self.info(
            f"Completed operation: {operation}",
            context=context
        )
    
    def log_operation_failed(
        self,
        operation: str,
        error: Exception,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log a failed operation.
        
        Args:
            operation: Operation name
            error: Exception that caused failure
            details: Operation details
        """
        self.error(
            f"Operation failed: {operation}",
            context={
                "operation": operation,
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                **(details or {})
            }
        )


def get_logger(
    name: str,
    log_file: Optional[str] = "logs/system.log",
    level: str = "INFO",
    console_output: bool = True
) -> StructuredLogger:
    """
    Get or create a structured logger.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Path to log file
        level: Logging level
        console_output: Whether to output to console
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(
        name=name,
        log_file=log_file,
        level=level,
        console_output=console_output
    )


# Convenience function for quick logging setup
def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/system.log",
    console_output: bool = True
):
    """
    Setup logging configuration for the entire application.
    
    Args:
        level: Logging level
        log_file: Path to log file
        console_output: Whether to output to console
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(console_handler)
