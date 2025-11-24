"""
Reporting module for NBA player performance predictions.

This module provides comprehensive reporting functionality including:
- Coach one-pager PDFs
- Analyst detail PDFs
- Benchmark comparison reports
- JSON and CSV exports
"""

from src.reporting.build import (
    ReportBuilder,
    CalibrationResult,
    create_benchmark_charts
)

__all__ = [
    'ReportBuilder',
    'CalibrationResult',
    'create_benchmark_charts'
]
