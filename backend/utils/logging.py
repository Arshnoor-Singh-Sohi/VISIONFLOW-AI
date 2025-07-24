"""
VisionFlow AI - Logging Configuration
====================================

This module provides centralized logging configuration for the entire
VisionFlow AI system. It sets up structured logging with proper formatting,
rotation, and different output destinations based on the environment.

Features:
- Structured logging with consistent formatting
- Automatic log rotation to prevent disk space issues
- Different log levels for different components
- Database logging for important events
- Performance monitoring integration
- Error tracking and alerting
"""

import os
import sys
import logging
import logging.handlers
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import json
import traceback

from ..config import get_settings


# =============================================================================
# CUSTOM LOG FORMATTER
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that creates structured log entries.
    
    This formatter adds context information and structures the log output
    for better readability and parsing by log analysis tools.
    """
    
    def __init__(self, include_traceback: bool = True):
        self.include_traceback = include_traceback
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with additional context."""
        
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add process and thread info for debugging
        log_data['process_id'] = os.getpid()
        log_data['thread_id'] = record.thread
        
        # Add exception info if present
        if record.exc_info and self.include_traceback:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info) if self.include_traceback else None
            }
        
        # Add custom fields if present
        custom_fields = {}
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 'process',
                          'message', 'exc_info', 'exc_text', 'stack_info'):
                custom_fields[key] = value
        
        if custom_fields:
            log_data['extra'] = custom_fields
        
        # Format as JSON for structured logging
        try:
            return json.dumps(log_data, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            # Fallback to simple string formatting
            return f"{log_data['timestamp']} | {log_data['level']} | {log_data['logger']} | {log_data['message']}"


class ColoredConsoleFormatter(logging.Formatter):
    """
    Formatter that adds colors to console output for better readability.
    
    This makes it easier to scan logs during development and debugging.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Create colored level name
        colored_level = f"{color}{record.levelname:<8}{reset}"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Format logger name (keep it short)
        logger_name = record.name.split('.')[-1] if '.' in record.name else record.name
        logger_name = f"{logger_name:<15}"[:15]
        
        # Format message
        message = record.getMessage()
        
        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return f"{timestamp} | {colored_level} | {logger_name} | {message}"


# =============================================================================
# DATABASE LOG HANDLER
# =============================================================================

class DatabaseLogHandler(logging.Handler):
    """
    Custom log handler that writes important logs to the database.
    
    This allows the system to track important events in the database
    for monitoring, alerting, and analysis purposes.
    """
    
    def __init__(self, min_level: int = logging.WARNING):
        super().__init__()
        self.min_level = min_level
        self.setLevel(min_level)
    
    def emit(self, record: logging.LogRecord):
        """Write log record to database."""
        
        try:
            # Only log warnings and above to database
            if record.levelno < self.min_level:
                return
            
            # Import here to avoid circular imports
            from ..database import db_manager
            from ..models.database_models import SystemLog
            
            # Extract relevant information
            log_entry = SystemLog(
                level=record.levelname,
                category=self._get_category(record.name),
                message=record.getMessage(),
                details=self._extract_details(record),
                hostname=os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                process_id=os.getpid()
            )
            
            # Add to database
            try:
                with db_manager.get_session_context() as db:
                    db.add(log_entry)
                    db.commit()
            except Exception as db_error:
                # Don't raise exception if database logging fails
                # This could cause infinite recursion
                print(f"Failed to log to database: {db_error}", file=sys.stderr)
                
        except Exception as e:
            # Handle errors in log handler gracefully
            print(f"Database log handler error: {e}", file=sys.stderr)
    
    def _get_category(self, logger_name: str) -> str:
        """Extract category from logger name."""
        
        if 'api' in logger_name.lower():
            return 'api'
        elif 'sam' in logger_name.lower():
            return 'sam_service'
        elif 'openai' in logger_name.lower():
            return 'openai_service'
        elif 'training' in logger_name.lower():
            return 'training'
        elif 'storage' in logger_name.lower():
            return 'storage'
        elif 'database' in logger_name.lower():
            return 'database'
        else:
            return 'system'
    
    def _extract_details(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract additional details from log record."""
        
        details = {
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread
        }
        
        # Add exception details if present
        if record.exc_info:
            details['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None
            }
        
        # Add custom fields
        for key, value in record.__dict__.items():
            if key.startswith('custom_') or key in ('user_id', 'image_id', 'request_id'):
                details[key] = value
        
        return details


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceLogFilter(logging.Filter):
    """
    Filter that adds performance metrics to log records.
    
    This helps track system performance and identify bottlenecks.
    """
    
    def __init__(self):
        super().__init__()
        self._start_time = datetime.now()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to the record."""
        
        try:
            # Add uptime
            uptime = datetime.now() - self._start_time
            record.uptime_seconds = uptime.total_seconds()
            
            # Add memory usage
            try:
                import psutil
                process = psutil.Process()
                record.memory_mb = process.memory_info().rss / (1024 * 1024)
                record.cpu_percent = process.cpu_percent()
            except ImportError:
                # psutil not available
                pass
            
        except Exception:
            # Don't fail if performance monitoring fails
            pass
        
        return True


# =============================================================================
# LOGGING SETUP FUNCTIONS
# =============================================================================

def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_database_logging: bool = True,
    enable_performance_monitoring: bool = True
) -> None:
    """
    Set up comprehensive logging for the VisionFlow AI system.
    
    This configures all loggers, handlers, and formatters needed
    for proper logging throughout the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses config setting)
        enable_database_logging: Whether to log to database
        enable_performance_monitoring: Whether to add performance metrics
    """
    
    settings = get_settings()
    
    # Determine log level
    if not log_level:
        log_level = settings.log_level
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Determine log file
    if not log_file:
        log_file = settings.log_file
    
    # Create log directory
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set root logger level
    root_logger.setLevel(numeric_level)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if sys.stdout.isatty():  # Only use colors for interactive terminals
        console_formatter = ColoredConsoleFormatter()
    else:
        console_formatter = StructuredFormatter(include_traceback=False)
    
    console_handler.setFormatter(console_formatter)
    
    # Add performance filter if enabled
    if enable_performance_monitoring:
        perf_filter = PerformanceLogFilter()
        console_handler.addFilter(perf_filter)
    
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=settings.log_max_size * 1024 * 1024,  # Convert MB to bytes
        backupCount=settings.log_backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(StructuredFormatter(include_traceback=True))
    
    if enable_performance_monitoring:
        file_handler.addFilter(perf_filter)
    
    root_logger.addHandler(file_handler)
    
    # Database handler for important events
    if enable_database_logging:
        try:
            db_handler = DatabaseLogHandler(min_level=logging.WARNING)
            root_logger.addHandler(db_handler)
        except Exception as e:
            # Don't fail startup if database logging setup fails
            console_handler.handle(logging.LogRecord(
                name=__name__,
                level=logging.WARNING,
                pathname=__file__,
                lineno=0,
                msg=f"Failed to setup database logging: {e}",
                args=(),
                exc_info=None
            ))
    
    # Configure specific loggers
    _configure_component_loggers(numeric_level)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")


def _configure_component_loggers(default_level: int) -> None:
    """Configure logging levels for different components."""
    
    # Set levels for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # SAM service might be very verbose
    logging.getLogger('sam_service').setLevel(max(default_level, logging.INFO))
    
    # Database queries can be noisy in debug mode
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
    
    # FastAPI access logs
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)


# =============================================================================
# CONTEXT MANAGERS AND UTILITIES
# =============================================================================

class LogContext:
    """
    Context manager for adding context to log messages.
    
    This allows you to add request IDs, user IDs, or other context
    information to all log messages within a block.
    """
    
    def __init__(self, **context):
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_function_call(logger: logging.Logger = None):
    """
    Decorator for logging function calls with parameters and timing.
    
    This is useful for debugging and performance monitoring.
    """
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Log function entry
            logger.debug(f"Entering {func_name}")
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Completed {func_name} in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                # Log exception
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed {func_name} after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    This is a convenience function that ensures consistent logger naming
    throughout the application.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    duration: float,
    **metrics
) -> None:
    """
    Log performance metrics for an operation.
    
    This provides a standardized way to log performance data
    that can be easily parsed and analyzed.
    
    Args:
        logger: Logger to use
        operation: Name of the operation
        duration: Duration in seconds
        **metrics: Additional metrics to log
    """
    
    metrics_data = {
        'operation': operation,
        'duration_seconds': duration,
        **metrics
    }
    
    # Add to log record as custom field
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg=f"Performance: {operation} completed in {duration:.3f}s",
        args=(),
        exc_info=None
    )
    
    # Add metrics as custom fields
    for key, value in metrics_data.items():
        setattr(record, f"metric_{key}", value)
    
    logger.handle(record)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any] = None,
    user_id: str = None,
    request_id: str = None
) -> None:
    """
    Log an error with additional context information.
    
    This provides a standardized way to log errors with context
    that's useful for debugging and monitoring.
    
    Args:
        logger: Logger to use
        error: Exception that occurred
        context: Additional context information
        user_id: User ID if applicable
        request_id: Request ID if applicable
    """
    
    error_context = {
        'error_type': type(error).__name__,
        'error_message': str(error),
    }
    
    if context:
        error_context.update(context)
    
    if user_id:
        error_context['user_id'] = user_id
    
    if request_id:
        error_context['request_id'] = request_id
    
    # Create log record with context
    with LogContext(**error_context):
        logger.error(f"Error occurred: {error}", exc_info=True)


# =============================================================================
# MONITORING INTEGRATION
# =============================================================================

def setup_error_alerting(webhook_url: Optional[str] = None) -> None:
    """
    Set up error alerting via webhooks.
    
    This sends critical errors to external monitoring systems
    like Slack or Discord for immediate notification.
    
    Args:
        webhook_url: Webhook URL for alerts
    """
    
    if not webhook_url:
        settings = get_settings()
        webhook_url = settings.alert_webhook_url
    
    if not webhook_url:
        return
    
    class WebhookHandler(logging.Handler):
        """Handler that sends critical errors to webhook."""
        
        def __init__(self, webhook_url: str):
            super().__init__()
            self.webhook_url = webhook_url
            self.setLevel(logging.CRITICAL)
        
        def emit(self, record):
            try:
                import requests
                
                message = {
                    "text": f"🚨 Critical Error in VisionFlow AI",
                    "attachments": [{
                        "color": "danger",
                        "fields": [
                            {"title": "Error", "value": record.getMessage(), "short": False},
                            {"title": "Logger", "value": record.name, "short": True},
                            {"title": "Time", "value": datetime.fromtimestamp(record.created).isoformat(), "short": True}
                        ]
                    }]
                }
                
                requests.post(self.webhook_url, json=message, timeout=5)
                
            except Exception:
                # Don't raise if webhook fails
                pass
    
    # Add webhook handler to root logger
    webhook_handler = WebhookHandler(webhook_url)
    logging.getLogger().addHandler(webhook_handler)