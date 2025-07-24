"""
VisionFlow AI - Helper Utilities
=================================

This module provides common utility functions used throughout the
VisionFlow AI system. These are general-purpose functions that don't
belong to any specific service but are used across multiple components.
"""

import os
import re
import uuid
import hashlib
import secrets
import string
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

from ..config import get_settings


# =============================================================================
# FILE AND PATH UTILITIES
# =============================================================================

def generate_unique_filename(
    original_filename: str,
    extension: Optional[str] = None,
    include_timestamp: bool = True
) -> str:
    """
    Generate a unique filename based on the original filename.
    
    This ensures no filename collisions while maintaining some
    human-readable connection to the original filename.
    
    Args:
        original_filename: Original filename
        extension: File extension (if None, extracted from original)
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        Unique filename string
    """
    # Clean the original filename
    base_name = Path(original_filename).stem
    base_name = sanitize_filename(base_name)
    
    # Get extension
    if not extension:
        extension = Path(original_filename).suffix.lower()
    
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    # Generate unique identifier
    unique_id = str(uuid.uuid4())[:8]
    
    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp}_{unique_id}{extension}"
    else:
        filename = f"{base_name}_{unique_id}{extension}"
    
    return filename


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize a filename to remove invalid characters.
    
    This ensures filenames are safe for all operating systems
    and don't contain problematic characters.
    
    Args:
        filename: Original filename
        max_length: Maximum length for the filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure not empty
    if not sanitized:
        sanitized = 'unnamed'
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file.
    
    This is useful for detecting duplicate files or verifying
    file integrity.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', etc.)
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def get_directory_size(directory_path: Union[str, Path]) -> int:
    """
    Calculate total size of a directory in bytes.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except (OSError, IOError):
                # Skip files that can't be accessed
                continue
    
    return total_size


def cleanup_old_files(
    directory_path: Union[str, Path],
    max_age_days: int = 7,
    file_pattern: Optional[str] = None
) -> int:
    """
    Clean up old files in a directory.
    
    Args:
        directory_path: Directory to clean
        max_age_days: Maximum age of files to keep
        file_pattern: Optional glob pattern for files to clean
        
    Returns:
        Number of files removed
    """
    path = Path(directory_path)
    
    if not path.exists():
        return 0
    
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    files_removed = 0
    
    # Use glob pattern if provided, otherwise all files
    if file_pattern:
        files = path.glob(file_pattern)
    else:
        files = path.rglob('*')
    
    for file_path in files:
        if file_path.is_file():
            try:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    files_removed += 1
                    
            except (OSError, IOError):
                # Skip files that can't be accessed
                continue
    
    return files_removed


# =============================================================================
# DATA FORMATTING AND CONVERSION
# =============================================================================

def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "256 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} EB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds into human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m 30s", "45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {seconds:.0f}s"
    
    hours = minutes // 60
    minutes = minutes % 60
    
    if hours < 24:
        return f"{hours}h {minutes}m"
    
    days = hours // 24
    hours = hours % 24
    
    return f"{days}d {hours}h"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def parse_json_safely(json_string: str, default: Any = None) -> Any:
    """
    Parse JSON string safely without raising exceptions.
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


# =============================================================================
# SECURITY AND ENCRYPTION
# =============================================================================

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Length of the token
        
    Returns:
        Random token string
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_api_key() -> str:
    """
    Generate an API key in a standard format.
    
    Returns:
        API key string
    """
    prefix = "vfa"  # VisionFlow AI prefix
    key_part = generate_secure_token(40)
    return f"{prefix}_{key_part}"


def hash_password(password: str) -> str:
    """
    Hash a password using a secure algorithm.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    import bcrypt
    
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password: Plain text password
        hashed: Hashed password
        
    Returns:
        True if password matches
    """
    import bcrypt
    
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


# =============================================================================
# APPLICATION INFORMATION
# =============================================================================

def get_app_info() -> Dict[str, Any]:
    """
    Get comprehensive application information.
    
    This provides system information useful for monitoring,
    debugging, and API responses.
    
    Returns:
        Dictionary with application information
    """
    settings = get_settings()
    
    # Get system information
    import platform
    import sys
    
    app_info = {
        'name': settings.app_name,
        'version': settings.app_version,
        'debug_mode': settings.debug,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'uptime_seconds': None,  # Would be calculated from startup time
        'system': {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0]
        },
        'configuration': {
            'max_file_size_mb': settings.max_file_size / (1024 * 1024),
            'allowed_extensions': settings.allowed_extensions,
            'training_enabled': settings.enable_training,
            'scheduler_enabled': settings.enable_scheduler
        }
    }
    
    # Add URLs based on environment
    if settings.debug:
        app_info['base_url'] = 'http://localhost:8000'
        app_info['ws_base_url'] = 'ws://localhost:8000'
    else:
        # In production, these would be actual domain URLs
        app_info['base_url'] = 'https://api.visionflow.ai'
        app_info['ws_base_url'] = 'wss://api.visionflow.ai'
    
    return app_info


def get_health_summary() -> Dict[str, Any]:
    """
    Get a quick health summary of the system.
    
    Returns:
        Dictionary with health information
    """
    try:
        import psutil
        
        # CPU and memory info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'status': 'healthy',
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': (disk.used / disk.total) * 100,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except ImportError:
        return {
            'status': 'unknown',
            'message': 'psutil not available',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

class VisionFlowError(Exception):
    """Base exception for VisionFlow AI."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            'error': self.message,
            'code': self.code,
            'details': self.details,
            'type': self.__class__.__name__
        }


class ValidationError(VisionFlowError):
    """Raised when input validation fails."""
    pass


class ProcessingError(VisionFlowError):
    """Raised when image processing fails."""
    pass


class ServiceUnavailableError(VisionFlowError):
    """Raised when external service is unavailable."""
    pass


def handle_exception(func):
    """
    Decorator for standardized exception handling.
    
    This decorator catches exceptions and converts them to
    standardized error responses.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except VisionFlowError:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            # Convert unexpected exceptions to ProcessingError
            raise ProcessingError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                code="PROCESSING_ERROR",
                details={'function': func.__name__, 'original_error': str(e)}
            )
    
    return wrapper


# =============================================================================
# ASYNC UTILITIES
# =============================================================================

async def run_in_background(func, *args, **kwargs):
    """
    Run a function in the background (fire and forget).
    
    This is useful for operations that don't need to block
    the main request/response cycle.
    """
    import asyncio
    
    loop = asyncio.get_event_loop()
    
    if asyncio.iscoroutinefunction(func):
        task = loop.create_task(func(*args, **kwargs))
    else:
        task = loop.run_in_executor(None, func, *args, **kwargs)
    
    # Don't await - let it run in background
    return task


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for delay on each retry
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Wait before retrying
                    import asyncio
                    await asyncio.sleep(min(delay, max_delay))
                    delay *= backoff_factor
            
            # All retries failed
            raise last_exception
        
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Wait before retrying
                    import time
                    time.sleep(min(delay, max_delay))
                    delay *= backoff_factor
            
            # All retries failed
            raise last_exception
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ('.yml', '.yaml'):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML configuration files")
        else:
            return json.load(f)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later configurations override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        if isinstance(config, dict):
            merged.update(config)
    
    return merged


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """
    Validate configuration has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        List of missing keys (empty if valid)
    """
    missing_keys = []
    
    for key in required_keys:
        if key not in config or config[key] is None:
            missing_keys.append(key)
    
    return missing_keys