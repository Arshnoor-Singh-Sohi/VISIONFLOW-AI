"""
VisionFlow AI - SAM Service Utilities
=====================================

Utility functions for the SAM service including logging setup, system monitoring,
file management, and other helper functions.
"""

import os
import time
import logging
import psutil
import platform
from typing import Dict, Any
from pathlib import Path


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the SAM service.
    
    This sets up both console and file logging with appropriate formatting
    for a production service.
    """
    # Create logs directory
    log_dir = Path("/app/logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - %(message)s"
    )
    
    # Set up root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler
            logging.FileHandler(
                log_dir / "sam_service.log",
                mode='a',
                encoding='utf-8'
            )
        ]
    )
    
    # Set specific log levels for noisy libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")


# =============================================================================
# SYSTEM MONITORING
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information for monitoring and debugging.
    
    This includes CPU, memory, disk usage, and other system metrics
    that are useful for performance monitoring and troubleshooting.
    """
    try:
        # CPU information
        cpu_info = {
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent_used': memory.percent
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            'total_gb': round(disk.total / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'percent_used': round((disk.used / disk.total) * 100, 1)
        }
        
        # Platform information
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        return {
            'timestamp': time.time(),
            'cpu': cpu_info,
            'memory': memory_info,
            'disk': disk_info,
            'platform': platform_info,
            'python_version': platform.python_version()
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get system info: {e}")
        return {
            'error': str(e),
            'timestamp': time.time()
        }


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information if available.
    
    This provides details about GPU memory usage, utilization,
    and other CUDA-specific metrics.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {'gpu_available': False}
        
        gpu_info = {
            'gpu_available': True,
            'gpu_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'devices': []
        }
        
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                'id': i,
                'name': device_props.name,
                'memory_total_gb': round(device_props.total_memory / (1024**3), 2),
                'memory_allocated_gb': round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                'memory_cached_gb': round(torch.cuda.memory_reserved(i) / (1024**3), 2),
                'compute_capability': f"{device_props.major}.{device_props.minor}"
            }
            gpu_info['devices'].append(device_info)
        
        return gpu_info
        
    except ImportError:
        return {'gpu_available': False, 'error': 'PyTorch not available'}
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get GPU info: {e}")
        return {'gpu_available': False, 'error': str(e)}


def monitor_resource_usage(func):
    """
    Decorator to monitor CPU and memory usage of a function.
    
    This is useful for profiling specific operations and identifying
    performance bottlenecks.
    """
    def wrapper(*args, **kwargs):
        # Get initial resource usage
        process = psutil.Process()
        start_cpu = process.cpu_percent()
        start_memory = process.memory_info().rss / (1024**2)  # MB
        start_time = time.time()
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Get final resource usage
            end_time = time.time()
            end_cpu = process.cpu_percent()
            end_memory = process.memory_info().rss / (1024**2)  # MB
            
            # Log resource usage
            logger = logging.getLogger(__name__)
            logger.info(
                f"Function {func.__name__} - "
                f"Time: {end_time - start_time:.2f}s, "
                f"CPU: {end_cpu - start_cpu:.1f}%, "
                f"Memory: {end_memory - start_memory:+.1f}MB"
            )
            
            return result
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Function {func.__name__} failed: {e}")
            raise
    
    return wrapper


# =============================================================================
# FILE MANAGEMENT
# =============================================================================

def cleanup_old_files(directory: Path, max_age_hours: int = 24) -> int:
    """
    Clean up files older than the specified age.
    
    This prevents temporary directories from growing indefinitely
    by removing old files that are no longer needed.
    
    Args:
        directory: Directory to clean up
        max_age_hours: Maximum age of files to keep (in hours)
        
    Returns:
        Number of files removed
    """
    if not directory.exists():
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    files_removed = 0
    
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        files_removed += 1
                    except Exception as e:
                        logging.getLogger(__name__).warning(
                            f"Failed to delete {file_path}: {e}"
                        )
        
        # Also remove empty directories
        for dir_path in directory.rglob('*'):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"Failed to remove empty directory {dir_path}: {e}"
                    )
        
        if files_removed > 0:
            logging.getLogger(__name__).info(
                f"Cleaned up {files_removed} files from {directory}"
            )
        
        return files_removed
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Cleanup failed for {directory}: {e}")
        return 0


def get_directory_size(directory: Path) -> Dict[str, Any]:
    """
    Get the total size of a directory and its contents.
    
    This provides information about disk usage for monitoring
    and capacity planning.
    """
    if not directory.exists():
        return {'exists': False}
    
    total_size = 0
    file_count = 0
    
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            'exists': True,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024**2), 2),
            'total_size_gb': round(total_size / (1024**3), 2),
            'file_count': file_count,
            'directory': str(directory)
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get directory size: {e}")
        return {
            'exists': True,
            'error': str(e),
            'directory': str(directory)
        }


def ensure_directory_exists(directory: Path, create_parents: bool = True) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
        create_parents: Whether to create parent directories
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        directory.mkdir(parents=create_parents, exist_ok=True)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(
            f"Failed to create directory {directory}: {e}"
        )
        return False


# =============================================================================
# HEALTH CHECK UTILITIES
# =============================================================================

def check_disk_space(path: str, min_free_gb: float = 1.0) -> Dict[str, Any]:
    """
    Check if there's enough free disk space.
    
    This is important for services that write temporary files
    to prevent running out of disk space.
    """
    try:
        disk_usage = psutil.disk_usage(path)
        free_gb = disk_usage.free / (1024**3)
        
        return {
            'path': path,
            'free_gb': round(free_gb, 2),
            'total_gb': round(disk_usage.total / (1024**3), 2),
            'used_percent': round((disk_usage.used / disk_usage.total) * 100, 1),
            'sufficient_space': free_gb >= min_free_gb,
            'min_required_gb': min_free_gb
        }
        
    except Exception as e:
        return {
            'path': path,
            'error': str(e),
            'sufficient_space': False
        }


def check_memory_usage(max_memory_percent: float = 90.0) -> Dict[str, Any]:
    """
    Check if memory usage is within acceptable limits.
    
    High memory usage can cause performance issues or out-of-memory errors.
    """
    try:
        memory = psutil.virtual_memory()
        
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent,
            'memory_ok': memory.percent <= max_memory_percent,
            'max_allowed_percent': max_memory_percent
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'memory_ok': False
        }


def comprehensive_health_check() -> Dict[str, Any]:
    """
    Perform a comprehensive health check of the system.
    
    This combines multiple checks to provide an overall health status.
    """
    health_status = {
        'timestamp': time.time(),
        'overall_healthy': True,
        'checks': {}
    }
    
    # Disk space check
    disk_check = check_disk_space('/', min_free_gb=1.0)
    health_status['checks']['disk_space'] = disk_check
    if not disk_check.get('sufficient_space', False):
        health_status['overall_healthy'] = False
    
    # Memory usage check
    memory_check = check_memory_usage(max_memory_percent=90.0)
    health_status['checks']['memory'] = memory_check
    if not memory_check.get('memory_ok', False):
        health_status['overall_healthy'] = False
    
    # System info
    health_status['system_info'] = get_system_info()
    
    # GPU info (if available)
    health_status['gpu_info'] = get_gpu_info()
    
    return health_status


# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================

class PerformanceTimer:
    """
    Context manager for timing operations.
    
    Usage:
        with PerformanceTimer("operation_name") as timer:
            # Do some work
            pass
        print(f"Operation took {timer.elapsed_time} seconds")
    """
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.debug(f"{self.operation_name} completed in {self.elapsed_time:.2f}s")
        else:
            self.logger.error(f"{self.operation_name} failed after {self.elapsed_time:.2f}s")


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into a human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "256 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def validate_image_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate that a file is a valid image.
    
    This performs basic validation without loading the full image,
    which is useful for API input validation.
    """
    try:
        if not file_path.exists():
            return {'valid': False, 'error': 'File does not exist'}
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            return {'valid': False, 'error': 'File is empty'}
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return {'valid': False, 'error': 'File too large (>50MB)'}
        
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        if file_path.suffix.lower() not in valid_extensions:
            return {'valid': False, 'error': 'Invalid file extension'}
        
        # Try to open with PIL to validate format
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                if width < 10 or height < 10:
                    return {'valid': False, 'error': 'Image too small'}
                if width > 8192 or height > 8192:
                    return {'valid': False, 'error': 'Image too large'}
        except Exception as e:
            return {'valid': False, 'error': f'Invalid image format: {e}'}
        
        return {
            'valid': True,
            'file_size_bytes': file_size,
            'file_size_formatted': format_bytes(file_size)
        }
        
    except Exception as e:
        return {'valid': False, 'error': f'Validation failed: {e}'}