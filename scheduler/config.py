"""
VisionFlow AI - Scheduler Configuration
=======================================

This module handles configuration for the scheduling system. Think of this as 
the "control panel" for your automated tasks - it determines when things run,
how often they execute, and what parameters they use.

The scheduler is like having a reliable assistant who never forgets to do
important maintenance tasks like:
- Processing new images that arrive overnight
- Training models when enough new data accumulates
- Cleaning up old temporary files
- Generating daily reports
- Checking system health
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime, time
from pydantic import BaseSettings, validator, Field
from pathlib import Path


class SchedulerSettings(BaseSettings):
    """
    Configuration settings for the VisionFlow AI scheduler.
    
    This class acts like a detailed instruction manual for your automated
    assistant, telling it exactly what to do, when to do it, and how to
    handle various situations that might arise.
    """
    
    # =============================================================================
    # CORE SCHEDULER SETTINGS
    # =============================================================================
    
    enabled: bool = Field(
        default=True,
        description="Whether the scheduler should run at all"
    )
    
    timezone: str = Field(
        default="UTC",
        description="Timezone for all scheduled tasks"
    )
    
    max_concurrent_tasks: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of tasks that can run simultaneously"
    )
    
    task_timeout_minutes: int = Field(
        default=60,
        ge=5,
        le=480,
        description="Maximum time a task can run before being killed"
    )
    
    # =============================================================================
    # DAILY PROCESSING CONFIGURATION
    # =============================================================================
    
    daily_processing_enabled: bool = Field(
        default=True,
        description="Enable daily automated image processing"
    )
    
    daily_processing_time: str = Field(
        default="02:00",
        regex=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$",
        description="Time to run daily processing (HH:MM format)"
    )
    
    daily_image_source: str = Field(
        default="folder",
        regex="^(folder|url|api)$",
        description="Source for daily images: folder, url, or api"
    )
    
    daily_image_folder: str = Field(
        default="./data/daily_inputs",
        description="Folder to monitor for new images"
    )
    
    daily_image_url: Optional[str] = Field(
        default=None,
        description="URL to fetch daily images from"
    )
    
    daily_max_images: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of images to process daily"
    )
    
    # =============================================================================
    # TRAINING SCHEDULE CONFIGURATION
    # =============================================================================
    
    auto_training_enabled: bool = Field(
        default=True,
        description="Enable automatic model training"
    )
    
    training_check_interval_hours: int = Field(
        default=6,
        ge=1,
        le=72,
        description="How often to check if training should be triggered"
    )
    
    min_new_samples_for_training: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Minimum new samples needed to trigger training"
    )
    
    training_time_preference: str = Field(
        default="night",
        regex="^(any|night|day|weekend)$",
        description="Preferred time for training: any, night, day, weekend"
    )
    
    max_training_duration_hours: int = Field(
        default=4,
        ge=1,
        le=24,
        description="Maximum time to allow for training before canceling"
    )
    
    # =============================================================================
    # MAINTENANCE SCHEDULE CONFIGURATION
    # =============================================================================
    
    maintenance_enabled: bool = Field(
        default=True,
        description="Enable scheduled maintenance tasks"
    )
    
    log_cleanup_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to keep log files"
    )
    
    temp_file_cleanup_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Number of hours to keep temporary files"
    )
    
    database_vacuum_interval_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="How often to vacuum/optimize the database"
    )
    
    backup_retention_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Number of days to keep backup files"
    )
    
    # =============================================================================
    # MONITORING AND REPORTING CONFIGURATION
    # =============================================================================
    
    health_check_interval_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="How often to run health checks"
    )
    
    daily_report_enabled: bool = Field(
        default=True,
        description="Generate and send daily reports"
    )
    
    daily_report_time: str = Field(
        default="08:00",
        regex=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$",
        description="Time to generate daily reports"
    )
    
    alert_webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for sending alerts"
    )
    
    alert_email_enabled: bool = Field(
        default=False,
        description="Enable email alerts"
    )
    
    alert_email_recipients: List[str] = Field(
        default_factory=list,
        description="Email addresses to receive alerts"
    )
    
    # =============================================================================
    # EXTERNAL SERVICE CONFIGURATION
    # =============================================================================
    
    backend_url: str = Field(
        default="http://localhost:8000",
        description="URL of the VisionFlow AI backend"
    )
    
    database_url: str = Field(
        default="sqlite:///./data/visionflow.db",
        description="Database connection URL"
    )
    
    redis_url: str = Field(
        default="redis://localhost:6379/1",
        description="Redis URL for task queue"
    )
    
    # =============================================================================
    # PERFORMANCE AND RESOURCE LIMITS
    # =============================================================================
    
    max_memory_usage_percent: int = Field(
        default=80,
        ge=50,
        le=95,
        description="Maximum memory usage before pausing tasks"
    )
    
    max_cpu_usage_percent: int = Field(
        default=70,
        ge=50,
        le=95,
        description="Maximum CPU usage before pausing tasks"
    )
    
    max_disk_usage_percent: int = Field(
        default=85,
        ge=70,
        le=95,
        description="Maximum disk usage before pausing tasks"
    )
    
    resource_check_interval_minutes: int = Field(
        default=5,
        ge=1,
        le=30,
        description="How often to check resource usage"
    )
    
    # =============================================================================
    # VALIDATORS
    # =============================================================================
    
    @validator('daily_image_folder')
    def ensure_daily_folder_exists(cls, v):
        """Create the daily images folder if it doesn't exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('daily_processing_time', 'daily_report_time')
    def validate_time_format(cls, v):
        """Validate time format and convert to time object for easier handling."""
        try:
            time.fromisoformat(f"{v}:00")  # Add seconds for validation
            return v
        except ValueError:
            raise ValueError(f"Invalid time format: {v}. Use HH:MM format.")
    
    @validator('alert_email_recipients')
    def validate_email_addresses(cls, v):
        """Validate email address formats."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        for email in v:
            if not re.match(email_pattern, email):
                raise ValueError(f"Invalid email address: {email}")
        
        return v
    
    @validator('backend_url', 'database_url', 'redis_url')
    def validate_urls(cls, v, field):
        """Validate URL formats."""
        import urllib.parse
        
        try:
            parsed = urllib.parse.urlparse(v)
            if not parsed.scheme:
                raise ValueError(f"URL must include scheme (http/https): {v}")
            return v
        except Exception:
            raise ValueError(f"Invalid URL format for {field.name}: {v}")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_prefix = "SCHEDULER_"
        case_sensitive = False
        
        # Allow environment variables to override config
        # For example: SCHEDULER_DAILY_PROCESSING_TIME=03:00
        
        schema_extra = {
            "example": {
                "enabled": True,
                "daily_processing_time": "02:00",
                "daily_image_folder": "./data/daily_inputs",
                "auto_training_enabled": True,
                "training_check_interval_hours": 6,
                "backend_url": "http://localhost:8000",
                "alert_webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
            }
        }


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

class TaskDefinition:
    """
    Defines a scheduled task with all its configuration.
    
    Think of this as a detailed job description that tells the scheduler
    exactly what a task should do, when it should run, and how to handle
    various situations.
    """
    
    def __init__(
        self,
        name: str,
        function: str,
        schedule: str,
        description: str,
        enabled: bool = True,
        timeout_minutes: int = 60,
        retry_attempts: int = 3,
        retry_delay_minutes: int = 5,
        required_resources: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.function = function  # Function to call (e.g., "daily_processor.process_daily_images")
        self.schedule = schedule  # Cron-like schedule (e.g., "0 2 * * *" for daily at 2 AM)
        self.description = description
        self.enabled = enabled
        self.timeout_minutes = timeout_minutes
        self.retry_attempts = retry_attempts
        self.retry_delay_minutes = retry_delay_minutes
        self.required_resources = required_resources or {}
        self.parameters = parameters or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task definition to dictionary for serialization."""
        return {
            'name': self.name,
            'function': self.function,
            'schedule': self.schedule,
            'description': self.description,
            'enabled': self.enabled,
            'timeout_minutes': self.timeout_minutes,
            'retry_attempts': self.retry_attempts,
            'retry_delay_minutes': self.retry_delay_minutes,
            'required_resources': self.required_resources,
            'parameters': self.parameters
        }


def get_default_tasks(settings: SchedulerSettings) -> List[TaskDefinition]:
    """
    Get the default set of scheduled tasks based on configuration.
    
    This function creates the standard "job board" for your automated
    assistant, listing all the regular tasks that need to be done to
    keep the system running smoothly.
    """
    tasks = []
    
    # Daily image processing task
    if settings.daily_processing_enabled:
        tasks.append(TaskDefinition(
            name="daily_image_processing",
            function="daily_processor.process_daily_images",
            schedule=f"0 {settings.daily_processing_time.split(':')[1]} {settings.daily_processing_time.split(':')[0]} * * *",
            description="Process new images that have been uploaded or collected",
            timeout_minutes=settings.task_timeout_minutes,
            parameters={
                'max_images': settings.daily_max_images,
                'source': settings.daily_image_source,
                'source_path': settings.daily_image_folder
            }
        ))
    
    # Training check task
    if settings.auto_training_enabled:
        tasks.append(TaskDefinition(
            name="training_check",
            function="daily_processor.check_training_trigger",
            schedule=f"0 0 */{settings.training_check_interval_hours} * * *",
            description="Check if model training should be triggered",
            timeout_minutes=settings.max_training_duration_hours * 60,
            parameters={
                'min_samples': settings.min_new_samples_for_training,
                'time_preference': settings.training_time_preference
            }
        ))
    
    # System maintenance tasks
    if settings.maintenance_enabled:
        tasks.extend([
            TaskDefinition(
                name="log_cleanup",
                function="daily_processor.cleanup_old_logs",
                schedule="0 30 1 * * *",  # Daily at 1:30 AM
                description="Remove old log files to save disk space",
                parameters={'days_to_keep': settings.log_cleanup_days}
            ),
            
            TaskDefinition(
                name="temp_file_cleanup",
                function="daily_processor.cleanup_temp_files",
                schedule="0 0 */4 * * *",  # Every 4 hours
                description="Remove temporary files and cache",
                parameters={'hours_to_keep': settings.temp_file_cleanup_hours}
            ),
            
            TaskDefinition(
                name="database_maintenance",
                function="daily_processor.maintain_database",
                schedule=f"0 0 3 */{settings.database_vacuum_interval_days} * *",
                description="Optimize database performance",
                timeout_minutes=120
            )
        ])
    
    # Health monitoring
    tasks.append(TaskDefinition(
        name="health_check",
        function="daily_processor.system_health_check",
        schedule=f"0 */{settings.health_check_interval_minutes} * * * *",
        description="Monitor system health and send alerts if needed",
        timeout_minutes=5,
        retry_attempts=1
    ))
    
    # Daily reporting
    if settings.daily_report_enabled:
        tasks.append(TaskDefinition(
            name="daily_report",
            function="daily_processor.generate_daily_report",
            schedule=f"0 {settings.daily_report_time.split(':')[1]} {settings.daily_report_time.split(':')[0]} * * *",
            description="Generate and send daily system report",
            parameters={
                'webhook_url': settings.alert_webhook_url,
                'email_recipients': settings.alert_email_recipients
            }
        ))
    
    return tasks


# =============================================================================
# CONFIGURATION FACTORY FUNCTIONS
# =============================================================================

def get_scheduler_settings() -> SchedulerSettings:
    """
    Get scheduler settings from environment and configuration files.
    
    This function acts like a configuration reader that gathers all
    the settings from various sources and creates a unified configuration
    object for the scheduler to use.
    """
    return SchedulerSettings()


def create_development_config() -> SchedulerSettings:
    """
    Create scheduler configuration optimized for development.
    
    This creates a "developer-friendly" configuration that runs tasks
    more frequently so you can test and debug the scheduling system
    without waiting for long intervals.
    """
    return SchedulerSettings(
        daily_processing_time="*/15 * * * *",  # Every 15 minutes for testing
        training_check_interval_hours=1,       # Check every hour
        health_check_interval_minutes=5,       # More frequent health checks
        log_cleanup_days=7,                    # Shorter retention for development
        max_concurrent_tasks=2,                # Lower limits for development
        task_timeout_minutes=30                # Shorter timeouts
    )


def create_production_config() -> SchedulerSettings:
    """
    Create scheduler configuration optimized for production.
    
    This creates a "production-ready" configuration that balances
    functionality with resource usage, using longer intervals and
    more conservative resource limits.
    """
    return SchedulerSettings(
        daily_processing_time="02:00",         # Early morning processing
        training_check_interval_hours=12,      # Check twice daily
        health_check_interval_minutes=15,      # Regular health monitoring
        log_cleanup_days=90,                   # Longer retention for analysis
        max_concurrent_tasks=5,                # Higher limits for production
        task_timeout_minutes=120,              # Longer timeouts for complex tasks
        max_memory_usage_percent=75,           # Conservative resource limits
        max_cpu_usage_percent=65
    )


def validate_schedule_format(schedule: str) -> bool:
    """
    Validate that a schedule string is in proper cron format.
    
    This function acts like a schedule format checker, making sure
    that task schedules are written correctly so the scheduler can
    understand and execute them properly.
    """
    try:
        from croniter import croniter
        return croniter.is_valid(schedule)
    except ImportError:
        # Fallback validation without croniter
        parts = schedule.split()
        
        # Basic cron format: "second minute hour day month weekday"
        if len(parts) != 6:
            return False
        
        # Check each field for basic validity
        ranges = [
            (0, 59),  # seconds
            (0, 59),  # minutes
            (0, 23),  # hours
            (1, 31),  # days
            (1, 12),  # months
            (0, 6)    # weekdays
        ]
        
        for i, (part, (min_val, max_val)) in enumerate(zip(parts, ranges)):
            if part == '*':
                continue
            
            # Handle ranges and lists
            if '/' in part or ',' in part or '-' in part:
                continue  # Complex expressions - assume valid
            
            try:
                val = int(part)
                if not (min_val <= val <= max_val):
                    return False
            except ValueError:
                return False
        
        return True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_next_run_time(schedule: str) -> datetime:
    """
    Calculate the next time a scheduled task should run.
    
    This function acts like a calendar calculator, taking a schedule
    and figuring out exactly when the next execution should happen.
    """
    try:
        from croniter import croniter
        cron = croniter(schedule, datetime.now())
        return cron.get_next(datetime)
    except ImportError:
        # Fallback: assume it's a simple daily schedule
        from datetime import timedelta
        return datetime.now() + timedelta(days=1)


def is_resource_available(
    settings: SchedulerSettings,
    required_resources: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Check if system resources are available for a task.
    
    This function acts like a resource manager, checking if the system
    has enough memory, CPU, and disk space available before allowing
    a task to start.
    """
    try:
        import psutil
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > settings.max_memory_usage_percent:
            return False, f"Memory usage too high: {memory.percent}%"
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > settings.max_cpu_usage_percent:
            return False, f"CPU usage too high: {cpu_percent}%"
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > settings.max_disk_usage_percent:
            return False, f"Disk usage too high: {disk_percent}%"
        
        # Check specific resource requirements
        required_memory = required_resources.get('memory_mb', 0)
        if required_memory > 0:
            available_memory = memory.available / (1024 * 1024)  # Convert to MB
            if available_memory < required_memory:
                return False, f"Insufficient memory: need {required_memory}MB, have {available_memory:.0f}MB"
        
        return True, "Resources available"
        
    except ImportError:
        # If psutil is not available, assume resources are available
        return True, "Resource checking not available"