"""
VisionFlow AI - Configuration Management
=======================================

This module handles all configuration for the VisionFlow AI backend.
We use Pydantic's BaseSettings for type-safe configuration with automatic
environment variable loading and validation.

Think of this as the "mission control" for your entire application - 
every service knows how to behave based on these settings.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings with automatic environment variable loading.
    
    Pydantic automatically loads environment variables that match field names.
    For example, DATABASE_URL env var automatically populates database_url field.
    """
    
    # =============================================================================
    # CORE APPLICATION SETTINGS
    # =============================================================================
    backend_url: str
    frontend_url: str
    auto_reload: bool = False
    domain: str
    ssl_cert_path: str
    ssl_key_path: str
    workers: int = 1  # default if needed

    app_name: str = "VisionFlow AI"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Security
    secret_key: str = Field(..., min_length=32)  # Required, minimum 32 characters
    
    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    database_url: str = Field(
        default="sqlite:///./data/visionflow.db",
        description="Database connection URL"
    )
    
    # =============================================================================
    # REDIS CONFIGURATION
    # =============================================================================
    redis_url: str = "redis://localhost:6379/0"
    
    # =============================================================================
    # EXTERNAL SERVICE URLS
    # =============================================================================
    sam_service_url: str = "http://localhost:8001"
    openai_api_key: str = Field(..., description="OpenAI API key is required")
    
    # =============================================================================
    # CORS SETTINGS
    # =============================================================================
    # cors_origins: List[str] = ["http://localhost:3000"]
    cors_origins: List[str] = Field(default_factory=list)
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Convert comma-separated string to list if needed."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    # =============================================================================
    # FILE UPLOAD SETTINGS
    # =============================================================================
    max_file_size: int = 10 * 1024 * 1024  # 10MB in bytes
    allowed_extensions: List[str] = ["jpg", "jpeg", "png", "webp"]
    
    @field_validator('allowed_extensions', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        """Convert comma-separated string to list if needed."""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(',')]
        return v
    
    # =============================================================================
    # STORAGE PATHS
    # =============================================================================
    upload_path: str = "./data/images"
    segments_path: str = "./data/segments"
    results_path: str = "./data/results"
    models_path: str = "./data/models"
    
    @field_validator('upload_path', 'segments_path', 'results_path', 'models_path')
    @classmethod
    def ensure_path_exists(cls, v):
        """Automatically create directories if they don't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    # =============================================================================
    # OPENAI CONFIGURATION
    # =============================================================================
    openai_model: str = "gpt-4-vision-preview"
    openai_max_tokens: int = 1000
    openai_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    
    # =============================================================================
    # SAM MODEL CONFIGURATION
    # =============================================================================
    sam_model_type: str = Field(default="vit_h", pattern="^(vit_h|vit_l|vit_b)$")
    sam_device: str = "cpu"
    sam_checkpoint_path: str = "./data/models/sam_vit_h_4b8939.pth"
    
    # =============================================================================
    # TRAINING CONFIGURATION
    # =============================================================================
    enable_training: bool = True
    training_batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    min_training_samples: int = Field(default=100, gt=0)
    
    # =============================================================================
    # SCHEDULER CONFIGURATION
    # =============================================================================
    enable_scheduler: bool = True
    daily_processing_time: str = Field(default="09:00", pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    daily_image_source: str = Field(default="folder", pattern="^(folder|url|api)$")
    daily_image_folder: str = "./data/daily_inputs"
    
    @field_validator('daily_image_folder')
    @classmethod
    def ensure_daily_folder_exists(cls, v):
        """Create daily images folder if it doesn't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: str = "./data/logs/visionflow.log"
    log_max_size: int = Field(default=100, gt=0)  # MB
    log_backup_count: int = Field(default=5, gt=0)
    
    @field_validator('log_file')
    @classmethod
    def ensure_log_dir_exists(cls, v):
        """Create log directory if it doesn't exist."""
        log_dir = os.path.dirname(v)
        os.makedirs(log_dir, exist_ok=True)
        return v
    
    # =============================================================================
    # API DOCUMENTATION SETTINGS
    # =============================================================================
    enable_docs: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    
    # =============================================================================
    # MONITORING SETTINGS
    # =============================================================================
    enable_monitoring: bool = True
    alert_webhook_url: Optional[str] = None
    
    # Email settings for alerts
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email_to: Optional[str] = None
    
    class Config:
        """
        Pydantic configuration.
        
        env_file tells Pydantic to automatically load from .env file
        case_sensitive=False means DATABASE_URL and database_url both work
        """
        env_file = ".env"
        case_sensitive = False
        
        # Example values for documentation
        json_schema_extra = {
            "example": {
                "app_name": "VisionFlow AI",
                "database_url": "postgresql://user:pass@localhost:5432/visionflow",
                "openai_api_key": "sk-your-api-key-here",
                "sam_service_url": "http://localhost:8001",
                "cors_origins": ["http://localhost:3000"],
                "max_file_size": 10485760,
                "enable_training": True
            }
        }


# =============================================================================
# CONFIGURATION FACTORY FUNCTION
# =============================================================================

@lru_cache()  # Cache the settings so we don't reload them on every import
def get_settings() -> Settings:
    """
    Get application settings.
    
    Using @lru_cache ensures we only create the settings object once
    and reuse it throughout the application. This is important for
    performance and consistency.
    
    Returns:
        Settings: Configured application settings
    """
    return Settings()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_production() -> bool:
    """Check if we're running in production mode."""
    settings = get_settings()
    return not settings.debug and "sqlite" not in settings.database_url.lower()


def get_database_engine_kwargs() -> dict:
    """
    Get database engine configuration based on database type.
    
    SQLite and PostgreSQL need different connection parameters.
    This function handles those differences automatically.
    """
    settings = get_settings()
    
    if "sqlite" in settings.database_url.lower():
        # SQLite specific settings
        return {
            "connect_args": {"check_same_thread": False},
            "echo": settings.debug
        }
    else:
        # PostgreSQL/other databases
        return {
            "pool_pre_ping": True,  # Validate connections before use
            "pool_recycle": 300,    # Recycle connections every 5 minutes
            "echo": settings.debug
        }


def validate_openai_key() -> bool:
    """
    Validate that OpenAI API key is properly formatted.
    
    OpenAI keys start with 'sk-' and have a specific length.
    This helps catch configuration errors early.
    """
    settings = get_settings()
    return (
        settings.openai_api_key and 
        settings.openai_api_key.startswith('sk-') and 
        len(settings.openai_api_key) > 20
    )


# =============================================================================
# STARTUP VALIDATION
# =============================================================================

def validate_configuration() -> List[str]:
    """
    Validate all configuration settings and return any errors.
    
    This function checks all critical settings and returns a list
    of error messages. Call this during application startup to
    catch configuration problems early.
    
    Returns:
        List[str]: List of configuration error messages (empty if valid)
    """
    errors = []
    settings = get_settings()
    
    # Validate OpenAI API key
    if not validate_openai_key():
        errors.append("Invalid or missing OpenAI API key")
    
    # Validate required directories exist and are writable
    required_dirs = [
        settings.upload_path,
        settings.segments_path,
        settings.results_path,
        settings.models_path
    ]
    
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            # Test write permission
            test_file = os.path.join(dir_path, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            errors.append(f"Cannot write to directory {dir_path}: {e}")
    
    # Validate SAM checkpoint exists if specified
    if (settings.sam_checkpoint_path and 
        not os.path.exists(settings.sam_checkpoint_path)):
        errors.append(f"SAM checkpoint not found: {settings.sam_checkpoint_path}")
    
    return errors


# Create a global settings instance for easy importing
settings = get_settings()