"""
VisionFlow AI - FastAPI Dependencies
====================================

This module defines dependency injection functions for FastAPI endpoints.
Dependencies provide a clean way to share common functionality across endpoints,
such as database sessions, authentication, validation, and service instances.

Think of dependencies as "shared utilities" that endpoints can request,
and FastAPI will automatically provide them when the endpoint is called.
"""

import logging
from typing import Optional, Dict, Any, Generator
from datetime import datetime
import time

from fastapi import Depends, HTTPException, Request, Header, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt

from .database import get_db
from .config import get_settings
from .services.sam_service import get_sam_service
from .services.openai_service import get_openai_service
from .services.storage_service import get_storage_service
from .services.training_service import get_training_service
from .models.database_models import SystemLog


# =============================================================================
# LOGGER SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DEPENDENCIES
# =============================================================================

def get_app_settings():
    """
    Dependency to get application settings.
    
    This provides access to configuration throughout the application
    without having to import and call get_settings() everywhere.
    """
    return get_settings()


# =============================================================================
# AUTHENTICATION DEPENDENCIES
# =============================================================================

# Security scheme for JWT tokens (if implementing authentication)
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings = Depends(get_app_settings)
) -> Optional[Dict[str, Any]]:
    """
    Dependency to get the current authenticated user.
    
    This validates JWT tokens and returns user information.
    For now, it's a placeholder that can be implemented when
    authentication is needed.
    
    Args:
        credentials: JWT credentials from Authorization header
        settings: Application settings
        
    Returns:
        User information dict or None if not authenticated
    """
    # For development, return a mock user or None
    if not credentials:
        return None
    
    try:
        # In production, you would validate the JWT token here
        # token = credentials.credentials
        # payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        # return {"user_id": payload["sub"], "email": payload["email"]}
        
        # For now, return a mock user for development
        return {
            "user_id": "dev_user",
            "email": "dev@visionflow.ai",
            "name": "Development User"
        }
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        return None


async def require_authentication(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Dependency that requires authentication.
    
    Use this dependency on endpoints that must have an authenticated user.
    """
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


# =============================================================================
# REQUEST VALIDATION DEPENDENCIES
# =============================================================================

async def validate_content_type(
    request: Request,
    allowed_types: list = ["application/json", "multipart/form-data"]
) -> str:
    """
    Dependency to validate request content type.
    
    This ensures endpoints only accept the content types they expect.
    """
    content_type = request.headers.get("content-type", "").split(";")[0]
    
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type. Expected one of: {allowed_types}"
        )
    
    return content_type


async def validate_file_upload(request: Request) -> bool:
    """
    Dependency to validate file upload requests.
    
    This checks that the request is properly formatted for file uploads
    and validates basic constraints.
    """
    content_type = request.headers.get("content-type", "")
    
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=400,
            detail="File upload must use multipart/form-data"
        )
    
    # Check content length if available
    content_length = request.headers.get("content-length")
    if content_length:
        content_length = int(content_length)
        max_size = get_settings().max_file_size
        
        if content_length > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
            )
    
    return True


# =============================================================================
# PAGINATION DEPENDENCIES
# =============================================================================

class PaginationParams:
    """Class to hold pagination parameters."""
    
    def __init__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page")
    ):
        self.page = page
        self.page_size = page_size
        self.offset = (page - 1) * page_size
        self.limit = page_size
    
    def get_pagination_info(self, total_count: int) -> Dict[str, Any]:
        """Get pagination metadata."""
        total_pages = (total_count + self.page_size - 1) // self.page_size
        
        return {
            "page": self.page,
            "page_size": self.page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": self.page < total_pages,
            "has_previous": self.page > 1
        }


def get_pagination_params() -> PaginationParams:
    """Dependency to get pagination parameters."""
    return PaginationParams()


# =============================================================================
# SERVICE DEPENDENCIES
# =============================================================================

def get_sam_service_dependency():
    """
    Dependency to get SAM service instance.
    
    This provides access to the SAM service for image segmentation.
    """
    return get_sam_service()


def get_openai_service_dependency():
    """
    Dependency to get OpenAI service instance.
    
    This provides access to the OpenAI service for classification.
    """
    return get_openai_service()


def get_storage_service_dependency():
    """
    Dependency to get storage service instance.
    
    This provides access to the storage service for file management.
    """
    return get_storage_service()


def get_training_service_dependency():
    """
    Dependency to get training service instance.
    
    This provides access to the training service for ML operations.
    """
    return get_training_service()


# =============================================================================
# REQUEST LOGGING DEPENDENCIES
# =============================================================================

class RequestLogger:
    """Class to handle request logging and timing."""
    
    def __init__(self, request: Request):
        self.request = request
        self.start_time = time.time()
        self.request_id = self._generate_request_id()
        self.user_id = None
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID for tracing."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def set_user_id(self, user_id: str):
        """Set the user ID for this request."""
        self.user_id = user_id
    
    def log_request_start(self):
        """Log the start of request processing."""
        logger.info(
            f"Request started: {self.request.method} {self.request.url.path}",
            extra={
                "request_id": self.request_id,
                "method": self.request.method,
                "path": self.request.url.path,
                "query_params": str(self.request.query_params),
                "user_agent": self.request.headers.get("user-agent"),
                "ip_address": self.request.client.host if self.request.client else None
            }
        )
    
    def log_request_end(self, status_code: int = 200, error: str = None):
        """Log the end of request processing."""
        duration = time.time() - self.start_time
        
        log_data = {
            "request_id": self.request_id,
            "method": self.request.method,
            "path": self.request.url.path,
            "status_code": status_code,
            "duration_seconds": duration,
            "user_id": self.user_id
        }
        
        if error:
            log_data["error"] = error
            logger.error(f"Request failed: {self.request.method} {self.request.url.path}", extra=log_data)
        else:
            logger.info(f"Request completed: {self.request.method} {self.request.url.path}", extra=log_data)


async def get_request_logger(request: Request) -> RequestLogger:
    """
    Dependency to get request logger.
    
    This provides request logging and timing functionality.
    """
    request_logger = RequestLogger(request)
    request_logger.log_request_start()
    return request_logger


# =============================================================================
# RATE LIMITING DEPENDENCIES
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}  # {client_ip: [(timestamp, count), ...]}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def is_allowed(
        self, 
        client_ip: str, 
        max_requests: int = 100, 
        window_seconds: int = 3600
    ) -> bool:
        """
        Check if a request is allowed under rate limiting.
        
        Args:
            client_ip: Client IP address
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()
        
        # Cleanup old entries periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(now, window_seconds)
            self.last_cleanup = now
        
        # Get or create request history for this IP
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        request_history = self.requests[client_ip]
        
        # Remove requests outside the window
        cutoff_time = now - window_seconds
        request_history[:] = [req for req in request_history if req[0] > cutoff_time]
        
        # Check if we're under the limit
        if len(request_history) >= max_requests:
            return False
        
        # Add this request
        request_history.append((now, 1))
        return True
    
    def _cleanup_old_entries(self, now: float, window_seconds: int):
        """Remove old entries to prevent memory growth."""
        cutoff_time = now - window_seconds
        
        for client_ip in list(self.requests.keys()):
            self.requests[client_ip][:] = [
                req for req in self.requests[client_ip] 
                if req[0] > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[client_ip]:
                del self.requests[client_ip]


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(
    request: Request,
    max_requests: int = 100,
    window_seconds: int = 3600
) -> bool:
    """
    Dependency to check rate limiting.
    
    This prevents abuse by limiting requests per IP address.
    """
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip, max_requests, window_seconds):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait before trying again.",
            headers={"Retry-After": str(window_seconds)}
        )
    
    return True


# =============================================================================
# DATABASE TRANSACTION DEPENDENCIES
# =============================================================================

async def get_db_transaction() -> Generator[Session, None, None]:
    """
    Dependency to get a database session with automatic transaction management.
    
    This ensures that database operations are properly wrapped in transactions
    and rolled back if an error occurs.
    """
    from .database import db_manager
    
    with db_manager.get_session_context() as session:
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction rolled back: {e}")
            raise


# =============================================================================
# HEALTH CHECK DEPENDENCIES
# =============================================================================

async def check_system_health(
    db: Session = Depends(get_db),
    settings = Depends(get_app_settings)
) -> Dict[str, Any]:
    """
    Dependency to check system health before processing requests.
    
    This can be used on critical endpoints to ensure the system
    is healthy before processing requests.
    """
    health_status = {
        "database": "unknown",
        "sam_service": "unknown",
        "openai_service": "unknown"
    }
    
    # Quick database check
    try:
        db.execute("SELECT 1")
        health_status["database"] = "healthy"
    except Exception:
        health_status["database"] = "unhealthy"
    
    # Quick SAM service check (optional)
    try:
        sam_service = get_sam_service()
        if sam_service:
            health_status["sam_service"] = "healthy"
    except Exception:
        health_status["sam_service"] = "unhealthy"
    
    # Quick OpenAI service check (optional)
    try:
        openai_service = get_openai_service()
        if openai_service:
            health_status["openai_service"] = "healthy"
    except Exception:
        health_status["openai_service"] = "unhealthy"
    
    return health_status


# =============================================================================
# REQUEST CONTEXT DEPENDENCIES
# =============================================================================

class RequestContext:
    """Class to hold request context information."""
    
    def __init__(
        self,
        request: Request,
        user: Optional[Dict[str, Any]] = None,
        request_logger: Optional[RequestLogger] = None
    ):
        self.request = request
        self.user = user
        self.request_logger = request_logger
        self.start_time = time.time()
    
    @property
    def user_id(self) -> Optional[str]:
        """Get the current user ID."""
        return self.user.get("user_id") if self.user else None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if the request is authenticated."""
        return self.user is not None
    
    def log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log user activity."""
        log_data = {
            "activity": activity,
            "user_id": self.user_id,
            "ip_address": self.request.client.host if self.request.client else None,
            "user_agent": self.request.headers.get("user-agent")
        }
        
        if details:
            log_data["details"] = details
        
        logger.info(f"User activity: {activity}", extra=log_data)


async def get_request_context(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
    request_logger: RequestLogger = Depends(get_request_logger)
) -> RequestContext:
    """
    Dependency to get request context.
    
    This provides a convenient way to access request information,
    user data, and logging throughout the request lifecycle.
    """
    context = RequestContext(request, user, request_logger)
    
    # Set user ID in request logger
    if context.user_id and request_logger:
        request_logger.set_user_id(context.user_id)
    
    return context


# =============================================================================
# VALIDATION DEPENDENCIES
# =============================================================================

async def validate_image_id(image_id: str, db: Session = Depends(get_db)) -> str:
    """
    Dependency to validate that an image ID exists in the database.
    
    This prevents endpoints from processing requests for non-existent images.
    """
    from .models.database_models import ImageRecord
    
    try:
        # Check if image exists
        image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
        
        if not image:
            raise HTTPException(
                status_code=404,
                detail=f"Image not found: {image_id}"
            )
        
        return image_id
        
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid image ID format"
        )


async def validate_training_run_id(run_id: str, db: Session = Depends(get_db)) -> str:
    """
    Dependency to validate that a training run ID exists in the database.
    """
    from .models.database_models import TrainingRun
    
    try:
        # Check if training run exists
        training_run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        
        if not training_run:
            raise HTTPException(
                status_code=404,
                detail=f"Training run not found: {run_id}"
            )
        
        return run_id
        
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid training run ID format"
        )


# =============================================================================
# PERFORMANCE MONITORING DEPENDENCIES
# =============================================================================

class PerformanceMonitor:
    """Class to monitor endpoint performance."""
    
    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self.start_time = time.time()
        self.memory_start = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def finish(self):
        """Log performance metrics when the request finishes."""
        duration = time.time() - self.start_time
        memory_end = self._get_memory_usage()
        memory_delta = memory_end - self.memory_start
        
        logger.info(
            f"Performance: {self.endpoint_name}",
            extra={
                "endpoint": self.endpoint_name,
                "duration_seconds": duration,
                "memory_start_mb": self.memory_start,
                "memory_end_mb": memory_end,
                "memory_delta_mb": memory_delta
            }
        )


async def get_performance_monitor(request: Request) -> PerformanceMonitor:
    """
    Dependency to get performance monitor.
    
    This can be used to monitor endpoint performance and resource usage.
    """
    endpoint_name = f"{request.method} {request.url.path}"
    return PerformanceMonitor(endpoint_name)