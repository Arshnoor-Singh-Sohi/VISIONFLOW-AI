"""
VisionFlow AI - Main FastAPI Application
========================================

This is the central API server that coordinates all VisionFlow AI services.
Think of this as the "command center" that orchestrates image processing,
classification, training, and data management.

Architecture Overview:
- FastAPI with async/await for high performance
- Dependency injection for clean separation of concerns
- Comprehensive error handling and logging
- WebSocket support for real-time updates
- Automatic API documentation with OpenAPI/Swagger
- Health checks and monitoring endpoints
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .config import get_settings, validate_configuration
from .database import init_database, close_database, get_db
from .api.router import api_router
from .utils.logging import setup_logging
from .utils.helpers import get_app_info


# =============================================================================
# CONFIGURATION AND GLOBALS
# =============================================================================

settings = get_settings()
logger = logging.getLogger(__name__)

# WebSocket connection manager for real-time updates
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected WebSocket clients."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

# Global connection manager instance
connection_manager = ConnectionManager()


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    This handles startup and shutdown events for the FastAPI application,
    ensuring all services are properly initialized and cleaned up.
    """
    logger.info("Starting VisionFlow AI backend...")
    
    # Validate configuration before starting
    config_errors = validate_configuration()
    if config_errors:
        logger.error("Configuration validation failed:")
        for error in config_errors:
            logger.error(f"  - {error}")
        raise RuntimeError("Invalid configuration")
    
    try:
        # Initialize database
        init_database()
        logger.info("Database initialized successfully")
        
        # Test external service connections
        await test_external_services()
        
        # Start background tasks
        background_tasks = await start_background_tasks()
        
        logger.info("VisionFlow AI backend started successfully")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    finally:
        # Cleanup during shutdown
        logger.info("Shutting down VisionFlow AI backend...")
        
        # Cancel background tasks
        for task in background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close database connections
        close_database()
        
        # Close WebSocket connections
        for connection in connection_manager.active_connections.copy():
            await connection.close()
        
        logger.info("VisionFlow AI backend shutdown complete")


async def test_external_services():
    """
    Test connections to external services during startup.
    
    This verifies that the SAM service and OpenAI API are accessible
    before allowing the application to start serving requests.
    """
    from .services.sam_service import get_sam_service
    from .services.openai_service import get_openai_service
    
    # Test SAM service
    try:
        sam_service = get_sam_service()
        health_result = await sam_service.health_check()
        if health_result['status'] != 'healthy':
            logger.warning(f"SAM service not healthy: {health_result}")
        else:
            logger.info("SAM service connection verified")
    except Exception as e:
        logger.error(f"SAM service connection failed: {e}")
        # Don't fail startup - SAM service might start later
    
    # Test OpenAI API
    try:
        openai_service = get_openai_service()
        test_result = await openai_service.test_connection()
        if test_result['status'] != 'success':
            logger.warning(f"OpenAI API test failed: {test_result}")
        else:
            logger.info("OpenAI API connection verified")
    except Exception as e:
        logger.error(f"OpenAI API connection failed: {e}")
        # Don't fail startup - might be a temporary issue


async def start_background_tasks() -> List[asyncio.Task]:
    """
    Start background tasks for the application.
    
    These tasks run continuously in the background to handle
    periodic maintenance, monitoring, and automated processing.
    """
    tasks = []
    
    # Start health monitoring task
    if settings.enable_monitoring:
        task = asyncio.create_task(health_monitoring_task())
        tasks.append(task)
        logger.info("Health monitoring task started")
    
    # Start automatic training task
    if settings.enable_training:
        task = asyncio.create_task(training_monitoring_task())
        tasks.append(task)
        logger.info("Training monitoring task started")
    
    return tasks


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="VisionFlow AI",
    description="Complete computer vision pipeline with SAM segmentation and OpenAI classification",
    version="1.0.0",
    docs_url=settings.docs_url if settings.enable_docs else None,
    redoc_url=settings.redoc_url if settings.enable_docs else None,
    lifespan=lifespan
)

# =============================================================================
# MIDDLEWARE CONFIGURATION
# =============================================================================

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security middleware
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1"] + settings.cors_origins
    )

# Static files middleware (for serving uploaded images, etc.)
app.mount("/static", StaticFiles(directory=settings.upload_path), name="static")


# =============================================================================
# API ROUTES
# =============================================================================

# Include all API routes
app.include_router(api_router, prefix="/api/v1")


# =============================================================================
# WEBSOCKET ENDPOINTS
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.
    
    This allows the frontend to receive real-time notifications about:
    - Image processing progress
    - Training status updates
    - System alerts and errors
    """
    await connection_manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await connection_manager.send_personal_message({
            "type": "connection",
            "message": "Connected to VisionFlow AI"
        }, websocket)
        
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_json()
            
            # Handle different message types from frontend
            if data.get("type") == "ping":
                await connection_manager.send_personal_message({
                    "type": "pong",
                    "timestamp": data.get("timestamp")
                }, websocket)
            
            elif data.get("type") == "subscribe":
                # Handle subscription to specific event types
                await connection_manager.send_personal_message({
                    "type": "subscribed",
                    "events": data.get("events", [])
                }, websocket)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """
    Root endpoint with basic application information.
    
    This provides a friendly welcome message and basic info
    about the API for developers.
    """
    app_info = get_app_info()
    
    return {
        "message": "Welcome to VisionFlow AI",
        "description": "Complete computer vision pipeline with SAM segmentation and OpenAI classification",
        "version": app_info["version"],
        "status": "running",
        "docs_url": f"{app_info['base_url']}/docs" if settings.enable_docs else None,
        "api_base": f"{app_info['base_url']}/api/v1",
        "websocket_url": f"{app_info['ws_base_url']}/ws",
        "features": [
            "Image upload and processing",
            "SAM-based segmentation",
            "OpenAI-powered classification",
            "Continuous model training",
            "Real-time WebSocket updates",
            "Comprehensive monitoring"
        ]
    }


@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    
    This provides detailed information about the health of all
    system components for monitoring and debugging.
    """
    from .database import db_manager
    from .services.sam_service import get_sam_service
    from .services.openai_service import get_openai_service
    
    health_status = {
        "status": "healthy",
        "timestamp": get_app_info()["timestamp"],
        "components": {}
    }
    
    # Check database health
    try:
        db_health = db_manager.health_check()
        health_status["components"]["database"] = db_health
        if db_health["status"] != "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check SAM service health
    try:
        sam_service = get_sam_service()
        sam_health = await sam_service.health_check()
        health_status["components"]["sam_service"] = sam_health
        if sam_health["status"] != "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["sam_service"] = {
            "status": "unreachable",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check OpenAI service
    try:
        openai_service = get_openai_service()
        openai_health = await openai_service.test_connection()
        health_status["components"]["openai_service"] = openai_health
        if openai_health["status"] != "success":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["openai_service"] = {
            "status": "failed",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Add system information
    health_status["system"] = get_app_info()
    
    # Return appropriate HTTP status code
    if health_status["status"] == "unhealthy":
        return JSONResponse(status_code=503, content=health_status)
    elif health_status["status"] == "degraded":
        return JSONResponse(status_code=200, content=health_status)
    else:
        return health_status


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def health_monitoring_task():
    """
    Background task for monitoring system health.
    
    This periodically checks system health and sends alerts
    if any issues are detected.
    """
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            # Perform health check
            health_data = await health_check()
            
            # Broadcast health updates via WebSocket
            if health_data["status"] != "healthy":
                await connection_manager.broadcast({
                    "type": "health_alert",
                    "status": health_data["status"],
                    "components": health_data["components"]
                })
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health monitoring task error: {e}")
            await asyncio.sleep(60)  # Wait before retrying


async def training_monitoring_task():
    """
    Background task for monitoring and triggering model training.
    
    This checks if there's enough new data to trigger training
    and manages the training process.
    """
    while True:
        try:
            await asyncio.sleep(3600)  # Check every hour
            
            # Check if training should be triggered
            from .services.training_service import get_training_service
            training_service = get_training_service()
            
            should_train = await training_service.should_trigger_training()
            if should_train:
                logger.info("Triggering automatic model training")
                
                # Broadcast training start notification
                await connection_manager.broadcast({
                    "type": "training_started",
                    "trigger": "automatic",
                    "timestamp": get_app_info()["timestamp"]
                })
                
                # Start training (this will run in background)
                asyncio.create_task(training_service.start_training())
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Training monitoring task error: {e}")
            await asyncio.sleep(300)  # Wait before retrying


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler with helpful information."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": f"The requested endpoint '{request.url.path}' was not found",
            "available_endpoints": {
                "api": "/api/v1/",
                "docs": "/docs" if settings.enable_docs else None,
                "health": "/health",
                "websocket": "/ws"
            }
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler with error logging."""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please check the logs for details.",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


# =============================================================================
# UTILITY FUNCTIONS FOR BACKGROUND TASKS
# =============================================================================

async def broadcast_processing_update(image_id: str, status: str, details: dict = None):
    """
    Broadcast image processing updates via WebSocket.
    
    This allows the frontend to show real-time progress updates
    as images move through the processing pipeline.
    """
    message = {
        "type": "processing_update",
        "image_id": image_id,
        "status": status,
        "timestamp": get_app_info()["timestamp"]
    }
    
    if details:
        message["details"] = details
    
    await connection_manager.broadcast(message)


async def broadcast_training_update(training_run_id: str, status: str, metrics: dict = None):
    """
    Broadcast model training updates via WebSocket.
    
    This keeps users informed about training progress and results.
    """
    message = {
        "type": "training_update",
        "training_run_id": training_run_id,
        "status": status,
        "timestamp": get_app_info()["timestamp"]
    }
    
    if metrics:
        message["metrics"] = metrics
    
    await connection_manager.broadcast(message)


# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",  # Changed from "main:app" to "backend.main:app"
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )