"""
VisionFlow AI - API Router Configuration
========================================

This module sets up the main API router that combines all endpoint modules.
Think of this as the "traffic director" that routes incoming requests to
the appropriate handlers.
"""

from fastapi import APIRouter

from .endpoints import images, results, training, health

# Create the main API router
api_router = APIRouter()

# Include all endpoint routers with their prefixes and tags
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"],
    responses={
        503: {"description": "Service unavailable"},
        200: {"description": "Service healthy"}
    }
)

api_router.include_router(
    images.router,
    prefix="/images",
    tags=["images"],
    responses={
        400: {"description": "Bad request"},
        413: {"description": "File too large"},
        422: {"description": "Invalid file format"}
    }
)

api_router.include_router(
    results.router,
    prefix="/results",
    tags=["results"],
    responses={
        404: {"description": "Results not found"}
    }
)

api_router.include_router(
    training.router,
    prefix="/training",
    tags=["training"],
    responses={
        409: {"description": "Training already in progress"},
        503: {"description": "Training service unavailable"}
    }
)