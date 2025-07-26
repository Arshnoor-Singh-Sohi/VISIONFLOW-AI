"""
VisionFlow AI - SAM Processing Service
=====================================

This is a standalone FastAPI service that runs the Segment Anything Model (SAM).
It's designed to run in a Docker container with all the necessary ML dependencies
isolated from the main application.

Why separate service?
- SAM has heavy dependencies (PyTorch, CUDA, etc.)
- Computational isolation (can run on GPU while main app uses CPU)
- Independent scaling (can run multiple SAM instances)
- Easier deployment and resource management
- Fault isolation (SAM crashes don't affect main app)
"""

import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from io import BytesIO

import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from sam_processor import SAMProcessor
from utils import setup_logging, get_system_info, cleanup_old_files


# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment variables with defaults
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/sam_vit_h_4b8939.pth')
DEVICE = os.getenv('DEVICE', 'cpu')  # 'cpu', 'cuda', or 'mps'
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2048'))  # Max dimension in pixels
CLEANUP_INTERVAL = int(os.getenv('CLEANUP_INTERVAL', '3600'))  # Cleanup every hour
TEMP_DIR = Path(os.getenv('TEMP_DIR', '/app/temp'))

# Create necessary directories
TEMP_DIR.mkdir(parents=True, exist_ok=True)
Path('/app/segments').mkdir(exist_ok=True)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="VisionFlow SAM Service",
    description="Segment Anything Model (SAM) processing service for VisionFlow AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for communication with main backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global SAM processor instance
sam_processor: Optional[SAMProcessor] = None


# =============================================================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize the SAM model when the service starts.
    
    This loads the model into memory and prepares it for processing.
    Loading can take 30-60 seconds depending on the model size and device.
    """
    global sam_processor
    
    logger.info("Starting VisionFlow SAM Service...")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Max image size: {MAX_IMAGE_SIZE}")
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"SAM model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        # Initialize SAM processor
        logger.info("Loading SAM model... (this may take a minute)")
        sam_processor = SAMProcessor(
            model_path=MODEL_PATH,
            device=DEVICE,
            max_image_size=MAX_IMAGE_SIZE
        )
        
        logger.info("SAM model loaded successfully!")
        
        # Start background cleanup task
        asyncio.create_task(periodic_cleanup())
        
        # Log system information
        system_info = get_system_info()
        logger.info(f"System info: {json.dumps(system_info, indent=2)}")
        
    except Exception as e:
        logger.error(f"Failed to initialize SAM service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the service shuts down."""
    logger.info("Shutting down SAM service...")
    
    # Clean up temporary files
    cleanup_old_files(TEMP_DIR, max_age_hours=0)  # Clean all files
    
    logger.info("SAM service shutdown complete")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    This provides detailed information about the service status,
    model readiness, and system resources.
    """
    try:
        # Check if SAM processor is loaded
        model_loaded = sam_processor is not None
        
        # Get system information
        system_info = get_system_info()
        
        # Check GPU availability if using CUDA
        gpu_info = {}
        if DEVICE == 'cuda' and torch.cuda.is_available():
            gpu_info = {
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                'gpu_memory_allocated': torch.cuda.memory_allocated(0) if torch.cuda.device_count() > 0 else 0,
                'gpu_memory_cached': torch.cuda.memory_reserved(0) if torch.cuda.device_count() > 0 else 0
            }
        
        return {
            "status": "healthy" if model_loaded else "initializing",
            "model_loaded": model_loaded,
            "device": DEVICE,
            "model_path": MODEL_PATH,
            "max_image_size": MAX_IMAGE_SIZE,
            "system_info": system_info,
            "gpu_info": gpu_info,
            "response_time_ms": time.time() * 1000  # For response time measurement
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/segment")
async def segment_image(
    image: UploadFile = File(...),
    config: str = Form(...)
):
    """
    Segment an uploaded image using SAM.
    
    This is the main endpoint that receives an image and returns all the
    segments (distinct objects/regions) found in the image.
    
    Args:
        image: The image file to segment
        config: JSON string with processing configuration
        
    Returns:
        JSON response with segment information and processing metadata
    """
    if sam_processor is None:
        raise HTTPException(
            status_code=503,
            detail="SAM model not loaded yet. Please wait and try again."
        )
    
    start_time = time.time()
    temp_image_path = None
    
    try:
        # Parse configuration
        try:
            config_data = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="Invalid JSON configuration")
        
        # Validate image file
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=422, detail="File must be an image")
        
        # Save uploaded image to temporary file
        temp_image_path = TEMP_DIR / f"temp_{int(time.time())}_{image.filename}"
        
        with open(temp_image_path, 'wb') as f:
            content = await image.read()
            f.write(content)
        
        logger.info(f"Processing image: {image.filename} ({len(content)} bytes)")
        
        # Validate image can be opened
        try:
            with Image.open(temp_image_path) as img:
                image_width, image_height = img.size
                logger.info(f"Image dimensions: {image_width}x{image_height}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid image file: {e}")
        
        # Process with SAM
        result = await sam_processor.process_image(
            image_path=str(temp_image_path),
            min_area=config_data.get('min_area', 1000),
            max_segments=config_data.get('max_segments', 60),
            confidence_threshold=config_data.get('confidence_threshold', 0.7)
        )
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'success': True,
            'image_info': {
                'filename': image.filename,
                'size_bytes': len(content),
                'dimensions': {'width': image_width, 'height': image_height}
            },
            'segments': result['segments'],
            'processing_time_seconds': processing_time,
            'model_info': {
                'model_type': sam_processor.model_type,
                'device': DEVICE,
                'total_segments_found': len(result['segments'])
            }
        }
        
        logger.info(f"Segmentation completed: {len(result['segments'])} segments "
                   f"in {processing_time:.2f}s")
        
        return response_data
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
    finally:
        # Clean up temporary file
        if temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_image_path}: {e}")


@app.post("/segment-batch")
async def segment_images_batch(
    images: List[UploadFile] = File(...),
    config: str = Form(...)
):
    """
    Segment multiple images in a batch.
    
    This processes multiple images sequentially (not concurrently)
    to avoid overwhelming system resources.
    """
    if sam_processor is None:
        raise HTTPException(
            status_code=503,
            detail="SAM model not loaded yet. Please wait and try again."
        )
    
    if len(images) > 10:  # Limit batch size
        raise HTTPException(
            status_code=422,
            detail="Batch size limited to 10 images"
        )
    
    try:
        config_data = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON configuration")
    
    results = []
    total_start_time = time.time()
    
    for i, image in enumerate(images):
        try:
            logger.info(f"Processing batch image {i+1}/{len(images)}: {image.filename}")
            
            # Process each image (reuse single image endpoint logic)
            temp_image_path = TEMP_DIR / f"batch_{int(time.time())}_{i}_{image.filename}"
            
            with open(temp_image_path, 'wb') as f:
                content = await image.read()
                f.write(content)
            
            result = await sam_processor.process_image(
                image_path=str(temp_image_path),
                min_area=config_data.get('min_area', 1000),
                max_segments=config_data.get('max_segments', 60),
                confidence_threshold=config_data.get('confidence_threshold', 0.7)
            )
            
            results.append({
                'filename': image.filename,
                'success': True,
                'segments': result['segments'],
                'segment_count': len(result['segments'])
            })
            
            # Clean up temp file
            temp_image_path.unlink()
            
        except Exception as e:
            logger.error(f"Failed to process batch image {image.filename}: {e}")
            results.append({
                'filename': image.filename,
                'success': False,
                'error': str(e)
            })
    
    total_processing_time = time.time() - total_start_time
    
    return {
        'success': True,
        'total_images': len(images),
        'successful_images': sum(1 for r in results if r['success']),
        'results': results,
        'total_processing_time_seconds': total_processing_time
    }


@app.get("/stats")
async def get_processing_stats():
    """
    Get processing statistics and performance metrics.
    
    This provides insights into service usage and performance
    for monitoring and optimization.
    """
    if sam_processor is None:
        return {"error": "SAM processor not initialized"}
    
    stats = sam_processor.get_stats()
    system_info = get_system_info()
    
    return {
        'processing_stats': stats,
        'system_info': system_info,
        'service_info': {
            'device': DEVICE,
            'model_path': MODEL_PATH,
            'max_image_size': MAX_IMAGE_SIZE,
            'temp_dir': str(TEMP_DIR)
        }
    }


@app.post("/clear-cache")
async def clear_cache():
    """
    Clear temporary files and reset statistics.
    
    This is useful for maintenance and testing.
    """
    try:
        # Clean up temporary files
        files_removed = cleanup_old_files(TEMP_DIR, max_age_hours=0)
        
        # Reset SAM processor statistics if available
        if sam_processor:
            sam_processor.reset_stats()
        
        logger.info(f"Cache cleared: {files_removed} files removed")
        
        return {
            'success': True,
            'files_removed': files_removed,
            'message': 'Cache cleared successfully'
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def periodic_cleanup():
    """
    Background task to periodically clean up old temporary files.
    
    This prevents the temporary directory from growing indefinitely
    by removing files older than a certain age.
    """
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            
            files_removed = cleanup_old_files(TEMP_DIR, max_age_hours=1)
            if files_removed > 0:
                logger.info(f"Periodic cleanup: removed {files_removed} old files")
                
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler with helpful message."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "available_endpoints": [
                "/health - Service health check",
                "/segment - Segment single image",
                "/segment-batch - Segment multiple images",
                "/stats - Get processing statistics",
                "/clear-cache - Clear temporary files",
                "/docs - API documentation"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler with error logging."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Please check the service logs for details"
        }
    )


# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configuration for running the service
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    workers = int(os.getenv('WORKERS', '1'))  # SAM is CPU/GPU intensive, usually 1 worker
    
    logger.info(f"Starting SAM service on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False  # Don't reload in production
    )