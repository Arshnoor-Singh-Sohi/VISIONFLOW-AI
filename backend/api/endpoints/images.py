"""
VisionFlow AI - Image Processing Endpoints
==========================================

This module handles all image-related API endpoints including:
- Image upload and validation
- Processing pipeline initiation
- Status monitoring and progress tracking
- Image metadata and history

These are the "core" endpoints that users interact with most frequently.
"""

import os
import uuid
import logging
import asyncio
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ...database import get_db
from ...config import get_settings
from ...models.database_models import ImageRecord, ProcessingStatus
from ...services.sam_service import get_sam_service
from ...services.openai_service import get_openai_service
from ...services.storage_service import get_storage_service
from ...utils.image_processing import validate_image_file, get_image_metadata
from ...utils.helpers import generate_unique_filename


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# =============================================================================

class ProcessingConfig(BaseModel):
    """Configuration for image processing pipeline."""
    min_area: int = 1000
    max_segments: int = 60
    confidence_threshold: float = 0.7
    classification_context: str = "food identification"
    enable_training: bool = True


class ImageUploadResponse(BaseModel):
    """Response model for image upload."""
    success: bool
    image_id: str
    filename: str
    message: str
    processing_status: str


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status."""
    image_id: str
    status: str
    progress_percentage: float
    current_step: str
    total_segments: Optional[int] = None
    completed_segments: Optional[int] = None
    error_message: Optional[str] = None
    estimated_completion_time: Optional[str] = None


class ImageListResponse(BaseModel):
    """Response model for image list."""
    images: List[dict]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


# =============================================================================
# IMAGE UPLOAD ENDPOINTS
# =============================================================================

@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    config: str = Form(default='{}'),
    user_id: Optional[str] = Form(default=None),
    db: Session = Depends(get_db)
):
    """
    Upload an image for processing.
    
    This endpoint handles file upload, validation, storage, and initiates
    the processing pipeline. The actual processing happens in the background
    so users get an immediate response.
    
    Args:
        image: The uploaded image file
        config: JSON string with processing configuration
        user_id: Optional user identifier
        db: Database session
        
    Returns:
        Upload confirmation with image ID and processing status
    """
    try:
        # Validate file type and size
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=422,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read file content
        file_content = await image.read()
        
        if len(file_content) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
            )
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=422,
                detail="Empty file uploaded"
            )
        
        # Parse processing configuration
        try:
            import json
            config_data = json.loads(config) if config else {}
            processing_config = ProcessingConfig(**config_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid configuration: {e}"
            )
        
        # Generate unique filename and save file
        file_extension = Path(image.filename).suffix.lower()
        if not file_extension:
            file_extension = '.jpg'  # Default extension
        
        unique_filename = generate_unique_filename(image.filename, file_extension)
        file_path = Path(settings.upload_path) / unique_filename
        
        # Save file to disk
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Get image metadata
        try:
            metadata = get_image_metadata(str(file_path))
        except Exception as e:
            logger.warning(f"Failed to get image metadata: {e}")
            metadata = {'width': 0, 'height': 0, 'channels': 3}
        
        # Create database record
        image_record = ImageRecord(
            filename=image.filename,
            file_path=str(file_path),
            file_size=len(file_content),
            mime_type=image.content_type,
            width=metadata.get('width'),
            height=metadata.get('height'),
            channels=metadata.get('channels'),
            status=ProcessingStatus.UPLOADED,
            user_id=user_id,
            processing_config=processing_config.dict()
        )
        
        db.add(image_record)
        db.commit()
        db.refresh(image_record)
        
        logger.info(f"Image uploaded successfully: {image_record.id} ({image.filename})")
        
        # Start background processing
        # background_tasks.add_task(
        #     process_image_pipeline,
        #     str(image_record.id),
        #     str(file_path),
        #     processing_config
        # )

        asyncio.create_task(
            process_image_pipeline(
                str(image_record.id),
                str(file_path),
                processing_config
            )
        )
        
        return ImageUploadResponse(
            success=True,
            image_id=str(image_record.id),
            filename=image.filename,
            message="Image uploaded successfully. Processing started.",
            processing_status=ProcessingStatus.UPLOADED.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Image upload failed. Please try again."
        )


@router.post("/process/{image_id}")
async def reprocess_image(
    image_id: str,
    background_tasks: BackgroundTasks,
    config: ProcessingConfig = ProcessingConfig(),
    db: Session = Depends(get_db)
):
    """
    Reprocess an existing image with new configuration.
    
    This allows users to re-run the processing pipeline on an
    already uploaded image with different parameters.
    """
    # Find the image record
    image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Check if file still exists
    if not os.path.exists(image_record.file_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    
    # Update processing configuration
    image_record.processing_config = config.dict()
    image_record.status = ProcessingStatus.UPLOADED
    image_record.error_message = None
    image_record.processing_started_at = None
    image_record.processing_completed_at = None
    
    db.commit()
    
    # Start background processing
    asyncio.create_task(
        process_image_pipeline(
            str(image_record.id),
            str(file_path),
            processing_config
        )
    )
    
    logger.info(f"Reprocessing initiated for image: {image_id}")
    
    return {
        "success": True,
        "image_id": image_id,
        "message": "Reprocessing started",
        "status": ProcessingStatus.UPLOADED.value
    }


# =============================================================================
# STATUS AND MONITORING ENDPOINTS
# =============================================================================

@router.get("/status/{image_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the current processing status of an image.
    
    This provides real-time status updates including progress percentage,
    current processing step, and estimated completion time.
    """
    image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Calculate progress percentage based on status
    progress_map = {
        ProcessingStatus.UPLOADED: 0,
        ProcessingStatus.SEGMENTING: 25,
        ProcessingStatus.CLASSIFYING: 60,
        ProcessingStatus.TRAINING: 80,
        ProcessingStatus.COMPLETED: 100,
        ProcessingStatus.FAILED: 0,
        ProcessingStatus.CANCELLED: 0
    }
    
    progress = progress_map.get(image_record.status, 0)
    
    # Get step description
    step_descriptions = {
        ProcessingStatus.UPLOADED: "Waiting to start processing",
        ProcessingStatus.SEGMENTING: "Segmenting image with SAM",
        ProcessingStatus.CLASSIFYING: "Classifying segments with OpenAI",
        ProcessingStatus.TRAINING: "Adding to training dataset",
        ProcessingStatus.COMPLETED: "Processing completed",
        ProcessingStatus.FAILED: "Processing failed",
        ProcessingStatus.CANCELLED: "Processing cancelled"
    }
    
    current_step = step_descriptions.get(image_record.status, "Unknown")
    
    # Calculate estimated completion time
    estimated_completion = None
    if image_record.processing_started_at and image_record.status != ProcessingStatus.COMPLETED:
        from datetime import timedelta
        elapsed = datetime.now(timezone.utc) - image_record.processing_started_at
        if progress > 0:
            total_estimated = elapsed * (100 / progress)
            remaining = total_estimated - elapsed
            estimated_completion = (datetime.now(timezone.utc) + remaining).isoformat()
    
    # Get segment information if available
    total_segments = None
    completed_segments = None
    
    if image_record.segments:
        total_segments = len(image_record.segments)
        completed_segments = len([s for s in image_record.segments if s.classifications])
    
    return ProcessingStatusResponse(
        image_id=image_id,
        status=image_record.status.value,
        progress_percentage=progress,
        current_step=current_step,
        total_segments=total_segments,
        completed_segments=completed_segments,
        error_message=image_record.error_message,
        estimated_completion_time=estimated_completion
    )


@router.get("/list", response_model=ImageListResponse)
async def list_images(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List uploaded images with pagination and filtering.
    
    This provides a paginated list of images with optional filtering
    by processing status and user ID.
    """
    # Build query
    query = db.query(ImageRecord)
    
    if status:
        try:
            status_enum = ProcessingStatus(status)
            query = query.filter(ImageRecord.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid status: {status}")
    
    if user_id:
        query = query.filter(ImageRecord.user_id == user_id)
    
    # Get total count
    total_count = query.count()
    
    # Apply pagination
    offset = (page - 1) * page_size
    images = query.order_by(ImageRecord.created_at.desc()).offset(offset).limit(page_size).all()
    
    # Format response
    image_list = []
    for image in images:
        image_data = {
            "id": str(image.id),
            "filename": image.filename,
            "status": image.status.value,
            "created_at": image.created_at.isoformat(),
            "file_size": image.file_size,
            "width": image.width,
            "height": image.height,
            "processing_started_at": image.processing_started_at.isoformat() if image.processing_started_at else None,
            "processing_completed_at": image.processing_completed_at.isoformat() if image.processing_completed_at else None,
            "segment_count": len(image.segments) if image.segments else 0,
            "classification_count": len(image.classifications) if image.classifications else 0
        }
        image_list.append(image_data)
    
    # Calculate pagination info
    has_next = (offset + page_size) < total_count
    has_previous = page > 1
    
    return ImageListResponse(
        images=image_list,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next=has_next,
        has_previous=has_previous
    )


# =============================================================================
# FILE ACCESS ENDPOINTS
# =============================================================================

@router.get("/download/{image_id}")
async def download_image(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Download the original uploaded image file.
    
    This provides access to the original image file for viewing or downloading.
    """
    image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not os.path.exists(image_record.file_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    
    return FileResponse(
        path=image_record.file_path,
        filename=image_record.filename,
        media_type=image_record.mime_type
    )


@router.get("/thumbnail/{image_id}")
async def get_thumbnail(
    image_id: str,
    size: int = 300,
    db: Session = Depends(get_db)
):
    """
    Get a thumbnail version of the image.
    
    This creates and returns a resized version of the image for
    quick preview in the frontend.
    """
    image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not os.path.exists(image_record.file_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    
    try:
        # Generate thumbnail using storage service
        storage_service = get_storage_service()
        thumbnail_path = await storage_service.create_thumbnail(
            image_record.file_path,
            size=size
        )
        
        return FileResponse(
            path=thumbnail_path,
            media_type="image/jpeg"
        )
        
    except Exception as e:
        logger.error(f"Failed to create thumbnail for {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create thumbnail")


# =============================================================================
# MANAGEMENT ENDPOINTS
# =============================================================================

@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete an image and all associated data.
    
    This removes the image record, file, and all processing results
    from the system. Use with caution!
    """
    image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Delete file from disk
        if os.path.exists(image_record.file_path):
            os.remove(image_record.file_path)
        
        # Delete associated segment files
        storage_service = get_storage_service()
        await storage_service.cleanup_image_files(image_id)
        
        # Delete database record (cascades to segments, classifications, etc.)
        db.delete(image_record)
        db.commit()
        
        logger.info(f"Image deleted: {image_id}")
        
        return {
            "success": True,
            "message": f"Image {image_id} deleted successfully"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete image {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete image")


@router.post("/{image_id}/cancel")
async def cancel_processing(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Cancel ongoing image processing.
    
    This stops the processing pipeline for an image if it's currently
    in progress. Useful if processing is taking too long or was started
    with wrong parameters.
    """
    image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if image_record.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot cancel processing. Current status: {image_record.status.value}"
        )
    
    # Update status to cancelled
    image_record.status = ProcessingStatus.CANCELLED
    image_record.processing_completed_at = datetime.now(timezone.utc)
    
    db.commit()
    
    logger.info(f"Processing cancelled for image: {image_id}")
    
    return {
        "success": True,
        "message": f"Processing cancelled for image {image_id}",
        "status": ProcessingStatus.CANCELLED.value
    }


# =============================================================================
# BACKGROUND PROCESSING PIPELINE
# =============================================================================

async def process_image_pipeline(
    image_id: str,
    file_path: str,
    config: ProcessingConfig
):
    logger.info(f"BACKGROUND PIPELINE STARTED for image: {image_id}")
    """
    Main image processing pipeline that runs in the background.
    
    This orchestrates the entire processing workflow:
    1. SAM segmentation
    2. OpenAI classification
    3. Training data creation
    4. Result storage
    
    This function runs asynchronously in the background so users
    get immediate feedback while processing continues.
    """
    from ...database import db_manager
    from ...main import broadcast_processing_update
    
    logger.info(f"Starting processing pipeline for image: {image_id}")
    
    try:
        # Get database session
        with db_manager.get_session_context() as db:
            # Update status to segmenting
            image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
            if not image_record:
                logger.error(f"Image record not found: {image_id}")
                return
            
            image_record.status = ProcessingStatus.SEGMENTING
            image_record.processing_started_at = datetime.now(timezone.utc)
            db.commit()
            
            # Broadcast status update
            await broadcast_processing_update(image_id, "segmenting", {
                "step": "Starting SAM segmentation"
            })
            
            # Step 1: SAM Segmentation
            sam_service = get_sam_service()
            sam_result = await sam_service.segment_image(
                file_path,
                min_area=config.min_area,
                max_segments=config.max_segments,
                confidence_threshold=config.confidence_threshold
            )
            
            logger.info(f"SAM segmentation completed: {len(sam_result.segments)} segments")
            
            # Store segments in database
            from ...models.database_models import ImageSegment
            
            for i, segment in enumerate(sam_result.segments):
                segment_record = ImageSegment(
                    image_id=image_record.id,
                    segment_index=i,
                    bbox_x=segment.bbox[0],
                    bbox_y=segment.bbox[1],
                    bbox_width=segment.bbox[2],
                    bbox_height=segment.bbox[3],
                    area=segment.area,
                    confidence_score=segment.confidence_score,
                    segment_path=segment.segment_image_path
                )
                db.add(segment_record)
            
            db.commit()
            
            # Step 2: OpenAI Classification
            image_record.status = ProcessingStatus.CLASSIFYING
            db.commit()
            
            await broadcast_processing_update(image_id, "classifying", {
                "step": "Starting OpenAI classification",
                "total_segments": len(sam_result.segments)
            })
            
            openai_service = get_openai_service()
            segment_paths = [s.segment_image_path for s in sam_result.segments if s.segment_image_path]
            
            classification_results = await openai_service.batch_classify_segments(
                segment_paths,
                context=config.classification_context
            )
            
            logger.info(f"OpenAI classification completed: {len(classification_results)} results")
            
            # Store classifications in database
            from ...models.database_models import Classification
            
            for i, classification in enumerate(classification_results):
                if i < len(image_record.segments):
                    segment = image_record.segments[i]
                    
                    classification_record = Classification(
                        image_id=image_record.id,
                        segment_id=segment.id,
                        primary_label=classification.primary_label,
                        confidence_score=classification.confidence_score,
                        alternative_labels=classification.alternative_labels,
                        raw_response=classification.raw_response,
                        model_used=classification.model_used,
                        tokens_used=classification.tokens_used
                    )
                    db.add(classification_record)
            
            db.commit()
            
            # Step 3: Training Data Creation (if enabled)
            if config.enable_training:
                image_record.status = ProcessingStatus.TRAINING
                db.commit()
                
                await broadcast_processing_update(image_id, "training", {
                    "step": "Creating training samples"
                })
                
                # Create training samples
                from ...models.database_models import TrainingSample
                
                for classification in image_record.classifications:
                    training_sample = TrainingSample(
                        image_id=image_record.id,
                        segment_id=classification.segment_id,
                        ground_truth_label=classification.primary_label,
                        label_source='openai'
                    )
                    db.add(training_sample)
                
                db.commit()
            
            # Step 4: Mark as completed
            image_record.status = ProcessingStatus.COMPLETED
            image_record.processing_completed_at = datetime.now(timezone.utc)
            db.commit()
            
            processing_time = (
                image_record.processing_completed_at - image_record.processing_started_at
            ).total_seconds()
            
            logger.info(f"Processing completed for image {image_id} in {processing_time:.2f}s")
            
            await broadcast_processing_update(image_id, "completed", {
                "step": "Processing completed",
                "processing_time_seconds": processing_time,
                "total_segments": len(image_record.segments),
                "total_classifications": len(image_record.classifications)
            })
            
    except Exception as e:
        logger.error(f"Processing failed for image {image_id}: {e}")
        
        # Update status to failed
        try:
            with db_manager.get_session_context() as db:
                image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
                if image_record:
                    image_record.status = ProcessingStatus.FAILED
                    image_record.error_message = str(e)
                    image_record.processing_completed_at = datetime.now(timezone.utc)
                    db.commit()
                    
                    await broadcast_processing_update(image_id, "failed", {
                        "step": "Processing failed",
                        "error": str(e)
                    })
        except Exception as db_error:
            logger.error(f"Failed to update error status: {db_error}")