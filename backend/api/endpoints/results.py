"""
VisionFlow AI - Results Viewing Endpoints
=========================================

This module handles endpoints for viewing and managing processing results:
- Viewing segmentation and classification results
- Downloading processed images with annotations
- Exporting results in various formats
- Human feedback and corrections
- Result analytics and statistics

These endpoints allow users to explore and interact with the AI processing results.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from pydantic import BaseModel
import enum

from ...database import get_db
from ...config import get_settings
from ...models.database_models import (
    ImageRecord, ImageSegment, Classification, ProcessingStatus
)
from ...services.storage_service import get_storage_service
from ...utils.image_processing import create_annotated_image


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SegmentInfo(BaseModel):
    """Information about a single image segment."""
    id: str
    bbox: List[int]  # [x, y, width, height]
    area: int
    confidence_score: float
    segment_image_url: Optional[str]
    classification: Optional[Dict[str, Any]]


class ImageResultSummary(BaseModel):
    """Summary of processing results for an image."""
    image_id: str
    filename: str
    status: str
    processing_time_seconds: Optional[float]
    total_segments: int
    total_classifications: int
    top_classifications: List[Dict[str, Any]]
    created_at: str
    completed_at: Optional[str]


class DetailedImageResult(BaseModel):
    """Detailed processing results for an image."""
    image_id: str
    filename: str
    status: str
    image_url: str
    thumbnail_url: str
    annotated_image_url: Optional[str]
    processing_config: Dict[str, Any]
    processing_time_seconds: Optional[float]
    segments: List[SegmentInfo]
    statistics: Dict[str, Any]
    created_at: str
    completed_at: Optional[str]


class FeedbackRequest(BaseModel):
    """Request model for human feedback on classifications."""
    classification_id: str
    correct_label: str
    confidence: float
    notes: Optional[str] = None

class ExportFormat(str, enum.Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    COCO = "coco"
    YOLO = "yolo"


# =============================================================================
# RESULT VIEWING ENDPOINTS
# =============================================================================

@router.get("/summary/{image_id}", response_model=ImageResultSummary)
async def get_result_summary(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a summary of processing results for an image.
    
    This provides a quick overview of the processing results without
    all the detailed segment information.
    """
    # Get image record with related data
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Calculate processing time
    processing_time = None
    if image.processing_started_at and image.processing_completed_at:
        processing_time = (
            image.processing_completed_at - image.processing_started_at
        ).total_seconds()
    
    # Get top classifications (most common labels)
    top_classifications_query = db.query(
        Classification.primary_label,
        func.count(Classification.id).label('count'),
        func.avg(Classification.confidence_score).label('avg_confidence')
    ).filter(
        Classification.image_id == image_id
    ).group_by(
        Classification.primary_label
    ).order_by(
        desc('count')
    ).limit(5)
    
    top_classifications = [
        {
            'label': result.primary_label,
            'count': result.count,
            'average_confidence': float(result.avg_confidence) if result.avg_confidence else 0.0
        }
        for result in top_classifications_query.all()
    ]
    
    return ImageResultSummary(
        image_id=str(image.id),
        filename=image.filename,
        status=image.status.value,
        processing_time_seconds=processing_time,
        total_segments=len(image.segments) if image.segments else 0,
        total_classifications=len(image.classifications) if image.classifications else 0,
        top_classifications=top_classifications,
        created_at=image.created_at.isoformat(),
        completed_at=image.processing_completed_at.isoformat() if image.processing_completed_at else None
    )


@router.get("/detailed/{image_id}", response_model=DetailedImageResult)
async def get_detailed_results(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed processing results for an image.
    
    This includes all segments, classifications, and generated URLs
    for viewing the results in the frontend.
    """
    # Get image record with all related data
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if image.status not in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
        raise HTTPException(
            status_code=422,
            detail=f"Results not available. Current status: {image.status.value}"
        )
    
    # Calculate processing time
    processing_time = None
    if image.processing_started_at and image.processing_completed_at:
        processing_time = (
            image.processing_completed_at - image.processing_started_at
        ).total_seconds()
    
    # Build segment information
    segments = []
    for segment in image.segments:
        # Find classification for this segment
        classification = None
        for cls in image.classifications:
            if cls.segment_id == segment.id:
                classification = {
                    'id': str(cls.id),
                    'primary_label': cls.primary_label,
                    'confidence_score': cls.confidence_score,
                    'alternative_labels': cls.alternative_labels,
                    'model_used': cls.model_used,
                    'human_verified': cls.human_verified,
                    'human_label': cls.human_label
                }
                break
        
        segment_info = SegmentInfo(
            id=str(segment.id),
            bbox=[segment.bbox_x, segment.bbox_y, segment.bbox_width, segment.bbox_height],
            area=segment.area,
            confidence_score=segment.confidence_score,
            segment_image_url=f"/api/v1/results/segment-image/{segment.id}" if segment.segment_path else None,
            classification=classification
        )
        segments.append(segment_info)
    
    # Calculate statistics
    if image.classifications:
        labels = [cls.primary_label for cls in image.classifications]
        avg_confidence = sum(cls.confidence_score for cls in image.classifications) / len(image.classifications)
        unique_labels = len(set(labels))
    else:
        avg_confidence = 0.0
        unique_labels = 0
    
    statistics = {
        'total_segments': len(image.segments) if image.segments else 0,
        'total_classifications': len(image.classifications) if image.classifications else 0,
        'unique_labels': unique_labels,
        'average_confidence': avg_confidence,
        'processing_time_seconds': processing_time
    }
    
    return DetailedImageResult(
        image_id=str(image.id),
        filename=image.filename,
        status=image.status.value,
        image_url=f"/api/v1/images/download/{image.id}",
        thumbnail_url=f"/api/v1/images/thumbnail/{image.id}",
        annotated_image_url=f"/api/v1/results/annotated/{image.id}",
        processing_config=image.processing_config or {},
        processing_time_seconds=processing_time,
        segments=segments,
        statistics=statistics,
        created_at=image.created_at.isoformat(),
        completed_at=image.processing_completed_at.isoformat() if image.processing_completed_at else None
    )


# =============================================================================
# IMAGE ACCESS ENDPOINTS
# =============================================================================

@router.get("/segment-image/{segment_id}")
async def get_segment_image(
    segment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the cropped image for a specific segment.
    
    This returns the individual segment image that was sent to OpenAI
    for classification.
    """
    segment = db.query(ImageSegment).filter(ImageSegment.id == segment_id).first()
    
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    
    if not segment.segment_path:
        raise HTTPException(status_code=404, detail="Segment image not available")
    
    import os
    if not os.path.exists(segment.segment_path):
        raise HTTPException(status_code=404, detail="Segment image file not found")
    
    return FileResponse(
        path=segment.segment_path,
        media_type="image/jpeg",
        filename=f"segment_{segment.segment_index}.jpg"
    )


@router.get("/annotated/{image_id}")
async def get_annotated_image(
    image_id: str,
    show_labels: bool = Query(True, description="Show classification labels"),
    show_confidence: bool = Query(True, description="Show confidence scores"),
    show_bbox: bool = Query(True, description="Show bounding boxes"),
    db: Session = Depends(get_db)
):
    """
    Get an annotated version of the image with bounding boxes and labels.
    
    This creates a visual representation of the processing results
    overlaid on the original image.
    """
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not image.segments:
        raise HTTPException(status_code=404, detail="No segments available for annotation")
    
    try:
        # Create annotated image
        storage_service = get_storage_service()
        annotated_path = await storage_service.create_annotated_image(
            image_id=image_id,
            show_labels=show_labels,
            show_confidence=show_confidence,
            show_bbox=show_bbox
        )
        
        return FileResponse(
            path=annotated_path,
            media_type="image/jpeg",
            filename=f"{image.filename}_annotated.jpg"
        )
        
    except Exception as e:
        logger.error(f"Failed to create annotated image for {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create annotated image")


# =============================================================================
# ANALYTICS AND STATISTICS ENDPOINTS
# =============================================================================

@router.get("/analytics/overview")
async def get_analytics_overview(
    days: int = Query(30, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get analytics overview for the specified time period.
    
    This provides high-level statistics about processing performance,
    popular classifications, and system usage.
    """
    from datetime import timedelta
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    # Total images processed
    total_images = db.query(ImageRecord).filter(
        ImageRecord.created_at >= start_date
    ).count()
    
    # Success rate
    successful_images = db.query(ImageRecord).filter(
        ImageRecord.created_at >= start_date,
        ImageRecord.status == ProcessingStatus.COMPLETED
    ).count()
    
    success_rate = (successful_images / total_images * 100) if total_images > 0 else 0
    
    # Average processing time
    avg_processing_time_query = db.query(
        func.avg(
            func.extract('epoch', ImageRecord.processing_completed_at - ImageRecord.processing_started_at)
        ).label('avg_time')
    ).filter(
        ImageRecord.created_at >= start_date,
        ImageRecord.status == ProcessingStatus.COMPLETED,
        ImageRecord.processing_started_at.isnot(None),
        ImageRecord.processing_completed_at.isnot(None)
    ).first()
    
    avg_processing_time = float(avg_processing_time_query.avg_time) if avg_processing_time_query.avg_time else 0.0
    
    # Most common classifications
    top_labels_query = db.query(
        Classification.primary_label,
        func.count(Classification.id).label('count')
    ).join(ImageRecord).filter(
        ImageRecord.created_at >= start_date
    ).group_by(
        Classification.primary_label
    ).order_by(
        desc('count')
    ).limit(10)
    
    top_labels = [
        {'label': result.primary_label, 'count': result.count}
        for result in top_labels_query.all()
    ]
    
    # Daily processing counts
    daily_counts_query = db.query(
        func.date(ImageRecord.created_at).label('date'),
        func.count(ImageRecord.id).label('count')
    ).filter(
        ImageRecord.created_at >= start_date
    ).group_by(
        func.date(ImageRecord.created_at)
    ).order_by('date')
    
    daily_counts = [
        {'date': result.date.isoformat(), 'count': result.count}
        for result in daily_counts_query.all()
    ]
    
    return {
        'period': {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'days': days
        },
        'summary': {
            'total_images': total_images,
            'successful_images': successful_images,
            'success_rate_percent': round(success_rate, 2),
            'average_processing_time_seconds': round(avg_processing_time, 2)
        },
        'top_classifications': top_labels,
        'daily_processing_counts': daily_counts
    }


@router.get("/analytics/performance")
async def get_performance_analytics(
    db: Session = Depends(get_db)
):
    """
    Get detailed performance analytics.
    
    This provides insights into processing performance, bottlenecks,
    and system efficiency metrics.
    """
    # Processing time statistics
    processing_times_query = db.query(
        func.extract('epoch', ImageRecord.processing_completed_at - ImageRecord.processing_started_at).label('duration')
    ).filter(
        ImageRecord.status == ProcessingStatus.COMPLETED,
        ImageRecord.processing_started_at.isnot(None),
        ImageRecord.processing_completed_at.isnot(None)
    )
    
    processing_times = [float(result.duration) for result in processing_times_query.all()]
    
    if processing_times:
        import statistics
        performance_stats = {
            'count': len(processing_times),
            'mean_seconds': statistics.mean(processing_times),
            'median_seconds': statistics.median(processing_times),
            'min_seconds': min(processing_times),
            'max_seconds': max(processing_times),
            'std_dev_seconds': statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0
        }
    else:
        performance_stats = {
            'count': 0,
            'mean_seconds': 0.0,
            'median_seconds': 0.0,
            'min_seconds': 0.0,
            'max_seconds': 0.0,
            'std_dev_seconds': 0.0
        }
    
    # Error rate by status
    status_counts = db.query(
        ImageRecord.status,
        func.count(ImageRecord.id).label('count')
    ).group_by(ImageRecord.status).all()
    
    status_distribution = {
        result.status.value: result.count
        for result in status_counts
    }
    
    # Segment and classification statistics
    segment_stats_query = db.query(
        func.avg(func.count(ImageSegment.id)).label('avg_segments'),
        func.max(func.count(ImageSegment.id)).label('max_segments'),
        func.min(func.count(ImageSegment.id)).label('min_segments')
    ).join(ImageRecord).filter(
        ImageRecord.status == ProcessingStatus.COMPLETED
    ).group_by(ImageSegment.image_id).first()
    
    segment_stats = {
        'average_segments_per_image': float(segment_stats_query.avg_segments) if segment_stats_query and segment_stats_query.avg_segments else 0.0,
        'max_segments_per_image': int(segment_stats_query.max_segments) if segment_stats_query and segment_stats_query.max_segments else 0,
        'min_segments_per_image': int(segment_stats_query.min_segments) if segment_stats_query and segment_stats_query.min_segments else 0
    }
    
    return {
        'processing_time_statistics': performance_stats,
        'status_distribution': status_distribution,
        'segment_statistics': segment_stats,
        'generated_at': datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# HUMAN FEEDBACK ENDPOINTS
# =============================================================================

@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit human feedback on a classification result.
    
    This allows users to correct AI classifications, which can be used
    to improve the model and track accuracy over time.
    """
    # Find the classification
    classification = db.query(Classification).filter(
        Classification.id == feedback.classification_id
    ).first()
    
    if not classification:
        raise HTTPException(status_code=404, detail="Classification not found")
    
    # Update with human feedback
    classification.human_verified = True
    classification.human_label = feedback.correct_label
    classification.human_feedback_notes = feedback.notes
    
    # Update the corresponding training sample if it exists
    from ...models.database_models import TrainingSample
    training_sample = db.query(TrainingSample).filter(
        TrainingSample.image_id == classification.image_id,
        TrainingSample.segment_id == classification.segment_id
    ).first()
    
    if training_sample:
        training_sample.ground_truth_label = feedback.correct_label
        training_sample.label_source = 'human'
    else:
        # Create new training sample with human label
        training_sample = TrainingSample(
            image_id=classification.image_id,
            segment_id=classification.segment_id,
            ground_truth_label=feedback.correct_label,
            label_source='human'
        )
        db.add(training_sample)
    
    db.commit()
    
    logger.info(f"Human feedback submitted for classification {feedback.classification_id}")
    
    return {
        "success": True,
        "message": "Feedback submitted successfully",
        "classification_id": feedback.classification_id,
        "corrected_label": feedback.correct_label
    }


@router.get("/feedback/accuracy")
async def get_accuracy_metrics(
    db: Session = Depends(get_db)
):
    """
    Get accuracy metrics based on human feedback.
    
    This analyzes how often the AI classifications match human corrections
    to provide insights into model performance.
    """
    # Total human-verified classifications
    total_verified = db.query(Classification).filter(
        Classification.human_verified == True
    ).count()
    
    if total_verified == 0:
        return {
            'total_verified_classifications': 0,
            'accuracy_rate': 0.0,
            'message': 'No human feedback available yet'
        }
    
    # Correct classifications (AI label matches human label)
    correct_classifications = db.query(Classification).filter(
        Classification.human_verified == True,
        Classification.primary_label == Classification.human_label
    ).count()
    
    accuracy_rate = (correct_classifications / total_verified) * 100
    
    # Most commonly corrected labels
    correction_query = db.query(
        Classification.primary_label.label('ai_label'),
        Classification.human_label.label('human_label'),
        func.count(Classification.id).label('count')
    ).filter(
        Classification.human_verified == True,
        Classification.primary_label != Classification.human_label
    ).group_by(
        Classification.primary_label,
        Classification.human_label
    ).order_by(
        desc('count')
    ).limit(10)
    
    common_corrections = [
        {
            'ai_label': result.ai_label,
            'human_label': result.human_label,
            'count': result.count
        }
        for result in correction_query.all()
    ]
    
    return {
        'total_verified_classifications': total_verified,
        'correct_classifications': correct_classifications,
        'accuracy_rate_percent': round(accuracy_rate, 2),
        'common_corrections': common_corrections
    }


# =============================================================================
# EXPORT ENDPOINTS
# =============================================================================

@router.get("/export/{image_id}")
async def export_results(
    image_id: str,
    format: ExportFormat = Query(ExportFormat.JSON, description="Export format"),
    db: Session = Depends(get_db)
):
    """
    Export processing results in various formats.
    
    This allows users to download results in different formats for
    external analysis or integration with other tools.
    """
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if image.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot export incomplete results. Status: {image.status.value}"
        )
    
    try:
        storage_service = get_storage_service()
        
        if format == ExportFormat.JSON:
            export_path = await storage_service.export_to_json(image_id)
            media_type = "application/json"
            filename = f"{image.filename}_results.json"
        
        elif format == ExportFormat.CSV:
            export_path = await storage_service.export_to_csv(image_id)
            media_type = "text/csv"
            filename = f"{image.filename}_results.csv"
        
        elif format == ExportFormat.COCO:
            export_path = await storage_service.export_to_coco(image_id)
            media_type = "application/json"
            filename = f"{image.filename}_coco.json"
        
        elif format == ExportFormat.YOLO:
            export_path = await storage_service.export_to_yolo(image_id)
            media_type = "text/plain"
            filename = f"{image.filename}_yolo.txt"
        
        else:
            raise HTTPException(status_code=422, detail=f"Unsupported export format: {format}")
        
        return FileResponse(
            path=export_path,
            media_type=media_type,
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Failed to export results for {image_id} in {format} format: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/export/batch")
async def export_batch_results(
    image_ids: List[str] = Query(..., description="List of image IDs to export"),
    format: ExportFormat = Query(ExportFormat.JSON, description="Export format"),
    db: Session = Depends(get_db)
):
    """
    Export results for multiple images in a single file.
    
    This is useful for bulk analysis or when working with datasets.
    """
    if len(image_ids) > 100:  # Limit batch size
        raise HTTPException(
            status_code=422,
            detail="Batch export limited to 100 images"
        )
    
    # Validate all images exist and are completed
    images = db.query(ImageRecord).filter(
        ImageRecord.id.in_(image_ids),
        ImageRecord.status == ProcessingStatus.COMPLETED
    ).all()
    
    if len(images) != len(image_ids):
        missing_ids = set(image_ids) - {str(img.id) for img in images}
        raise HTTPException(
            status_code=404,
            detail=f"Images not found or not completed: {list(missing_ids)}"
        )
    
    try:
        storage_service = get_storage_service()
        
        if format == ExportFormat.JSON:
            export_path = await storage_service.export_batch_to_json(image_ids)
            media_type = "application/json"
            filename = f"batch_results_{len(image_ids)}_images.json"
        
        elif format == ExportFormat.CSV:
            export_path = await storage_service.export_batch_to_csv(image_ids)
            media_type = "text/csv"
            filename = f"batch_results_{len(image_ids)}_images.csv"
        
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Batch export not supported for format: {format}"
            )
        
        return FileResponse(
            path=export_path,
            media_type=media_type,
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Failed to export batch results: {e}")
        raise HTTPException(status_code=500, detail=f"Batch export failed: {str(e)}")