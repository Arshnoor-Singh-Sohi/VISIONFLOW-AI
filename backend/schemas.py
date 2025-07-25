"""
VisionFlow AI - Pydantic Schemas
===============================

This module defines Pydantic models for request/response validation and serialization.
These schemas ensure data integrity throughout the API and provide automatic validation,
documentation, and type conversion.

Think of schemas as "contracts" that define exactly what data should look like
when it comes into or goes out of our API endpoints.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4


# =============================================================================
# ENUMS FOR CONSISTENT VALUES
# =============================================================================

class ProcessingStatusEnum(str, Enum):
    """Processing status values for image pipeline."""
    UPLOADED = "uploaded"
    SEGMENTING = "segmenting"
    CLASSIFYING = "classifying"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingStatusEnum(str, Enum):
    """Training status values for model training."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class ExportFormatEnum(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    COCO = "coco"
    YOLO = "yolo"


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class TimestampMixin(BaseModel):
    """Mixin for models that include timestamps."""
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        # Allow datetime serialization
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ResponseBase(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# IMAGE PROCESSING SCHEMAS
# =============================================================================

class ProcessingConfigSchema(BaseModel):
    """Schema for image processing configuration."""
    min_area: int = Field(default=1000, ge=100, le=50000, description="Minimum segment area in pixels")
    max_segments: int = Field(default=60, ge=1, le=200, description="Maximum number of segments to process")
    confidence_threshold: float = Field(default=0.7, ge=0.1, le=0.9, description="Minimum confidence threshold")
    classification_context: str = Field(default="food identification", description="Context for AI classification")
    enable_training: bool = Field(default=True, description="Whether to use results for training")

    @validator('classification_context')
    def validate_context(cls, v):
        allowed_contexts = ['food identification', 'general objects', 'kitchen items']
        if v not in allowed_contexts:
            raise ValueError(f'Context must be one of: {allowed_contexts}')
        return v


class ImageUploadRequest(BaseModel):
    """Schema for image upload request metadata."""
    config: ProcessingConfigSchema = Field(default_factory=ProcessingConfigSchema)
    user_id: Optional[str] = Field(None, max_length=100, description="User identifier")


class ImageUploadResponse(ResponseBase):
    """Schema for image upload response."""
    image_id: UUID4
    filename: str
    processing_status: ProcessingStatusEnum


class ProcessingStatusResponse(BaseModel):
    """Schema for processing status response."""
    image_id: UUID4
    status: ProcessingStatusEnum
    progress_percentage: float = Field(ge=0, le=100)
    current_step: str
    total_segments: Optional[int] = None
    completed_segments: Optional[int] = None
    error_message: Optional[str] = None
    estimated_completion_time: Optional[datetime] = None


class ImageListRequest(BaseModel):
    """Schema for image list request parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    status: Optional[ProcessingStatusEnum] = None
    user_id: Optional[str] = None


class ImageSummarySchema(BaseModel):
    """Schema for image summary information."""
    id: UUID4
    filename: str
    status: ProcessingStatusEnum
    created_at: datetime
    file_size: int
    width: Optional[int] = None
    height: Optional[int] = None
    segment_count: int = 0
    classification_count: int = 0
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None


class ImageListResponse(ResponseBase):
    """Schema for image list response."""
    images: List[ImageSummarySchema]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


# =============================================================================
# SEGMENTATION AND CLASSIFICATION SCHEMAS
# =============================================================================

class BoundingBoxSchema(BaseModel):
    """Schema for bounding box coordinates."""
    x: int = Field(ge=0, description="X coordinate")
    y: int = Field(ge=0, description="Y coordinate")
    width: int = Field(gt=0, description="Width in pixels")
    height: int = Field(gt=0, description="Height in pixels")


class ClassificationSchema(BaseModel):
    """Schema for classification results."""
    id: UUID4
    primary_label: str
    confidence_score: float = Field(ge=0, le=1)
    alternative_labels: Optional[List[Dict[str, float]]] = None
    model_used: str
    tokens_used: Optional[int] = None
    human_verified: bool = False
    human_label: Optional[str] = None
    human_feedback_notes: Optional[str] = None


class SegmentSchema(BaseModel):
    """Schema for image segment information."""
    id: UUID4
    segment_index: int
    bbox: BoundingBoxSchema
    area: int = Field(gt=0, description="Segment area in pixels")
    confidence_score: float = Field(ge=0, le=1)
    segment_image_url: Optional[str] = None
    classification: Optional[ClassificationSchema] = None


class ResultSummaryResponse(BaseModel):
    """Schema for result summary response."""
    image_id: UUID4
    filename: str
    status: ProcessingStatusEnum
    processing_time_seconds: Optional[float] = None
    total_segments: int
    total_classifications: int
    top_classifications: List[Dict[str, Any]]
    created_at: datetime
    completed_at: Optional[datetime] = None


class ProcessingStatisticsSchema(BaseModel):
    """Schema for processing statistics."""
    total_segments: int
    total_classifications: int
    unique_labels: int
    average_confidence: float = Field(ge=0, le=1)
    processing_time_seconds: Optional[float] = None


class DetailedResultsResponse(BaseModel):
    """Schema for detailed results response."""
    image_id: UUID4
    filename: str
    status: ProcessingStatusEnum
    image_url: str
    thumbnail_url: str
    annotated_image_url: Optional[str] = None
    processing_config: Dict[str, Any]
    processing_time_seconds: Optional[float] = None
    segments: List[SegmentSchema]
    statistics: ProcessingStatisticsSchema
    created_at: datetime
    completed_at: Optional[datetime] = None


# =============================================================================
# FEEDBACK SCHEMAS
# =============================================================================

class FeedbackRequest(BaseModel):
    """Schema for human feedback submission."""
    classification_id: UUID4
    correct_label: str = Field(min_length=1, max_length=100, description="Correct classification label")
    confidence: float = Field(ge=0, le=1, description="Confidence in the correction")
    notes: Optional[str] = Field(None, max_length=500, description="Additional feedback notes")

    @validator('correct_label')
    def validate_label(cls, v):
        # Clean and validate the label
        v = v.strip()
        if not v:
            raise ValueError('Label cannot be empty')
        return v


class FeedbackResponse(ResponseBase):
    """Schema for feedback submission response."""
    classification_id: UUID4
    corrected_label: str


class AccuracyMetricsResponse(BaseModel):
    """Schema for accuracy metrics response."""
    total_verified_classifications: int
    correct_classifications: int
    accuracy_rate_percent: float
    common_corrections: List[Dict[str, Any]]


# =============================================================================
# TRAINING SCHEMAS
# =============================================================================

class TrainingConfigSchema(BaseModel):
    """Schema for training configuration."""
    run_name: str = Field(min_length=1, max_length=100, description="Name for this training run")
    model_type: str = Field(default="image_classifier", description="Type of model to train")
    batch_size: int = Field(default=32, ge=1, le=256, description="Training batch size")
    learning_rate: float = Field(default=0.001, gt=0, le=1, description="Learning rate")
    num_epochs: int = Field(default=10, ge=1, le=1000, description="Number of training epochs")
    train_test_split: float = Field(default=0.8, gt=0, lt=1, description="Train/test split ratio")
    min_samples_per_class: int = Field(default=5, ge=1, description="Minimum samples per class")
    use_human_labels_only: bool = Field(default=False, description="Use only human-verified labels")
    augmentation_enabled: bool = Field(default=True, description="Enable data augmentation")
    early_stopping_patience: int = Field(default=3, ge=1, description="Early stopping patience")


class TrainingRunSummarySchema(BaseModel):
    """Schema for training run summary."""
    id: UUID4
    run_name: str
    status: TrainingStatusEnum
    model_type: str
    num_samples: int
    current_epoch: int
    total_epochs: int
    progress_percentage: float = Field(ge=0, le=100)
    train_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    error_message: Optional[str] = None


class TrainingProgressResponse(BaseModel):
    """Schema for training progress response."""
    training_run_id: UUID4
    status: TrainingStatusEnum
    current_epoch: int
    total_epochs: int
    progress_percentage: float = Field(ge=0, le=100)
    current_metrics: Dict[str, float]
    estimated_completion_time: Optional[datetime] = None
    last_updated: datetime


class DatasetSummaryResponse(BaseModel):
    """Schema for dataset summary response."""
    total_samples: int
    samples_by_source: Dict[str, int]
    samples_by_label: Dict[str, int]
    human_verified_samples: int
    ready_for_training: bool
    min_samples_needed: int
    issues: List[str] = []


# =============================================================================
# EXPORT SCHEMAS
# =============================================================================

class ExportRequest(BaseModel):
    """Schema for export request."""
    format: ExportFormatEnum = ExportFormatEnum.JSON
    include_metadata: bool = True
    include_segments: bool = True
    include_classifications: bool = True


class BatchExportRequest(BaseModel):
    """Schema for batch export request."""
    image_ids: List[UUID4] = Field(min_items=1, max_items=100, description="List of image IDs to export")
    format: ExportFormatEnum = ExportFormatEnum.JSON

    @validator('image_ids')
    def validate_unique_ids(cls, v):
        if len(v) != len(set(v)):
            raise ValueError('Image IDs must be unique')
        return v


# =============================================================================
# ANALYTICS SCHEMAS
# =============================================================================

class AnalyticsOverviewRequest(BaseModel):
    """Schema for analytics overview request."""
    days: int = Field(default=30, ge=1, le=365, description="Number of days to analyze")


class AnalyticsOverviewResponse(BaseModel):
    """Schema for analytics overview response."""
    period: Dict[str, Any]
    summary: Dict[str, Any]
    top_classifications: List[Dict[str, Any]]
    daily_processing_counts: List[Dict[str, Any]]


class PerformanceMetricsResponse(BaseModel):
    """Schema for performance metrics response."""
    processing_time_statistics: Dict[str, float]
    status_distribution: Dict[str, int]
    segment_statistics: Dict[str, float]
    generated_at: datetime


# =============================================================================
# SYSTEM HEALTH SCHEMAS
# =============================================================================

class ComponentHealthSchema(BaseModel):
    """Schema for component health status."""
    status: str = Field(description="Health status: healthy, degraded, unhealthy")
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


class SystemHealthResponse(BaseModel):
    """Schema for system health response."""
    overall_status: str
    timestamp: datetime
    components: Dict[str, ComponentHealthSchema]
    metrics: Optional[Dict[str, Any]] = None
    system_info: Optional[Dict[str, Any]] = None


class ProcessingMetricsResponse(BaseModel):
    """Schema for processing metrics response."""
    time_range_hours: int
    total_processed: int
    status_distribution: Dict[str, int]
    success_rate_percent: float
    average_processing_time_seconds: float
    throughput_images_per_hour: float
    current_queue_length: int


# =============================================================================
# ERROR SCHEMAS
# =============================================================================

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationErrorResponse(ErrorResponse):
    """Schema for validation error responses."""
    validation_errors: List[Dict[str, Any]]


# =============================================================================
# WEBSOCKET SCHEMAS
# =============================================================================

class WebSocketMessage(BaseModel):
    """Schema for WebSocket messages."""
    type: str
    data: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingUpdateMessage(WebSocketMessage):
    """Schema for processing update WebSocket messages."""
    type: str = "processing_update"
    image_id: UUID4
    status: ProcessingStatusEnum
    progress_percentage: float = Field(ge=0, le=100)
    details: Optional[Dict[str, Any]] = None


class TrainingUpdateMessage(WebSocketMessage):
    """Schema for training update WebSocket messages."""
    type: str = "training_update"
    training_run_id: UUID4
    status: TrainingStatusEnum
    metrics: Optional[Dict[str, Any]] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_response(
    error_message: str,
    detail: Optional[str] = None,
    error_code: Optional[str] = None
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=error_message,
        detail=detail,
        error_code=error_code
    )


def create_validation_error_response(
    error_message: str,
    validation_errors: List[Dict[str, Any]]
) -> ValidationErrorResponse:
    """Create a standardized validation error response."""
    return ValidationErrorResponse(
        error=error_message,
        validation_errors=validation_errors
    )