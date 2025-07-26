"""
VisionFlow AI - Database Models
===============================

This module defines the database schema for VisionFlow AI using SQLAlchemy ORM.
Think of these models as the "blueprint" for how data is stored and how different
pieces of information relate to each other.

Why SQLAlchemy ORM?
- It provides a Pythonic way to work with databases
- Automatic SQL generation and optimization
- Database-agnostic (works with SQLite, PostgreSQL, MySQL, etc.)
- Built-in relationship management and lazy loading
- Excellent integration with FastAPI and Pydantic
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List
import enum

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, 
    JSON, ForeignKey, Enum, LargeBinary, Index
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, Session
# from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

import sqlalchemy as sa
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID

class UniversalUUID(TypeDecorator):
    """
    Universal UUID type that works with both PostgreSQL and SQLite.
    
    This automatically chooses the right UUID implementation:
    - PostgreSQL: Uses native UUID type for optimal performance
    - SQLite: Stores UUIDs as strings (36 characters)
    - Other databases: Falls back to string storage
    
    This is like having a universal power adapter that works in any country!
    """
    
    impl = CHAR
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        """Choose the right UUID implementation based on the database type."""
        if dialect.name == 'postgresql':
            # Use PostgreSQL's native UUID type
            return dialect.type_descriptor(PostgreSQLUUID(as_uuid=True))
        else:
            # For SQLite and other databases, use 36-character strings
            return dialect.type_descriptor(CHAR(36))
    
    def process_bind_param(self, value, dialect):
        """Convert Python UUID objects to the right format for storage."""
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            # PostgreSQL can handle UUID objects directly
            return value
        else:
            # For SQLite, convert UUID to string format
            if not isinstance(value, uuid.UUID):
                return str(value)
            return str(value)
    
    def process_result_value(self, value, dialect):
        """Convert stored values back to Python UUID objects."""
        if value is None:
            return value
        else:
            # Always return a proper UUID object regardless of storage format
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value

# =============================================================================
# BASE MODEL SETUP
# =============================================================================

Base = declarative_base()


class TimestampMixin:
    """
    Mixin class that adds created_at and updated_at timestamps to models.
    
    This is a common pattern in database design - every record should know
    when it was created and last modified. We use a mixin so we don't have
    to repeat this code in every model.
    """
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        comment="When this record was created"
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="When this record was last updated"
    )


class UUIDMixin:
    """
    Mixin class that adds a UUID primary key to models.
    
    UUIDs are better than sequential integers for primary keys because:
    - They're globally unique (can merge databases safely)
    - They don't reveal information about your data volume
    - They're harder to guess (better security)
    """
    id = Column(
        UniversalUUID(),  # New universal version that works everywhere
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique identifier for this record"
    )


# =============================================================================
# ENUMS FOR STATUS TRACKING
# =============================================================================

class ProcessingStatus(str, enum.Enum):
    """
    Status tracking for image processing pipeline.
    
    Think of this like a package tracking system - we want to know
    exactly where each image is in the processing pipeline.
    """
    UPLOADED = "uploaded"           # Image just uploaded, waiting to be processed
    SEGMENTING = "segmenting"       # Currently being segmented by SAM
    CLASSIFYING = "classifying"     # Currently being classified by OpenAI
    TRAINING = "training"           # Being used for model training
    COMPLETED = "completed"         # All processing complete
    FAILED = "failed"              # Processing failed at some step
    CANCELLED = "cancelled"        # Processing was cancelled by user


class TrainingStatus(str, enum.Enum):
    """Status tracking for model training."""
    PENDING = "pending"            # Waiting for enough data to start training
    IN_PROGRESS = "in_progress"    # Currently training
    COMPLETED = "completed"        # Training completed successfully
    FAILED = "failed"             # Training failed
    PAUSED = "paused"             # Training paused by user


# =============================================================================
# CORE DATA MODELS
# =============================================================================

class ImageRecord(Base, UUIDMixin, TimestampMixin):
    """
    Central record for each image processed by the system.
    
    This is the "main character" of our database - everything else
    relates back to an image record. Think of it as the master record
    that tracks an image's journey through our pipeline.
    """
    __tablename__ = "images"
    
    # Basic image information
    filename = Column(String(255), nullable=False, comment="Original filename")
    file_path = Column(String(500), nullable=False, comment="Path to stored image file")
    file_size = Column(Integer, nullable=False, comment="File size in bytes")
    mime_type = Column(String(50), nullable=False, comment="MIME type (image/jpeg, etc.)")
    
    # Image dimensions and metadata
    width = Column(Integer, comment="Image width in pixels")
    height = Column(Integer, comment="Image height in pixels")
    channels = Column(Integer, comment="Number of color channels")
    
    # Processing status and metadata
    status = Column(
        Enum(ProcessingStatus), 
        default=ProcessingStatus.UPLOADED,
        nullable=False,
        comment="Current processing status"
    )
    processing_started_at = Column(DateTime(timezone=True), comment="When processing began")
    processing_completed_at = Column(DateTime(timezone=True), comment="When processing finished")
    
    # User context (for multi-user systems)
    user_id = Column(String(100), comment="ID of user who uploaded this image")
    
    # Error tracking
    error_message = Column(Text, comment="Error message if processing failed")
    error_details = Column(JSON, comment="Detailed error information")
    
    # Processing configuration snapshot
    processing_config = Column(JSON, comment="Configuration used for processing this image")
    
    # Relationships to other tables
    segments = relationship("ImageSegment", back_populates="image", cascade="all, delete-orphan")
    classifications = relationship("Classification", back_populates="image", cascade="all, delete-orphan")
    training_samples = relationship("TrainingSample", back_populates="image", cascade="all, delete-orphan")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_images_status', 'status'),
        Index('idx_images_user_id', 'user_id'),
        Index('idx_images_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ImageRecord(id={self.id}, filename={self.filename}, status={self.status})>"


class ImageSegment(Base, UUIDMixin, TimestampMixin):
    """
    Individual segments created by SAM (Segment Anything Model).
    
    When SAM processes an image, it creates multiple segments - think of it
    like cutting a photograph into puzzle pieces, where each piece represents
    a distinct object or region.
    """
    __tablename__ = "image_segments"
    
    # Link back to the original image
    image_id = Column(UniversalUUID(), ForeignKey('images.id'), nullable=False)
    
    # Segment identification
    segment_index = Column(Integer, nullable=False, comment="Order of this segment in the original image")
    
    # Bounding box coordinates (x, y, width, height)
    bbox_x = Column(Integer, nullable=False, comment="Bounding box X coordinate")
    bbox_y = Column(Integer, nullable=False, comment="Bounding box Y coordinate")
    bbox_width = Column(Integer, nullable=False, comment="Bounding box width")
    bbox_height = Column(Integer, nullable=False, comment="Bounding box height")
    
    # Segment properties
    area = Column(Integer, nullable=False, comment="Segment area in pixels")
    confidence_score = Column(Float, comment="SAM's confidence in this segment")
    
    # File storage
    segment_path = Column(String(500), comment="Path to cropped segment image")
    mask_path = Column(String(500), comment="Path to segment mask file")
    
    # Segment metadata from SAM
    sam_metadata = Column(JSON, comment="Additional metadata from SAM model")
    
    # Relationships
    image = relationship("ImageRecord", back_populates="segments")
    classifications = relationship("Classification", back_populates="segment", cascade="all, delete-orphan")
    
    # Indexes for spatial queries
    __table_args__ = (
        Index('idx_segments_image_id', 'image_id'),
        Index('idx_segments_bbox', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'),
        Index('idx_segments_area', 'area'),
    )
    
    def __repr__(self):
        return f"<ImageSegment(id={self.id}, image_id={self.image_id}, index={self.segment_index})>"


class Classification(Base, UUIDMixin, TimestampMixin):
    """
    Classification results from OpenAI API for each segment.
    
    This is where we store what OpenAI "thinks" each segment contains.
    Think of it as having an expert food critic look at each piece of
    your segmented image and tell you what they see.
    """
    __tablename__ = "classifications"
    
    # Links to image and segment
    image_id = Column(UniversalUUID(), ForeignKey('images.id'), nullable=False)
    segment_id = Column(UniversalUUID(), ForeignKey('image_segments.id'), nullable=False)
    
    # OpenAI classification results
    primary_label = Column(String(100), nullable=False, comment="Primary classification label")
    confidence_score = Column(Float, comment="Confidence in primary classification")
    
    # Alternative classifications (OpenAI often provides multiple options)
    alternative_labels = Column(JSON, comment="Alternative classification options with scores")
    
    # Full OpenAI response
    raw_response = Column(JSON, comment="Complete response from OpenAI API")
    
    # Model information
    model_used = Column(String(50), comment="OpenAI model used for classification")
    tokens_used = Column(Integer, comment="Number of tokens consumed")
    
    # Human feedback (for training improvement)
    human_verified = Column(Boolean, default=False, comment="Has a human verified this classification?")
    human_label = Column(String(100), comment="Human-corrected label if different from AI")
    human_feedback_notes = Column(Text, comment="Additional human feedback")
    
    # Relationships
    image = relationship("ImageRecord", back_populates="classifications")
    segment = relationship("ImageSegment", back_populates="classifications")
    
    # Indexes for label searches and filtering
    __table_args__ = (
        Index('idx_classifications_image_id', 'image_id'),
        Index('idx_classifications_segment_id', 'segment_id'),
        Index('idx_classifications_primary_label', 'primary_label'),
        Index('idx_classifications_human_verified', 'human_verified'),
    )
    
    def __repr__(self):
        return f"<Classification(id={self.id}, label={self.primary_label}, confidence={self.confidence_score})>"


class TrainingSample(Base, UUIDMixin, TimestampMixin):
    """
    Training samples for continuous learning.
    
    This table stores data that will be used to train our local model.
    Think of it as a "study guide" that gets better over time as we
    collect more examples.
    """
    __tablename__ = "training_samples"
    
    # Source data
    image_id = Column(UniversalUUID(), ForeignKey('images.id'), nullable=False)
    segment_id = Column(UniversalUUID(), ForeignKey('image_segments.id'), nullable=True)
    
    # Training labels
    ground_truth_label = Column(String(100), nullable=False, comment="Correct label for training")
    label_source = Column(String(20), nullable=False, comment="Source of label: 'openai', 'human', 'model'")
    
    # Feature data (could be image features, embeddings, etc.)
    features = Column(JSON, comment="Feature vector or other training data")
    
    # Training metadata
    used_in_training = Column(Boolean, default=False, comment="Has this sample been used in training?")
    training_run_id = Column(UniversalUUID(), comment="ID of training run that used this sample")
    
    # Quality metrics
    difficulty_score = Column(Float, comment="How difficult is this sample to classify?")
    importance_weight = Column(Float, default=1.0, comment="Weight for this sample in training")
    
    # Relationships
    image = relationship("ImageRecord", back_populates="training_samples")
    
    __table_args__ = (
        Index('idx_training_samples_image_id', 'image_id'),
        Index('idx_training_samples_label', 'ground_truth_label'),
        Index('idx_training_samples_used', 'used_in_training'),
    )
    
    def __repr__(self):
        return f"<TrainingSample(id={self.id}, label={self.ground_truth_label}, source={self.label_source})>"


class TrainingRun(Base, UUIDMixin, TimestampMixin):
    """
    Record of each model training session.
    
    This tracks each time we train or retrain our local model,
    including performance metrics and configuration.
    """
    __tablename__ = "training_runs"
    
    # Training identification
    run_name = Column(String(100), comment="Human-readable name for this training run")
    model_type = Column(String(50), nullable=False, comment="Type of model being trained")
    
    # Training configuration
    config = Column(JSON, nullable=False, comment="Training configuration used")
    
    # Training data
    num_samples = Column(Integer, nullable=False, comment="Number of training samples used")
    train_test_split = Column(Float, default=0.8, comment="Fraction of data used for training vs testing")
    
    # Training progress
    status = Column(
        Enum(TrainingStatus), 
        default=TrainingStatus.PENDING,
        nullable=False
    )
    current_epoch = Column(Integer, default=0, comment="Current training epoch")
    total_epochs = Column(Integer, comment="Total epochs planned")
    
    # Performance metrics
    train_accuracy = Column(Float, comment="Training accuracy achieved")
    validation_accuracy = Column(Float, comment="Validation accuracy achieved")
    train_loss = Column(Float, comment="Final training loss")
    validation_loss = Column(Float, comment="Final validation loss")
    
    # Model storage
    model_path = Column(String(500), comment="Path to saved model file")
    model_size_bytes = Column(Integer, comment="Size of saved model in bytes")
    
    # Timing information
    training_started_at = Column(DateTime(timezone=True))
    training_completed_at = Column(DateTime(timezone=True))
    training_duration_seconds = Column(Integer, comment="Total training time in seconds")
    
    # Error tracking
    error_message = Column(Text, comment="Error message if training failed")
    error_details = Column(JSON, comment="Detailed error information")
    
    # Performance tracking over time
    metrics_history = Column(JSON, comment="Per-epoch metrics for plotting training curves")
    
    __table_args__ = (
        Index('idx_training_runs_status', 'status'),
        Index('idx_training_runs_model_type', 'model_type'),
        Index('idx_training_runs_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<TrainingRun(id={self.id}, name={self.run_name}, status={self.status})>"


class SystemLog(Base, UUIDMixin, TimestampMixin):
    """
    System-wide logging and monitoring.
    
    This table stores important system events, errors, and metrics
    for monitoring and debugging. Think of it as the system's diary.
    """
    __tablename__ = "system_logs"
    
    # Log classification
    level = Column(String(10), nullable=False, comment="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    category = Column(String(50), nullable=False, comment="Log category: api, training, processing, etc.")
    
    # Log content
    message = Column(Text, nullable=False, comment="Log message")
    details = Column(JSON, comment="Additional structured log data")
    
    # Context information
    user_id = Column(String(100), comment="User associated with this log entry")
    image_id = Column(UniversalUUID(), comment="Image associated with this log entry")
    request_id = Column(String(100), comment="Request ID for tracing")
    
    # System information
    hostname = Column(String(100), comment="Server hostname")
    process_id = Column(Integer, comment="Process ID")
    memory_usage_mb = Column(Float, comment="Memory usage at time of log")
    cpu_usage_percent = Column(Float, comment="CPU usage at time of log")
    
    __table_args__ = (
        Index('idx_system_logs_level', 'level'),
        Index('idx_system_logs_category', 'category'),
        Index('idx_system_logs_created_at', 'created_at'),
        Index('idx_system_logs_user_id', 'user_id'),
    )
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level={self.level}, category={self.category})>"


# =============================================================================
# DATABASE UTILITY FUNCTIONS
# =============================================================================

def create_all_tables(engine):
    """
    Create all database tables.
    
    This function creates all the tables defined above in the database.
    It's safe to call multiple times - it won't recreate existing tables.
    """
    Base.metadata.create_all(bind=engine)


def get_table_counts(db_session: Session) -> dict:
    """
    Get count of records in each table for monitoring.
    
    This is useful for system monitoring and understanding
    how much data you have in each table.
    """
    return {
        'images': db_session.query(ImageRecord).count(),
        'segments': db_session.query(ImageSegment).count(),
        'classifications': db_session.query(Classification).count(),
        'training_samples': db_session.query(TrainingSample).count(),
        'training_runs': db_session.query(TrainingRun).count(),
        'system_logs': db_session.query(SystemLog).count(),
    }


def cleanup_old_logs(db_session: Session, days_to_keep: int = 30):
    """
    Clean up old log entries to prevent database bloat.
    
    System logs can grow very large over time. This function
    removes log entries older than the specified number of days.
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
    
    deleted_count = db_session.query(SystemLog).filter(
        SystemLog.created_at < cutoff_date
    ).delete()
    
    db_session.commit()
    return deleted_count


def get_processing_statistics(db_session: Session) -> dict:
    """
    Get summary statistics about image processing.
    
    This provides insights into how your system is performing:
    - How many images are being processed?
    - What's the success rate?
    - Average processing time?
    """
    from sqlalchemy import func
    
    stats = db_session.query(
        ImageRecord.status,
        func.count(ImageRecord.id).label('count'),
        func.avg(
            func.extract('epoch', ImageRecord.processing_completed_at - ImageRecord.processing_started_at)
        ).label('avg_processing_time_seconds')
    ).group_by(ImageRecord.status).all()
    
    return {
        'by_status': {stat.status: {'count': stat.count, 'avg_time': stat.avg_processing_time_seconds} 
                     for stat in stats},
        'total_images': db_session.query(ImageRecord).count(),
        'completed_today': db_session.query(ImageRecord).filter(
            ImageRecord.status == ProcessingStatus.COMPLETED,
            ImageRecord.created_at >= datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()
    }