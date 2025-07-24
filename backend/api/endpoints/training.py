"""
VisionFlow AI - Model Training Endpoints
========================================

This module handles endpoints for model training and management:
- Starting and monitoring training runs
- Managing training datasets
- Viewing training metrics and progress
- Model evaluation and comparison
- Training configuration management

These endpoints enable continuous learning and model improvement based on
accumulated data and human feedback.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from pydantic import BaseModel, Field

from ...database import get_db
from ...config import get_settings
from ...models.database_models import (
    TrainingRun, TrainingSample, TrainingStatus, ImageRecord, Classification
)
from ...services.training_service import get_training_service


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class TrainingConfig(BaseModel):
    """Configuration for training runs."""
    run_name: str = Field(..., min_length=1, max_length=100)
    model_type: str = Field(default="image_classifier", description="Type of model to train")
    batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    num_epochs: int = Field(default=10, ge=1, le=1000)
    train_test_split: float = Field(default=0.8, gt=0.0, lt=1.0)
    min_samples_per_class: int = Field(default=5, ge=1)
    use_human_labels_only: bool = Field(default=False)
    augmentation_enabled: bool = Field(default=True)
    early_stopping_patience: int = Field(default=3, ge=1)


class TrainingRunSummary(BaseModel):
    """Summary information about a training run."""
    id: str
    run_name: str
    status: str
    model_type: str
    num_samples: int
    current_epoch: int
    total_epochs: int
    progress_percentage: float
    train_accuracy: Optional[float]
    validation_accuracy: Optional[float]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_seconds: Optional[int]
    error_message: Optional[str]


class DetailedTrainingRun(BaseModel):
    """Detailed information about a training run."""
    id: str
    run_name: str
    status: str
    model_type: str
    config: Dict[str, Any]
    num_samples: int
    train_test_split: float
    current_epoch: int
    total_epochs: int
    progress_percentage: float
    metrics: Dict[str, Any]
    model_path: Optional[str]
    model_size_bytes: Optional[int]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_seconds: Optional[int]
    error_message: Optional[str]
    metrics_history: Optional[List[Dict[str, Any]]]


class DatasetSummary(BaseModel):
    """Summary of the training dataset."""
    total_samples: int
    samples_by_source: Dict[str, int]
    samples_by_label: Dict[str, int]
    human_verified_samples: int
    ready_for_training: bool
    min_samples_needed: int


class TrainingProgress(BaseModel):
    """Real-time training progress information."""
    training_run_id: str
    status: str
    current_epoch: int
    total_epochs: int
    progress_percentage: float
    current_metrics: Dict[str, float]
    estimated_completion_time: Optional[str]
    last_updated: str


# =============================================================================
# TRAINING MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/start", response_model=TrainingRunSummary)
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start a new training run.
    
    This initiates model training with the specified configuration.
    Training runs in the background and progress can be monitored
    via the status endpoints.
    """
    try:
        # Check if training is already in progress
        active_training = db.query(TrainingRun).filter(
            TrainingRun.status == TrainingStatus.IN_PROGRESS
        ).first()
        
        if active_training:
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: {active_training.run_name}"
            )
        
        # Validate dataset has enough samples
        training_service = get_training_service()
        dataset_info = await training_service.get_dataset_info()
        
        if not dataset_info['ready_for_training']:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient training data. Need at least {dataset_info['min_samples_needed']} samples"
            )
        
        # Create training run record
        training_run = TrainingRun(
            run_name=config.run_name,
            model_type=config.model_type,
            config=config.dict(),
            num_samples=dataset_info['total_samples'],
            train_test_split=config.train_test_split,
            total_epochs=config.num_epochs,
            status=TrainingStatus.PENDING
        )
        
        db.add(training_run)
        db.commit()
        db.refresh(training_run)
        
        logger.info(f"Training run created: {training_run.id} ({config.run_name})")
        
        # Start training in background
        background_tasks.add_task(
            run_training_pipeline,
            str(training_run.id),
            config
        )
        
        return TrainingRunSummary(
            id=str(training_run.id),
            run_name=training_run.run_name,
            status=training_run.status.value,
            model_type=training_run.model_type,
            num_samples=training_run.num_samples,
            current_epoch=training_run.current_epoch,
            total_epochs=training_run.total_epochs,
            progress_percentage=0.0,
            train_accuracy=None,
            validation_accuracy=None,
            created_at=training_run.created_at.isoformat(),
            started_at=None,
            completed_at=None,
            duration_seconds=None,
            error_message=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start training")


@router.get("/runs", response_model=List[TrainingRunSummary])
async def list_training_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    List training runs with optional filtering.
    
    This provides a paginated list of all training runs with
    summary information for monitoring and management.
    """
    # Build query
    query = db.query(TrainingRun)
    
    if status:
        try:
            status_enum = TrainingStatus(status)
            query = query.filter(TrainingRun.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid status: {status}")
    
    # Get training runs
    training_runs = query.order_by(desc(TrainingRun.created_at)).offset(offset).limit(limit).all()
    
    # Format response
    runs = []
    for run in training_runs:
        # Calculate progress
        progress = 0.0
        if run.total_epochs > 0:
            progress = (run.current_epoch / run.total_epochs) * 100
        
        # Calculate duration
        duration = None
        if run.training_started_at and run.training_completed_at:
            duration = int((run.training_completed_at - run.training_started_at).total_seconds())
        
        run_summary = TrainingRunSummary(
            id=str(run.id),
            run_name=run.run_name,
            status=run.status.value,
            model_type=run.model_type,
            num_samples=run.num_samples,
            current_epoch=run.current_epoch,
            total_epochs=run.total_epochs,
            progress_percentage=progress,
            train_accuracy=run.train_accuracy,
            validation_accuracy=run.validation_accuracy,
            created_at=run.created_at.isoformat(),
            started_at=run.training_started_at.isoformat() if run.training_started_at else None,
            completed_at=run.training_completed_at.isoformat() if run.training_completed_at else None,
            duration_seconds=duration,
            error_message=run.error_message
        )
        runs.append(run_summary)
    
    return runs


@router.get("/runs/{run_id}", response_model=DetailedTrainingRun)
async def get_training_run(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific training run.
    
    This includes complete configuration, metrics history, and
    all available information about the training process.
    """
    training_run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    # Calculate progress
    progress = 0.0
    if training_run.total_epochs > 0:
        progress = (training_run.current_epoch / training_run.total_epochs) * 100
    
    # Calculate duration
    duration = None
    if training_run.training_started_at and training_run.training_completed_at:
        duration = int((training_run.training_completed_at - training_run.training_started_at).total_seconds())
    
    # Compile metrics
    metrics = {
        'train_accuracy': training_run.train_accuracy,
        'validation_accuracy': training_run.validation_accuracy,
        'train_loss': training_run.train_loss,
        'validation_loss': training_run.validation_loss
    }
    
    return DetailedTrainingRun(
        id=str(training_run.id),
        run_name=training_run.run_name,
        status=training_run.status.value,
        model_type=training_run.model_type,
        config=training_run.config or {},
        num_samples=training_run.num_samples,
        train_test_split=training_run.train_test_split,
        current_epoch=training_run.current_epoch,
        total_epochs=training_run.total_epochs,
        progress_percentage=progress,
        metrics=metrics,
        model_path=training_run.model_path,
        model_size_bytes=training_run.model_size_bytes,
        created_at=training_run.created_at.isoformat(),
        started_at=training_run.training_started_at.isoformat() if training_run.training_started_at else None,
        completed_at=training_run.training_completed_at.isoformat() if training_run.training_completed_at else None,
        duration_seconds=duration,
        error_message=training_run.error_message,
        metrics_history=training_run.metrics_history
    )


# =============================================================================
# TRAINING MONITORING ENDPOINTS
# =============================================================================

@router.get("/progress/{run_id}", response_model=TrainingProgress)
async def get_training_progress(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get real-time training progress for a specific run.
    
    This provides up-to-date information about training progress
    for live monitoring in the frontend.
    """
    training_run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    # Calculate progress
    progress = 0.0
    if training_run.total_epochs > 0:
        progress = (training_run.current_epoch / training_run.total_epochs) * 100
    
    # Estimate completion time
    estimated_completion = None
    if (training_run.status == TrainingStatus.IN_PROGRESS and 
        training_run.training_started_at and 
        training_run.current_epoch > 0):
        
        from datetime import timedelta
        elapsed = datetime.now(timezone.utc) - training_run.training_started_at
        epochs_per_second = training_run.current_epoch / elapsed.total_seconds()
        remaining_epochs = training_run.total_epochs - training_run.current_epoch
        
        if epochs_per_second > 0:
            remaining_seconds = remaining_epochs / epochs_per_second
            estimated_completion = (
                datetime.now(timezone.utc) + timedelta(seconds=remaining_seconds)
            ).isoformat()
    
    # Get current metrics
    current_metrics = {}
    if training_run.metrics_history and training_run.current_epoch > 0:
        # Get latest metrics from history
        latest_metrics = training_run.metrics_history[-1] if training_run.metrics_history else {}
        current_metrics = {
            'train_accuracy': latest_metrics.get('train_accuracy', 0.0),
            'validation_accuracy': latest_metrics.get('validation_accuracy', 0.0),
            'train_loss': latest_metrics.get('train_loss', 0.0),
            'validation_loss': latest_metrics.get('validation_loss', 0.0)
        }
    
    return TrainingProgress(
        training_run_id=str(training_run.id),
        status=training_run.status.value,
        current_epoch=training_run.current_epoch,
        total_epochs=training_run.total_epochs,
        progress_percentage=progress,
        current_metrics=current_metrics,
        estimated_completion_time=estimated_completion,
        last_updated=datetime.now(timezone.utc).isoformat()
    )


@router.get("/metrics/{run_id}")
async def get_training_metrics(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed training metrics and history for visualization.
    
    This provides all metrics data needed to create training curves
    and performance visualizations in the frontend.
    """
    training_run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    if not training_run.metrics_history:
        return {
            'training_run_id': str(training_run.id),
            'metrics_available': False,
            'message': 'No metrics available yet'
        }
    
    # Extract metrics for plotting
    epochs = []
    train_accuracy = []
    validation_accuracy = []
    train_loss = []
    validation_loss = []
    
    for epoch_metrics in training_run.metrics_history:
        epochs.append(epoch_metrics.get('epoch', 0))
        train_accuracy.append(epoch_metrics.get('train_accuracy', 0.0))
        validation_accuracy.append(epoch_metrics.get('validation_accuracy', 0.0))
        train_loss.append(epoch_metrics.get('train_loss', 0.0))
        validation_loss.append(epoch_metrics.get('validation_loss', 0.0))
    
    return {
        'training_run_id': str(training_run.id),
        'metrics_available': True,
        'epochs': epochs,
        'train_accuracy': train_accuracy,
        'validation_accuracy': validation_accuracy,
        'train_loss': train_loss,
        'validation_loss': validation_loss,
        'best_validation_accuracy': max(validation_accuracy) if validation_accuracy else 0.0,
        'final_train_accuracy': train_accuracy[-1] if train_accuracy else 0.0,
        'final_validation_accuracy': validation_accuracy[-1] if validation_accuracy else 0.0
    }


# =============================================================================
# DATASET MANAGEMENT ENDPOINTS
# =============================================================================

@router.get("/dataset", response_model=DatasetSummary)
async def get_dataset_summary(
    db: Session = Depends(get_db)
):
    """
    Get summary information about the training dataset.
    
    This provides insights into the available training data,
    including data distribution and readiness for training.
    """
    try:
        training_service = get_training_service()
        dataset_info = await training_service.get_dataset_info()
        
        return DatasetSummary(
            total_samples=dataset_info['total_samples'],
            samples_by_source=dataset_info['samples_by_source'],
            samples_by_label=dataset_info['samples_by_label'],
            human_verified_samples=dataset_info['human_verified_samples'],
            ready_for_training=dataset_info['ready_for_training'],
            min_samples_needed=dataset_info['min_samples_needed']
        )
        
    except Exception as e:
        logger.error(f"Failed to get dataset summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dataset information")


@router.get("/dataset/samples")
async def list_training_samples(
    label: Optional[str] = Query(None, description="Filter by label"),
    source: Optional[str] = Query(None, description="Filter by source (openai, human, model)"),
    verified_only: bool = Query(False, description="Show only human-verified samples"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    List training samples with filtering options.
    
    This allows detailed inspection of the training dataset
    for quality control and analysis.
    """
    # Build query
    query = db.query(TrainingSample).join(ImageRecord)
    
    if label:
        query = query.filter(TrainingSample.ground_truth_label == label)
    
    if source:
        query = query.filter(TrainingSample.label_source == source)
    
    if verified_only:
        query = query.join(Classification).filter(Classification.human_verified == True)
    
    # Get samples
    samples = query.order_by(desc(TrainingSample.created_at)).offset(offset).limit(limit).all()
    
    # Format response
    sample_list = []
    for sample in samples:
        sample_data = {
            'id': str(sample.id),
            'image_id': str(sample.image_id),
            'segment_id': str(sample.segment_id) if sample.segment_id else None,
            'ground_truth_label': sample.ground_truth_label,
            'label_source': sample.label_source,
            'used_in_training': sample.used_in_training,
            'difficulty_score': sample.difficulty_score,
            'importance_weight': sample.importance_weight,
            'created_at': sample.created_at.isoformat(),
            'image_filename': sample.image.filename if sample.image else None
        }
        sample_list.append(sample_data)
    
    return {
        'samples': sample_list,
        'total_count': query.count(),
        'limit': limit,
        'offset': offset
    }


# =============================================================================
# TRAINING CONTROL ENDPOINTS
# =============================================================================

@router.post("/runs/{run_id}/pause")
async def pause_training(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Pause an ongoing training run.
    
    This allows pausing training that's currently in progress,
    which can be useful for resource management or parameter adjustments.
    """
    training_run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    if training_run.status != TrainingStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot pause training. Current status: {training_run.status.value}"
        )
    
    # Update status
    training_run.status = TrainingStatus.PAUSED
    db.commit()
    
    logger.info(f"Training run paused: {run_id}")
    
    return {
        'success': True,
        'message': f'Training run {run_id} paused successfully',
        'status': TrainingStatus.PAUSED.value
    }


@router.post("/runs/{run_id}/resume")
async def resume_training(
    run_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Resume a paused training run.
    
    This continues training from where it was paused,
    maintaining all previous progress and metrics.
    """
    training_run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    if training_run.status != TrainingStatus.PAUSED:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot resume training. Current status: {training_run.status.value}"
        )
    
    # Check for other active training
    active_training = db.query(TrainingRun).filter(
        TrainingRun.status == TrainingStatus.IN_PROGRESS,
        TrainingRun.id != run_id
    ).first()
    
    if active_training:
        raise HTTPException(
            status_code=409,
            detail=f"Another training run is already in progress: {active_training.run_name}"
        )
    
    # Update status and resume
    training_run.status = TrainingStatus.IN_PROGRESS
    db.commit()
    
    # Resume training in background
    config = TrainingConfig(**training_run.config)
    background_tasks.add_task(
        run_training_pipeline,
        run_id,
        config,
        resume=True
    )
    
    logger.info(f"Training run resumed: {run_id}")
    
    return {
        'success': True,
        'message': f'Training run {run_id} resumed successfully',
        'status': TrainingStatus.IN_PROGRESS.value
    }


@router.post("/runs/{run_id}/stop")
async def stop_training(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Stop a training run permanently.
    
    This terminates training and marks the run as failed.
    Use with caution as this cannot be undone.
    """
    training_run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    if training_run.status not in [TrainingStatus.IN_PROGRESS, TrainingStatus.PAUSED]:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot stop training. Current status: {training_run.status.value}"
        )
    
    # Update status
    training_run.status = TrainingStatus.FAILED
    training_run.error_message = "Training stopped by user"
    training_run.training_completed_at = datetime.now(timezone.utc)
    db.commit()
    
    logger.info(f"Training run stopped: {run_id}")
    
    return {
        'success': True,
        'message': f'Training run {run_id} stopped successfully',
        'status': TrainingStatus.FAILED.value
    }


# =============================================================================
# MODEL MANAGEMENT ENDPOINTS
# =============================================================================

@router.get("/models")
async def list_trained_models(
    db: Session = Depends(get_db)
):
    """
    List all successfully trained models.
    
    This provides information about available models that can be
    used for inference or comparison.
    """
    # Get completed training runs with models
    completed_runs = db.query(TrainingRun).filter(
        TrainingRun.status == TrainingStatus.COMPLETED,
        TrainingRun.model_path.isnot(None)
    ).order_by(desc(TrainingRun.training_completed_at)).all()
    
    models = []
    for run in completed_runs:
        model_info = {
            'training_run_id': str(run.id),
            'model_name': run.run_name,
            'model_type': run.model_type,
            'model_path': run.model_path,
            'model_size_bytes': run.model_size_bytes,
            'validation_accuracy': run.validation_accuracy,
            'train_accuracy': run.train_accuracy,
            'num_samples': run.num_samples,
            'created_at': run.training_completed_at.isoformat(),
            'training_duration_seconds': run.training_duration_seconds
        }
        models.append(model_info)
    
    return {
        'models': models,
        'total_count': len(models)
    }


@router.get("/models/{run_id}/download")
async def download_model(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Download a trained model file.
    
    This allows downloading the actual model file for deployment
    or external use.
    """
    training_run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")
    
    if training_run.status != TrainingStatus.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Model not available. Training status: {training_run.status.value}"
        )
    
    if not training_run.model_path:
        raise HTTPException(status_code=404, detail="Model file not found")
    
    import os
    if not os.path.exists(training_run.model_path):
        raise HTTPException(status_code=404, detail="Model file not found on disk")
    
    filename = f"{training_run.run_name}_model.pth"
    return FileResponse(
        path=training_run.model_path,
        media_type="application/octet-stream",
        filename=filename
    )


# =============================================================================
# BACKGROUND TRAINING PIPELINE
# =============================================================================

async def run_training_pipeline(
    training_run_id: str,
    config: TrainingConfig,
    resume: bool = False
):
    """
    Execute the training pipeline in the background.
    
    This is the main training orchestration function that manages
    the entire training process from data preparation to model saving.
    """
    from ...database import db_manager
    from ...main import broadcast_training_update
    
    logger.info(f"Starting training pipeline for run: {training_run_id}")
    
    try:
        training_service = get_training_service()
        
        # Start the training process
        await training_service.train_model(
            training_run_id=training_run_id,
            config=config.dict(),
            resume=resume,
            progress_callback=lambda status, metrics: asyncio.create_task(
                broadcast_training_update(training_run_id, status, metrics)
            )
        )
        
        logger.info(f"Training completed successfully for run: {training_run_id}")
        
    except Exception as e:
        logger.error(f"Training failed for run {training_run_id}: {e}")
        
        # Update status to failed
        try:
            with db_manager.get_session_context() as db:
                training_run = db.query(TrainingRun).filter(
                    TrainingRun.id == training_run_id
                ).first()
                
                if training_run:
                    training_run.status = TrainingStatus.FAILED
                    training_run.error_message = str(e)
                    training_run.training_completed_at = datetime.now(timezone.utc)
                    db.commit()
                    
                    await broadcast_training_update(training_run_id, "failed", {
                        "error": str(e)
                    })
        except Exception as db_error:
            logger.error(f"Failed to update training status: {db_error}")