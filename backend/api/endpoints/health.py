"""
VisionFlow AI - Health Check Endpoints
======================================

This module provides comprehensive health monitoring endpoints for
system status, performance metrics, and service availability checks.
These endpoints are essential for monitoring, alerting, and debugging.
"""

import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from ...database import get_db, db_manager
from ...config import get_settings
from ...models.database_models import (
    ImageRecord, ProcessingStatus, TrainingRun, TrainingStatus, SystemLog
)
from ...services.sam_service import get_sam_service
from ...services.openai_service import get_openai_service
from ...utils.helpers import get_app_info


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


# =============================================================================
# BASIC HEALTH ENDPOINTS
# =============================================================================

@router.get("/")
async def basic_health_check():
    """
    Basic health check endpoint.
    
    This provides a simple up/down status check that load balancers
    and monitoring systems can use for quick health verification.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "VisionFlow AI",
        "version": get_app_info()["version"]
    }


@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check with detailed component status.
    
    This provides detailed information about all system components,
    perfect for monitoring dashboards and debugging.
    """
    health_status = {
        "overall_status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {},
        "metrics": {},
        "system_info": get_app_info()
    }
    
    # Database health
    try:
        db_health = db_manager.health_check()
        health_status["components"]["database"] = db_health
        if db_health["status"] != "healthy":
            health_status["overall_status"] = "degraded"
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "unhealthy"
    
    # SAM service health
    try:
        sam_service = get_sam_service()
        sam_health = await sam_service.health_check()
        health_status["components"]["sam_service"] = sam_health
        if sam_health["status"] != "healthy":
            health_status["overall_status"] = "degraded"
    except Exception as e:
        health_status["components"]["sam_service"] = {
            "status": "unreachable",
            "error": str(e)
        }
        if health_status["overall_status"] == "healthy":
            health_status["overall_status"] = "degraded"
    
    # OpenAI service health
    try:
        openai_service = get_openai_service()
        openai_health = await openai_service.test_connection()
        health_status["components"]["openai_service"] = openai_health
        if openai_health["status"] != "success":
            health_status["overall_status"] = "degraded"
    except Exception as e:
        health_status["components"]["openai_service"] = {
            "status": "failed",
            "error": str(e)
        }
        if health_status["overall_status"] == "healthy":
            health_status["overall_status"] = "degraded"
    
    # Get processing metrics
    try:
        processing_metrics = await get_processing_metrics(db)
        health_status["metrics"]["processing"] = processing_metrics
    except Exception as e:
        logger.error(f"Failed to get processing metrics: {e}")
        health_status["metrics"]["processing"] = {"error": str(e)}
    
    # Get training metrics
    try:
        training_metrics = await get_training_metrics(db)
        health_status["metrics"]["training"] = training_metrics
    except Exception as e:
        logger.error(f"Failed to get training metrics: {e}")
        health_status["metrics"]["training"] = {"error": str(e)}
    
    # Return appropriate HTTP status
    if health_status["overall_status"] == "unhealthy":
        return JSONResponse(status_code=503, content=health_status)
    elif health_status["overall_status"] == "degraded":
        return JSONResponse(status_code=200, content=health_status)
    else:
        return health_status


# =============================================================================
# COMPONENT-SPECIFIC HEALTH ENDPOINTS
# =============================================================================

@router.get("/database")
async def database_health(db: Session = Depends(get_db)):
    """
    Detailed database health check.
    
    This provides comprehensive information about database performance,
    connection status, and data statistics.
    """
    try:
        # Basic connectivity test
        db_health = db_manager.health_check()
        
        # Additional database metrics
        from ...models.database_models import get_table_counts, get_processing_statistics
        
        table_counts = get_table_counts(db)
        processing_stats = get_processing_statistics(db)
        
        # Database size information
        from ...database import get_database_size
        size_info = get_database_size()
        
        return {
            "status": "healthy",
            "connection_test": db_health,
            "table_counts": table_counts,
            "processing_statistics": processing_stats,
            "database_size": size_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


@router.get("/sam-service")
async def sam_service_health():
    """
    SAM service health check with performance metrics.
    
    This provides detailed information about the SAM service status,
    performance, and resource usage.
    """
    try:
        sam_service = get_sam_service()
        
        # Get health check
        health_result = await sam_service.health_check()
        
        # Get usage statistics
        usage_stats = sam_service.get_usage_stats()
        
        return {
            "service_health": health_result,
            "usage_statistics": usage_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"SAM service health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unreachable",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


@router.get("/openai-service")
async def openai_service_health():
    """
    OpenAI service health check with usage metrics.
    
    This provides information about OpenAI API connectivity,
    usage statistics, and cost tracking.
    """
    try:
        openai_service = get_openai_service()
        
        # Test connection
        connection_test = await openai_service.test_connection()
        
        # Get usage statistics
        usage_stats = openai_service.get_usage_stats()
        
        return {
            "connection_test": connection_test,
            "usage_statistics": usage_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"OpenAI service health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


# =============================================================================
# PERFORMANCE MONITORING ENDPOINTS
# =============================================================================

@router.get("/metrics/processing")
async def processing_metrics(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """
    Get detailed processing performance metrics.
    
    This provides insights into image processing performance,
    throughput, and success rates over time.
    """
    try:
        metrics = await get_processing_metrics(db, hours)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get processing metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get processing metrics")


@router.get("/metrics/training")
async def training_metrics(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get training system performance metrics.
    
    This provides insights into model training frequency,
    success rates, and performance trends.
    """
    try:
        metrics = await get_training_metrics(db, days)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get training metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training metrics")


@router.get("/metrics/system")
async def system_metrics():
    """
    Get system-level performance metrics.
    
    This provides information about CPU, memory, disk usage,
    and other system resources.
    """
    try:
        import psutil
        import os
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_metrics = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent_used": memory.percent
        }
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_metrics = {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": round((disk.used / disk.total) * 100, 1)
        }
        
        # Process metrics
        process = psutil.Process()
        process_metrics = {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": round(process.memory_info().rss / (1024**2), 2),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files())
        }
        
        return {
            "cpu": {
                "percent_used": cpu_percent,
                "core_count": cpu_count
            },
            "memory": memory_metrics,
            "disk": disk_metrics,
            "process": process_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system metrics")


# =============================================================================
# ERROR AND LOG MONITORING ENDPOINTS
# =============================================================================

@router.get("/errors/recent")
async def recent_errors(
    hours: int = 24,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get recent error logs for monitoring and debugging.
    
    This provides access to recent errors and warnings
    for troubleshooting and system monitoring.
    """
    try:
        # Calculate time range
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Query recent error logs
        error_logs = db.query(SystemLog).filter(
            SystemLog.created_at >= start_time,
            SystemLog.level.in_(['ERROR', 'CRITICAL'])
        ).order_by(SystemLog.created_at.desc()).limit(limit).all()
        
        # Format errors
        errors = []
        for log in error_logs:
            error_data = {
                "id": str(log.id),
                "level": log.level,
                "category": log.category,
                "message": log.message,
                "details": log.details,
                "timestamp": log.created_at.isoformat(),
                "user_id": log.user_id,
                "image_id": str(log.image_id) if log.image_id else None
            }
            errors.append(error_data)
        
        # Error statistics
        error_counts = db.query(
            SystemLog.level,
            func.count(SystemLog.id).label('count')
        ).filter(
            SystemLog.created_at >= start_time,
            SystemLog.level.in_(['ERROR', 'CRITICAL', 'WARNING'])
        ).group_by(SystemLog.level).all()
        
        error_stats = {result.level: result.count for result in error_counts}
        
        return {
            "time_range_hours": hours,
            "errors": errors,
            "error_statistics": error_stats,
            "total_errors": len(errors)
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent errors: {e}")
        raise HTTPException(status_code=500, detail="Failed to get error logs")


@router.get("/logs/summary")
async def log_summary(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """
    Get summary of log activity by level and category.
    
    This provides an overview of system activity and helps
    identify patterns in errors and warnings.
    """
    try:
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Logs by level
        level_counts = db.query(
            SystemLog.level,
            func.count(SystemLog.id).label('count')
        ).filter(
            SystemLog.created_at >= start_time
        ).group_by(SystemLog.level).all()
        
        logs_by_level = {result.level: result.count for result in level_counts}
        
        # Logs by category
        category_counts = db.query(
            SystemLog.category,
            func.count(SystemLog.id).label('count')
        ).filter(
            SystemLog.created_at >= start_time
        ).group_by(SystemLog.category).order_by(func.count(SystemLog.id).desc()).limit(10).all()
        
        logs_by_category = {result.category: result.count for result in category_counts}
        
        # Hourly log counts
        hourly_counts = db.query(
            func.date_trunc('hour', SystemLog.created_at).label('hour'),
            func.count(SystemLog.id).label('count')
        ).filter(
            SystemLog.created_at >= start_time
        ).group_by(func.date_trunc('hour', SystemLog.created_at)).order_by('hour').all()
        
        hourly_data = [
            {
                "hour": result.hour.isoformat(),
                "count": result.count
            }
            for result in hourly_counts
        ]
        
        return {
            "time_range_hours": hours,
            "logs_by_level": logs_by_level,
            "logs_by_category": logs_by_category,
            "hourly_counts": hourly_data,
            "total_logs": sum(logs_by_level.values())
        }
        
    except Exception as e:
        logger.error(f"Failed to get log summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get log summary")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def get_processing_metrics(db: Session, hours: int = 24) -> Dict[str, Any]:
    """
    Calculate detailed processing performance metrics.
    
    This function analyzes processing data to provide insights
    into system performance and efficiency.
    """
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    # Total images processed
    total_processed = db.query(ImageRecord).filter(
        ImageRecord.created_at >= start_time
    ).count()
    
    # Processing by status
    status_counts = db.query(
        ImageRecord.status,
        func.count(ImageRecord.id).label('count')
    ).filter(
        ImageRecord.created_at >= start_time
    ).group_by(ImageRecord.status).all()
    
    status_distribution = {result.status.value: result.count for result in status_counts}
    
    # Success rate
    completed_count = status_distribution.get(ProcessingStatus.COMPLETED.value, 0)
    success_rate = (completed_count / total_processed * 100) if total_processed > 0 else 0
    
    # Average processing time
    avg_processing_time_query = db.query(
        func.avg(
            func.extract('epoch', ImageRecord.processing_completed_at - ImageRecord.processing_started_at)
        ).label('avg_time')
    ).filter(
        ImageRecord.created_at >= start_time,
        ImageRecord.status == ProcessingStatus.COMPLETED,
        ImageRecord.processing_started_at.isnot(None),
        ImageRecord.processing_completed_at.isnot(None)
    ).first()
    
    avg_processing_time = float(avg_processing_time_query.avg_time) if avg_processing_time_query.avg_time else 0.0
    
    # Throughput (images per hour)
    throughput = total_processed / hours if hours > 0 else 0
    
    # Processing queue length (pending images)
    queue_length = db.query(ImageRecord).filter(
        ImageRecord.status.in_([
            ProcessingStatus.UPLOADED,
            ProcessingStatus.SEGMENTING,
            ProcessingStatus.CLASSIFYING,
            ProcessingStatus.TRAINING
        ])
    ).count()
    
    return {
        "time_range_hours": hours,
        "total_processed": total_processed,
        "status_distribution": status_distribution,
        "success_rate_percent": round(success_rate, 2),
        "average_processing_time_seconds": round(avg_processing_time, 2),
        "throughput_images_per_hour": round(throughput, 2),
        "current_queue_length": queue_length
    }


async def get_training_metrics(db: Session, days: int = 30) -> Dict[str, Any]:
    """
    Calculate training system performance metrics.
    
    This function analyzes training data to provide insights
    into model training frequency and success rates.
    """
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Total training runs
    total_runs = db.query(TrainingRun).filter(
        TrainingRun.created_at >= start_time
    ).count()
    
    # Training runs by status
    status_counts = db.query(
        TrainingRun.status,
        func.count(TrainingRun.id).label('count')
    ).filter(
        TrainingRun.created_at >= start_time
    ).group_by(TrainingRun.status).all()
    
    status_distribution = {result.status.value: result.count for result in status_counts}
    
    # Success rate
    completed_count = status_distribution.get(TrainingStatus.COMPLETED.value, 0)
    success_rate = (completed_count / total_runs * 100) if total_runs > 0 else 0
    
    # Average training time
    avg_training_time_query = db.query(
        func.avg(TrainingRun.training_duration_seconds).label('avg_duration')
    ).filter(
        TrainingRun.created_at >= start_time,
        TrainingRun.status == TrainingStatus.COMPLETED,
        TrainingRun.training_duration_seconds.isnot(None)
    ).first()
    
    avg_training_time = float(avg_training_time_query.avg_duration) if avg_training_time_query.avg_duration else 0.0
    
    # Best model accuracy
    best_accuracy_query = db.query(
        func.max(TrainingRun.validation_accuracy).label('best_accuracy')
    ).filter(
        TrainingRun.created_at >= start_time,
        TrainingRun.status == TrainingStatus.COMPLETED
    ).first()
    
    best_accuracy = float(best_accuracy_query.best_accuracy) if best_accuracy_query.best_accuracy else 0.0
    
    # Training frequency
    frequency = total_runs / days if days > 0 else 0
    
    return {
        "time_range_days": days,
        "total_training_runs": total_runs,
        "status_distribution": status_distribution,
        "success_rate_percent": round(success_rate, 2),
        "average_training_time_seconds": round(avg_training_time, 2),
        "best_validation_accuracy": round(best_accuracy, 4),
        "training_frequency_per_day": round(frequency, 2)
    }