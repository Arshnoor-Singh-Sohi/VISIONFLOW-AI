"""
VisionFlow AI - Daily Processing Tasks
=====================================

This module contains all the automated tasks that keep VisionFlow AI running 
smoothly in the background. Think of this as your digital maintenance crew
that works around the clock to process images, train models, clean up files,
and keep everything optimized.

Each function in this module represents a specific "job" that the scheduler
can assign and execute automatically. These tasks are designed to be robust,
well-logged, and capable of handling errors gracefully.
"""

import os
import sys
import asyncio
import logging
import shutil
import glob
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    from backend.database import db_manager
    from backend.models.database_models import (
        ImageRecord, ProcessingStatus, TrainingRun, TrainingStatus, SystemLog
    )
    from backend.services.training_service import get_training_service
    from backend.config import get_settings
    from backend.utils.helpers import format_bytes, format_duration
except ImportError as e:
    print(f"Warning: Could not import VisionFlow modules: {e}")
    print("Some functions may not work properly outside the VisionFlow environment")

# Set up logging for this module
logger = logging.getLogger(__name__)


# =============================================================================
# IMAGE PROCESSING TASKS
# =============================================================================

async def process_daily_images(max_images: int = 50, source: str = "folder", source_path: str = "./data/daily_inputs") -> Dict[str, Any]:
    """
    Process new images that have been uploaded or collected overnight.
    
    This function acts like an overnight worker who processes all the images
    that have accumulated while everyone was sleeping. It's designed to handle
    batches of images efficiently and safely, with comprehensive error handling
    and progress tracking.
    
    Args:
        max_images: Maximum number of images to process in this run
        source: Where to get images from ("folder", "url", "api")
        source_path: Path to folder containing images (if source is "folder")
        
    Returns:
        Dictionary with processing results and statistics
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting daily image processing: max_images={max_images}, source={source}")
    
    results = {
        "start_time": start_time.isoformat(),
        "source": source,
        "source_path": source_path,
        "images_found": 0,
        "images_processed": 0,
        "images_failed": 0,
        "processing_time_seconds": 0,
        "errors": [],
        "success": False
    }
    
    try:
        if source == "folder":
            image_files = await _find_images_in_folder(source_path, max_images)
        elif source == "url":
            image_files = await _download_images_from_url(source_path, max_images)
        elif source == "api":
            image_files = await _fetch_images_from_api(source_path, max_images)
        else:
            raise ValueError(f"Unknown source type: {source}")
        
        results["images_found"] = len(image_files)
        logger.info(f"Found {len(image_files)} images to process")
        
        if not image_files:
            logger.info("No new images found to process")
            results["success"] = True
            return results
        
        # Process each image through the VisionFlow pipeline
        for i, image_path in enumerate(image_files, 1):
            try:
                logger.info(f"Processing image {i}/{len(image_files)}: {image_path}")
                
                # Upload image to VisionFlow backend
                success = await _upload_image_to_backend(image_path)
                
                if success:
                    results["images_processed"] += 1
                    
                    # Move processed image to completed folder
                    await _move_processed_image(image_path, success=True)
                    
                else:
                    results["images_failed"] += 1
                    results["errors"].append(f"Failed to upload: {image_path}")
                    
                    # Move failed image to error folder
                    await _move_processed_image(image_path, success=False)
                
                # Add small delay to prevent overwhelming the system
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                results["images_failed"] += 1
                results["errors"].append(f"Error with {image_path}: {str(e)}")
                
                # Move failed image to error folder
                await _move_processed_image(image_path, success=False)
        
        # Calculate final statistics
        end_time = datetime.now(timezone.utc)
        results["processing_time_seconds"] = (end_time - start_time).total_seconds()
        results["end_time"] = end_time.isoformat()
        results["success"] = True
        
        logger.info(f"Daily image processing completed: {results['images_processed']} processed, "
                   f"{results['images_failed']} failed in {results['processing_time_seconds']:.1f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Daily image processing failed: {e}")
        results["errors"].append(f"Task failed: {str(e)}")
        results["processing_time_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        return results


async def _find_images_in_folder(folder_path: str, max_count: int) -> List[str]:
    """
    Find image files in a specified folder.
    
    This function acts like a careful file scout, looking through a folder
    to find valid image files while respecting limits and checking file formats.
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning(f"Image folder does not exist: {folder_path}")
        return []
    
    # Image file extensions we can process
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    image_files = []
    for ext in image_extensions:
        pattern = folder / f"*{ext}"
        files = glob.glob(str(pattern), recursive=False)
        image_files.extend(files)
        
        # Also check uppercase extensions
        pattern = folder / f"*{ext.upper()}"
        files = glob.glob(str(pattern), recursive=False)
        image_files.extend(files)
    
    # Sort by modification time (oldest first) and limit the count
    image_files.sort(key=lambda x: os.path.getmtime(x))
    
    return image_files[:max_count]


async def _upload_image_to_backend(image_path: str) -> bool:
    """
    Upload an image to the VisionFlow backend for processing.
    
    This function acts like a delivery person, taking an image file and
    submitting it to the backend API for the full processing pipeline.
    """
    try:
        import aiohttp
        import aiofiles
        
        settings = get_settings()
        backend_url = settings.cors_origins[0] if settings.cors_origins else "http://localhost:8000"
        
        async with aiohttp.ClientSession() as session:
            # Read the image file
            async with aiofiles.open(image_path, 'rb') as f:
                file_content = await f.read()
            
            # Prepare the upload
            data = aiohttp.FormData()
            data.add_field(
                'image',
                file_content,
                filename=os.path.basename(image_path),
                content_type='image/jpeg'
            )
            data.add_field('user_id', 'scheduler')
            data.add_field('config', json.dumps({
                'min_area': 1000,
                'max_segments': 50,
                'confidence_threshold': 0.7,
                'classification_context': 'food identification'
            }))
            
            # Submit to backend
            async with session.post(
                f"{backend_url}/api/v1/images/upload",
                data=data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully uploaded image: {result.get('image_id')}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Upload failed with status {response.status}: {error_text}")
                    return False
                    
    except Exception as e:
        logger.error(f"Error uploading image {image_path}: {e}")
        return False


async def _move_processed_image(image_path: str, success: bool):
    """
    Move a processed image to the appropriate completion folder.
    
    This function acts like a filing clerk, organizing processed images
    into different folders based on whether they were successful or failed.
    """
    try:
        base_path = Path(image_path).parent
        filename = Path(image_path).name
        
        if success:
            target_folder = base_path / "completed"
        else:
            target_folder = base_path / "failed"
        
        target_folder.mkdir(exist_ok=True)
        target_path = target_folder / filename
        
        shutil.move(image_path, target_path)
        logger.debug(f"Moved image to {target_path}")
        
    except Exception as e:
        logger.error(f"Error moving processed image {image_path}: {e}")


# =============================================================================
# TRAINING TASKS
# =============================================================================

async def check_training_trigger(min_samples: int = 100, time_preference: str = "night") -> Dict[str, Any]:
    """
    Check if model training should be triggered based on available data.
    
    This function acts like a training coordinator who analyzes the current
    situation and decides whether it's a good time to start training a new
    model based on available data and system conditions.
    
    Args:
        min_samples: Minimum number of new samples needed to trigger training
        time_preference: Preferred time for training ("any", "night", "day", "weekend")
        
    Returns:
        Dictionary with training decision and reasoning
    """
    logger.info("Checking if training should be triggered")
    
    results = {
        "should_train": False,
        "reason": "",
        "new_samples_count": 0,
        "total_samples_count": 0,
        "time_check_passed": False,
        "resource_check_passed": False,
        "training_run_id": None
    }
    
    try:
        # Check if training is already in progress
        with db_manager.get_session_context() as db:
            active_training = db.query(TrainingRun).filter(
                TrainingRun.status.in_([
                    TrainingStatus.PENDING,
                    TrainingStatus.IN_PROGRESS,
                    TrainingStatus.PAUSED
                ])
            ).first()
            
            if active_training:
                results["reason"] = f"Training already in progress: {active_training.run_name}"
                return results
            
            # Get the last completed training run
            last_training = db.query(TrainingRun).filter(
                TrainingRun.status == TrainingStatus.COMPLETED
            ).order_by(TrainingRun.training_completed_at.desc()).first()
            
            # Count new samples since last training
            if last_training:
                cutoff_time = last_training.training_completed_at
                new_samples_count = db.query(TrainingSample).filter(
                    TrainingSample.created_at > cutoff_time,
                    TrainingSample.used_in_training == False
                ).count()
            else:
                new_samples_count = db.query(TrainingSample).filter(
                    TrainingSample.used_in_training == False
                ).count()
            
            results["new_samples_count"] = new_samples_count
            results["total_samples_count"] = db.query(TrainingSample).count()
        
        # Check if we have enough new samples
        if new_samples_count < min_samples:
            results["reason"] = f"Insufficient new samples: {new_samples_count} < {min_samples}"
            return results
        
        # Check time preference
        time_check = _check_time_preference(time_preference)
        results["time_check_passed"] = time_check["allowed"]
        
        if not time_check["allowed"]:
            results["reason"] = f"Time preference not met: {time_check['reason']}"
            return results
        
        # Check system resources
        resource_check = await _check_training_resources()
        results["resource_check_passed"] = resource_check["available"]
        
        if not resource_check["available"]:
            results["reason"] = f"Insufficient resources: {resource_check['reason']}"
            return results
        
        # All checks passed - trigger training
        results["should_train"] = True
        results["reason"] = f"Training triggered: {new_samples_count} new samples, conditions favorable"
        
        # Start the training process
        training_run_id = await _start_training_run(new_samples_count)
        results["training_run_id"] = training_run_id
        
        logger.info(f"Training triggered: {training_run_id}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error checking training trigger: {e}")
        results["reason"] = f"Error: {str(e)}"
        return results


def _check_time_preference(preference: str) -> Dict[str, Any]:
    """
    Check if the current time matches the training time preference.
    
    This function acts like a scheduling assistant, checking whether
    the current time is appropriate for resource-intensive training
    based on the configured preferences.
    """
    now = datetime.now()
    current_hour = now.hour
    current_weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    if preference == "any":
        return {"allowed": True, "reason": "Any time allowed"}
    
    elif preference == "night":
        # Night time: 10 PM to 6 AM
        if current_hour >= 22 or current_hour <= 6:
            return {"allowed": True, "reason": "Night time hours"}
        else:
            return {"allowed": False, "reason": f"Not night time (current hour: {current_hour})"}
    
    elif preference == "day":
        # Day time: 8 AM to 6 PM
        if 8 <= current_hour <= 18:
            return {"allowed": True, "reason": "Day time hours"}
        else:
            return {"allowed": False, "reason": f"Not day time (current hour: {current_hour})"}
    
    elif preference == "weekend":
        # Weekend: Saturday (5) and Sunday (6)
        if current_weekday >= 5:
            return {"allowed": True, "reason": "Weekend day"}
        else:
            return {"allowed": False, "reason": f"Not weekend (current day: {current_weekday})"}
    
    else:
        return {"allowed": False, "reason": f"Unknown time preference: {preference}"}


async def _check_training_resources() -> Dict[str, Any]:
    """
    Check if system resources are sufficient for training.
    
    This function acts like a resource manager, ensuring that the system
    has enough memory, CPU, and disk space available for the intensive
    training process without affecting other operations.
    """
    try:
        import psutil
        
        # Check memory (need at least 2GB available)
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        
        if available_memory_gb < 2.0:
            return {
                "available": False,
                "reason": f"Insufficient memory: {available_memory_gb:.1f}GB available, need 2GB+"
            }
        
        # Check CPU usage (should be below 70%)
        cpu_percent = psutil.cpu_percent(interval=5)
        if cpu_percent > 70:
            return {
                "available": False,
                "reason": f"High CPU usage: {cpu_percent}%, should be below 70%"
            }
        
        # Check disk space (need at least 1GB free)
        disk = psutil.disk_usage('/')
        available_disk_gb = disk.free / (1024**3)
        
        if available_disk_gb < 1.0:
            return {
                "available": False,
                "reason": f"Insufficient disk space: {available_disk_gb:.1f}GB available, need 1GB+"
            }
        
        return {
            "available": True,
            "reason": f"Resources available: {available_memory_gb:.1f}GB RAM, {cpu_percent}% CPU, {available_disk_gb:.1f}GB disk"
        }
        
    except ImportError:
        # If psutil is not available, assume resources are available
        return {"available": True, "reason": "Resource checking not available"}


async def _start_training_run(sample_count: int) -> str:
    """
    Start a new training run with the accumulated samples.
    
    This function acts like a training instructor, setting up a new
    training session with all the necessary parameters and starting
    the training process in the background.
    """
    try:
        training_service = get_training_service()
        
        # Create training configuration
        config = {
            "run_name": f"Automated Training {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "model_type": "random_forest",
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 15,
            "train_test_split": 0.8,
            "use_human_labels_only": False,
            "min_samples_per_class": 5
        }
        
        # Start training in background
        asyncio.create_task(training_service.start_training(config))
        
        logger.info(f"Started training run with {sample_count} samples")
        return config["run_name"]
        
    except Exception as e:
        logger.error(f"Failed to start training run: {e}")
        raise


# =============================================================================
# MAINTENANCE TASKS
# =============================================================================

async def cleanup_old_logs(days_to_keep: int = 30) -> Dict[str, Any]:
    """
    Remove old log files to save disk space.
    
    This function acts like a digital janitor, cleaning up old log files
    that are no longer needed while preserving recent logs for debugging
    and analysis purposes.
    """
    logger.info(f"Starting log cleanup: keeping logs from last {days_to_keep} days")
    
    results = {
        "files_found": 0,
        "files_removed": 0,
        "space_freed_bytes": 0,
        "errors": []
    }
    
    try:
        settings = get_settings()
        log_dir = Path(settings.log_file).parent
        
        if not log_dir.exists():
            logger.info(f"Log directory does not exist: {log_dir}")
            return results
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Find log files to remove
        log_patterns = ["*.log", "*.log.*", "*.out", "*.err"]
        
        for pattern in log_patterns:
            for log_file in log_dir.glob(pattern):
                try:
                    results["files_found"] += 1
                    
                    # Check file modification time
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        
                        results["files_removed"] += 1
                        results["space_freed_bytes"] += file_size
                        
                        logger.debug(f"Removed old log file: {log_file}")
                        
                except Exception as e:
                    logger.error(f"Error removing log file {log_file}: {e}")
                    results["errors"].append(f"Error with {log_file}: {str(e)}")
        
        logger.info(f"Log cleanup completed: removed {results['files_removed']} files, "
                   f"freed {format_bytes(results['space_freed_bytes'])}")
        
        return results
        
    except Exception as e:
        logger.error(f"Log cleanup failed: {e}")
        results["errors"].append(f"Task failed: {str(e)}")
        return results


async def cleanup_temp_files(hours_to_keep: int = 24) -> Dict[str, Any]:
    """
    Remove temporary files and cache to free up disk space.
    
    This function acts like a cache cleaner, removing temporary files
    that accumulate during image processing and model operations.
    """
    logger.info(f"Starting temp file cleanup: keeping files from last {hours_to_keep} hours")
    
    results = {
        "files_found": 0,
        "files_removed": 0,
        "space_freed_bytes": 0,
        "errors": []
    }
    
    try:
        settings = get_settings()
        
        # Directories to clean
        temp_dirs = [
            Path(settings.results_path) / "thumbnails",
            Path(settings.results_path) / "exports",
            Path("/tmp"),  # System temp directory
        ]
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)
        
        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
                
            # Clean temporary files
            temp_patterns = ["tmp_*", "*.tmp", "*_temp", "*.cache"]
            
            for pattern in temp_patterns:
                for temp_file in temp_dir.glob(pattern):
                    try:
                        results["files_found"] += 1
                        
                        # Check file modification time
                        file_mtime = datetime.fromtimestamp(temp_file.stat().st_mtime)
                        
                        if file_mtime < cutoff_time:
                            if temp_file.is_file():
                                file_size = temp_file.stat().st_size
                                temp_file.unlink()
                                
                                results["files_removed"] += 1
                                results["space_freed_bytes"] += file_size
                                
                            elif temp_file.is_dir():
                                dir_size = sum(f.stat().st_size for f in temp_file.rglob('*') if f.is_file())
                                shutil.rmtree(temp_file)
                                
                                results["files_removed"] += 1
                                results["space_freed_bytes"] += dir_size
                            
                            logger.debug(f"Removed temp file/dir: {temp_file}")
                            
                    except Exception as e:
                        logger.error(f"Error removing temp file {temp_file}: {e}")
                        results["errors"].append(f"Error with {temp_file}: {str(e)}")
        
        logger.info(f"Temp file cleanup completed: removed {results['files_removed']} items, "
                   f"freed {format_bytes(results['space_freed_bytes'])}")
        
        return results
        
    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}")
        results["errors"].append(f"Task failed: {str(e)}")
        return results


async def maintain_database() -> Dict[str, Any]:
    """
    Perform database maintenance tasks to optimize performance.
    
    This function acts like a database administrator, running maintenance
    operations to keep the database performing efficiently as it grows
    over time with accumulated data.
    """
    logger.info("Starting database maintenance")
    
    results = {
        "vacuum_completed": False,
        "stats_updated": False,
        "size_before_bytes": 0,
        "size_after_bytes": 0,
        "maintenance_time_seconds": 0,
        "errors": []
    }
    
    start_time = datetime.now()
    
    try:
        settings = get_settings()
        
        # Check database size before maintenance
        if "sqlite" in settings.database_url.lower():
            results.update(await _maintain_sqlite_database())
        else:
            results.update(await _maintain_postgresql_database())
        
        # Calculate maintenance time
        results["maintenance_time_seconds"] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Database maintenance completed in {results['maintenance_time_seconds']:.1f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Database maintenance failed: {e}")
        results["errors"].append(f"Task failed: {str(e)}")
        results["maintenance_time_seconds"] = (datetime.now() - start_time).total_seconds()
        return results


async def _maintain_sqlite_database() -> Dict[str, Any]:
    """Perform SQLite-specific maintenance operations."""
    results = {}
    
    try:
        with db_manager.get_session_context() as db:
            # Get database size before
            db_file = get_settings().database_url.replace("sqlite:///", "")
            if os.path.exists(db_file):
                results["size_before_bytes"] = os.path.getsize(db_file)
            
            # Run VACUUM to optimize database
            db.execute("VACUUM")
            results["vacuum_completed"] = True
            
            # Update statistics
            db.execute("ANALYZE")
            results["stats_updated"] = True
            
            db.commit()
            
            # Get database size after
            if os.path.exists(db_file):
                results["size_after_bytes"] = os.path.getsize(db_file)
            
            logger.info("SQLite database maintenance completed")
            
    except Exception as e:
        logger.error(f"SQLite maintenance failed: {e}")
        results["errors"] = [str(e)]
    
    return results


async def _maintain_postgresql_database() -> Dict[str, Any]:
    """Perform PostgreSQL-specific maintenance operations."""
    results = {}
    
    try:
        with db_manager.get_session_context() as db:
            # Run VACUUM ANALYZE on all tables
            # Note: VACUUM requires special handling in PostgreSQL
            db.execute("VACUUM ANALYZE")
            results["vacuum_completed"] = True
            results["stats_updated"] = True
            
            logger.info("PostgreSQL database maintenance completed")
            
    except Exception as e:
        logger.error(f"PostgreSQL maintenance failed: {e}")
        results["errors"] = [str(e)]
    
    return results


# =============================================================================
# MONITORING TASKS
# =============================================================================

async def system_health_check() -> Dict[str, Any]:
    """
    Perform comprehensive system health check and send alerts if needed.
    
    This function acts like a system doctor, examining all the vital signs
    of the VisionFlow AI system and reporting any issues that need attention.
    """
    logger.debug("Running system health check")
    
    health_status = {
        "overall_status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "alerts": [],
        "metrics": {}
    }
    
    try:
        # Check database health
        db_health = db_manager.health_check()
        health_status["checks"]["database"] = db_health
        
        if db_health["status"] != "healthy":
            health_status["overall_status"] = "degraded"
            health_status["alerts"].append(f"Database unhealthy: {db_health.get('error', 'Unknown error')}")
        
        # Check system resources
        resource_status = await _check_system_resources()
        health_status["checks"]["resources"] = resource_status
        health_status["metrics"].update(resource_status.get("metrics", {}))
        
        if not resource_status["healthy"]:
            health_status["overall_status"] = "degraded"
            health_status["alerts"].extend(resource_status.get("alerts", []))
        
        # Check processing queue
        queue_status = await _check_processing_queue()
        health_status["checks"]["processing_queue"] = queue_status
        
        if not queue_status["healthy"]:
            health_status["overall_status"] = "degraded"
            health_status["alerts"].extend(queue_status.get("alerts", []))
        
        # Send alerts if there are any issues
        if health_status["alerts"]:
            await _send_health_alerts(health_status)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_status["overall_status"] = "unhealthy"
        health_status["alerts"].append(f"Health check failed: {str(e)}")
        return health_status


async def _check_system_resources() -> Dict[str, Any]:
    """Check system resource usage and availability."""
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        alerts = []
        
        # Check for resource issues
        if cpu_percent > 85:
            alerts.append(f"High CPU usage: {cpu_percent}%")
        
        if memory.percent > 90:
            alerts.append(f"High memory usage: {memory.percent}%")
        
        if (disk.used / disk.total) * 100 > 90:
            alerts.append(f"Low disk space: {(disk.used / disk.total) * 100:.1f}% used")
        
        return {
            "healthy": len(alerts) == 0,
            "alerts": alerts,
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            }
        }
        
    except ImportError:
        return {
            "healthy": True,
            "alerts": ["Resource monitoring not available (psutil not installed)"],
            "metrics": {}
        }


async def _check_processing_queue() -> Dict[str, Any]:
    """Check the image processing queue for backlogs or stuck items."""
    try:
        with db_manager.get_session_context() as db:
            # Count images in different states
            pending_count = db.query(ImageRecord).filter(
                ImageRecord.status == ProcessingStatus.UPLOADED
            ).count()
            
            processing_count = db.query(ImageRecord).filter(
                ImageRecord.status.in_([
                    ProcessingStatus.SEGMENTING,
                    ProcessingStatus.CLASSIFYING,
                    ProcessingStatus.TRAINING
                ])
            ).count()
            
            # Check for stuck items (processing for more than 1 hour)
            stuck_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            stuck_count = db.query(ImageRecord).filter(
                ImageRecord.status.in_([
                    ProcessingStatus.SEGMENTING,
                    ProcessingStatus.CLASSIFYING,
                    ProcessingStatus.TRAINING
                ]),
                ImageRecord.processing_started_at < stuck_cutoff
            ).count()
            
            alerts = []
            
            if pending_count > 100:
                alerts.append(f"Large processing queue: {pending_count} pending images")
            
            if stuck_count > 0:
                alerts.append(f"Stuck processing items: {stuck_count} images stuck for >1 hour")
            
            return {
                "healthy": len(alerts) == 0,
                "alerts": alerts,
                "metrics": {
                    "pending_count": pending_count,
                    "processing_count": processing_count,
                    "stuck_count": stuck_count
                }
            }
            
    except Exception as e:
        return {
            "healthy": False,
            "alerts": [f"Queue check failed: {str(e)}"],
            "metrics": {}
        }


async def _send_health_alerts(health_status: Dict[str, Any]):
    """Send health alerts via configured channels."""
    try:
        settings = get_settings()
        
        # Send webhook alert if configured
        if settings.alert_webhook_url:
            await _send_webhook_alert(health_status, settings.alert_webhook_url)
        
        # Send email alert if configured
        if settings.alert_email_enabled and settings.alert_email_recipients:
            await _send_email_alert(health_status, settings.alert_email_recipients)
        
    except Exception as e:
        logger.error(f"Failed to send health alerts: {e}")


async def _send_webhook_alert(health_status: Dict[str, Any], webhook_url: str):
    """Send alert to webhook (e.g., Slack, Discord)."""
    try:
        import aiohttp
        
        message = {
            "text": f"🚨 VisionFlow AI Health Alert",
            "attachments": [{
                "color": "warning" if health_status["overall_status"] == "degraded" else "danger",
                "fields": [
                    {"title": "Status", "value": health_status["overall_status"], "short": True},
                    {"title": "Time", "value": health_status["timestamp"], "short": True},
                    {"title": "Alerts", "value": "\n".join(health_status["alerts"]), "short": False}
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=message, timeout=10) as response:
                if response.status == 200:
                    logger.info("Health alert sent via webhook")
                else:
                    logger.error(f"Webhook alert failed: {response.status}")
                    
    except Exception as e:
        logger.error(f"Failed to send webhook alert: {e}")


async def generate_daily_report(webhook_url: str = None, email_recipients: List[str] = None) -> Dict[str, Any]:
    """
    Generate and send a comprehensive daily report.
    
    This function acts like a daily news reporter, gathering information
    about system performance, processing statistics, and overall health
    to create a comprehensive report for administrators.
    """
    logger.info("Generating daily report")
    
    report_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {},
        "statistics": {},
        "alerts": [],
        "recommendations": []
    }
    
    try:
        # Get processing statistics for the last 24 hours
        with db_manager.get_session_context() as db:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            
            # Image processing stats
            total_processed = db.query(ImageRecord).filter(
                ImageRecord.created_at >= yesterday
            ).count()
            
            successful = db.query(ImageRecord).filter(
                ImageRecord.created_at >= yesterday,
                ImageRecord.status == ProcessingStatus.COMPLETED
            ).count()
            
            failed = db.query(ImageRecord).filter(
                ImageRecord.created_at >= yesterday,
                ImageRecord.status == ProcessingStatus.FAILED
            ).count()
            
            # Training stats
            training_runs = db.query(TrainingRun).filter(
                TrainingRun.created_at >= yesterday
            ).count()
            
            report_data["statistics"] = {
                "images_processed": total_processed,
                "images_successful": successful,
                "images_failed": failed,
                "success_rate_percent": (successful / max(total_processed, 1)) * 100,
                "training_runs": training_runs
            }
        
        # System resource summary
        resource_status = await _check_system_resources()
        report_data["statistics"]["resources"] = resource_status.get("metrics", {})
        
        # Generate summary and recommendations
        report_data["summary"] = _generate_report_summary(report_data["statistics"])
        report_data["recommendations"] = _generate_recommendations(report_data["statistics"])
        
        # Send the report
        if webhook_url:
            await _send_report_webhook(report_data, webhook_url)
        
        if email_recipients:
            await _send_report_email(report_data, email_recipients)
        
        logger.info("Daily report generated and sent")
        
        return report_data
        
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        report_data["error"] = str(e)
        return report_data


def _generate_report_summary(stats: Dict[str, Any]) -> str:
    """Generate a human-readable summary of the day's activities."""
    processed = stats.get("images_processed", 0)
    success_rate = stats.get("success_rate_percent", 0)
    training_runs = stats.get("training_runs", 0)
    
    summary = f"Processed {processed} images with {success_rate:.1f}% success rate. "
    
    if training_runs > 0:
        summary += f"Completed {training_runs} training run(s). "
    
    if success_rate < 80:
        summary += "Success rate is below normal levels. "
    
    if processed == 0:
        summary += "No images were processed today. "
    
    return summary


def _generate_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on system statistics."""
    recommendations = []
    
    success_rate = stats.get("success_rate_percent", 100)
    if success_rate < 80:
        recommendations.append("Investigate failed image processing to improve success rate")
    
    failed_count = stats.get("images_failed", 0)
    if failed_count > 10:
        recommendations.append("High number of failed images - check error logs")
    
    cpu_percent = stats.get("resources", {}).get("cpu_percent", 0)
    if cpu_percent > 70:
        recommendations.append("High CPU usage detected - consider scaling resources")
    
    memory_percent = stats.get("resources", {}).get("memory_percent", 0)
    if memory_percent > 80:
        recommendations.append("High memory usage - monitor for memory leaks")
    
    disk_percent = stats.get("resources", {}).get("disk_percent", 0)
    if disk_percent > 85:
        recommendations.append("Low disk space - schedule cleanup or add storage")
    
    if not recommendations:
        recommendations.append("System is performing well - no immediate action needed")
    
    return recommendations


async def _send_report_webhook(report_data: Dict[str, Any], webhook_url: str):
    """Send daily report via webhook."""
    try:
        import aiohttp
        
        message = {
            "text": f"📊 VisionFlow AI Daily Report - {report_data['date']}",
            "attachments": [{
                "color": "good",
                "fields": [
                    {"title": "Images Processed", "value": str(report_data["statistics"]["images_processed"]), "short": True},
                    {"title": "Success Rate", "value": f"{report_data['statistics']['success_rate_percent']:.1f}%", "short": True},
                    {"title": "Training Runs", "value": str(report_data["statistics"]["training_runs"]), "short": True},
                    {"title": "Summary", "value": report_data["summary"], "short": False}
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=message, timeout=10) as response:
                if response.status == 200:
                    logger.info("Daily report sent via webhook")
                else:
                    logger.error(f"Report webhook failed: {response.status}")
                    
    except Exception as e:
        logger.error(f"Failed to send report webhook: {e}")