"""
VisionFlow AI - API Endpoints Tests
===================================

This module contains comprehensive tests for all API endpoints.
Think of these tests as quality control inspectors that check every
"door" (endpoint) in your API to make sure it opens correctly,
handles visitors properly, and responds appropriately to different
types of requests.

These tests cover:
- Image upload and processing endpoints
- Results viewing and export endpoints
- Training management endpoints
- Health check and monitoring endpoints
- Error handling and edge cases
- Authentication and authorization (when implemented)
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
import io

# FastAPI testing imports
from fastapi.testclient import TestClient
from fastapi import status
import httpx

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.main import app
from backend.models.database_models import ImageRecord, ProcessingStatus, TrainingRun, TrainingStatus
from backend.database import get_db


# =============================================================================
# TEST FIXTURES AND SETUP
# =============================================================================

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_db_session():
    """Create a mock database session for testing."""
    return Mock()


@pytest.fixture
def sample_image_file():
    """Create a sample image file for upload testing."""
    from PIL import Image
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='blue')
    
    # Save to bytes buffer
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    return ("test_image.jpg", img_buffer, "image/jpeg")


@pytest.fixture
def mock_image_record():
    """Create a mock image record for testing."""
    mock_record = Mock(spec=ImageRecord)
    mock_record.id = "test-image-id"
    mock_record.filename = "test_image.jpg"
    mock_record.file_path = "/test/path/test_image.jpg"
    mock_record.file_size = 1024
    mock_record.mime_type = "image/jpeg"
    mock_record.width = 100
    mock_record.height = 100
    mock_record.channels = 3
    mock_record.status = ProcessingStatus.COMPLETED
    mock_record.processing_config = {}
    mock_record.segments = []
    mock_record.classifications = []
    mock_record.created_at = "2023-01-01T00:00:00Z"
    mock_record.processing_started_at = None
    mock_record.processing_completed_at = None
    return mock_record


@pytest.fixture
def mock_training_run():
    """Create a mock training run for testing."""
    mock_run = Mock(spec=TrainingRun)
    mock_run.id = "test-training-id"
    mock_run.run_name = "Test Training Run"
    mock_run.status = TrainingStatus.COMPLETED
    mock_run.model_type = "random_forest"
    mock_run.config = {"batch_size": 32}
    mock_run.num_samples = 100
    mock_run.train_test_split = 0.8
    mock_run.current_epoch = 10
    mock_run.total_epochs = 10
    mock_run.train_accuracy = 0.95
    mock_run.validation_accuracy = 0.88
    mock_run.created_at = "2023-01-01T00:00:00Z"
    mock_run.training_started_at = "2023-01-01T01:00:00Z"
    mock_run.training_completed_at = "2023-01-01T02:00:00Z"
    return mock_run


# =============================================================================
# IMAGE ENDPOINTS TESTS
# =============================================================================

class TestImageEndpoints:
    """Test suite for image-related API endpoints."""
    
    def test_upload_image_success(self, client, sample_image_file):
        """Test successful image upload."""
        filename, file_buffer, content_type = sample_image_file
        
        with patch('backend.api.endpoints.images.get_db') as mock_get_db:
            with patch('backend.api.endpoints.images.generate_unique_filename') as mock_filename:
                with patch('backend.api.endpoints.images.get_image_metadata') as mock_metadata:
                    with patch('backend.api.endpoints.images.process_image_pipeline') as mock_process:
                        
                        # Setup mocks
                        mock_db = Mock()
                        mock_get_db.return_value = mock_db
                        mock_filename.return_value = "unique_test_image.jpg"
                        mock_metadata.return_value = {"width": 100, "height": 100, "channels": 3}
                        
                        # Mock database operations
                        mock_record = Mock()
                        mock_record.id = "test-image-id"
                        mock_db.add.return_value = None
                        mock_db.commit.return_value = None
                        mock_db.refresh.return_value = None
                        
                        # Mock the context manager for the database session
                        mock_db.__enter__ = Mock(return_value=mock_db)
                        mock_db.__exit__ = Mock(return_value=None)
                        
                        with patch('builtins.open', create=True) as mock_open:
                            mock_open.return_value.__enter__.return_value.write = Mock()
                            
                            with patch('backend.api.endpoints.images.ImageRecord') as mock_image_record_class:
                                mock_image_record_class.return_value = mock_record
                                
                                # Make the request
                                response = client.post(
                                    "/api/v1/images/upload",
                                    files={"image": (filename, file_buffer, content_type)},
                                    data={"user_id": "test_user"}
                                )
                        
                        # Verify response
                        assert response.status_code == status.HTTP_200_OK
                        data = response.json()
                        assert data["success"] is True
                        assert data["filename"] == filename
                        assert "image_id" in data
    
    def test_upload_image_invalid_file_type(self, client):
        """Test image upload with invalid file type."""
        # Create a text file instead of an image
        text_file = ("test.txt", io.StringIO("This is not an image"), "text/plain")
        
        response = client.post(
            "/api/v1/images/upload",
            files={"image": text_file}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "must be an image" in response.json()["detail"]
    
    def test_upload_image_file_too_large(self, client):
        """Test image upload with file that's too large."""
        # Create a large dummy file (simulate large image)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        large_file = ("large_image.jpg", io.BytesIO(large_content), "image/jpeg")
        
        response = client.post(
            "/api/v1/images/upload",
            files={"image": large_file}
        )
        
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "too large" in response.json()["detail"]
    
    def test_get_processing_status_success(self, client, mock_image_record):
        """Test getting processing status for an image."""
        with patch('backend.api.endpoints.images.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = mock_image_record
            
            response = client.get("/api/v1/images/status/test-image-id")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["image_id"] == "test-image-id"
            assert data["status"] == ProcessingStatus.COMPLETED.value
            assert "progress_percentage" in data
    
    def test_get_processing_status_not_found(self, client):
        """Test getting processing status for non-existent image."""
        with patch('backend.api.endpoints.images.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            response = client.get("/api/v1/images/status/nonexistent-id")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_list_images_success(self, client, mock_image_record):
        """Test listing images with pagination."""
        with patch('backend.api.endpoints.images.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock query chain
            mock_query = Mock()
            mock_query.count.return_value = 1
            mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_image_record]
            mock_db.query.return_value = mock_query
            
            response = client.get("/api/v1/images/list?page=1&page_size=20")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_count"] == 1
            assert len(data["images"]) == 1
            assert data["images"][0]["id"] == "test-image-id"
    
    def test_list_images_with_filters(self, client, mock_image_record):
        """Test listing images with status filter."""
        with patch('backend.api.endpoints.images.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock filtered query
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.count.return_value = 1
            mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_image_record]
            mock_db.query.return_value = mock_query
            
            response = client.get("/api/v1/images/list?status=completed")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_count"] == 1
    
    def test_download_image_success(self, client, mock_image_record):
        """Test downloading an image file."""
        with patch('backend.api.endpoints.images.get_db') as mock_get_db:
            with patch('os.path.exists') as mock_exists:
                with patch('backend.api.endpoints.images.FileResponse') as mock_file_response:
                    
                    mock_db = Mock()
                    mock_get_db.return_value = mock_db
                    mock_db.query.return_value.filter.return_value.first.return_value = mock_image_record
                    mock_exists.return_value = True
                    mock_file_response.return_value = Mock()
                    
                    response = client.get("/api/v1/images/download/test-image-id")
                    
                    # Verify FileResponse was called
                    mock_file_response.assert_called_once()
    
    def test_delete_image_success(self, client, mock_image_record):
        """Test deleting an image and its data."""
        with patch('backend.api.endpoints.images.get_db') as mock_get_db:
            with patch('os.path.exists') as mock_exists:
                with patch('os.remove') as mock_remove:
                    with patch('backend.api.endpoints.images.get_storage_service') as mock_storage:
                        
                        mock_db = Mock()
                        mock_get_db.return_value = mock_db
                        mock_db.query.return_value.filter.return_value.first.return_value = mock_image_record
                        mock_exists.return_value = True
                        
                        mock_storage_service = AsyncMock()
                        mock_storage.return_value = mock_storage_service
                        
                        response = client.delete("/api/v1/images/test-image-id")
                        
                        assert response.status_code == status.HTTP_200_OK
                        data = response.json()
                        assert data["success"] is True
    
    def test_cancel_processing_success(self, client, mock_image_record):
        """Test canceling image processing."""
        # Set image to processing state
        mock_image_record.status = ProcessingStatus.SEGMENTING
        
        with patch('backend.api.endpoints.images.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = mock_image_record
            
            response = client.post("/api/v1/images/test-image-id/cancel")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert ProcessingStatus.CANCELLED.value in data["status"]


# =============================================================================
# RESULTS ENDPOINTS TESTS
# =============================================================================

class TestResultsEndpoints:
    """Test suite for results-related API endpoints."""
    
    def test_get_result_summary_success(self, client, mock_image_record):
        """Test getting result summary for an image."""
        with patch('backend.api.endpoints.results.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = mock_image_record
            
            # Mock top classifications query
            mock_db.query.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
            
            response = client.get("/api/v1/results/summary/test-image-id")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["image_id"] == "test-image-id"
            assert data["filename"] == "test_image.jpg"
            assert data["status"] == ProcessingStatus.COMPLETED.value
    
    def test_get_detailed_results_success(self, client, mock_image_record):
        """Test getting detailed results for an image."""
        with patch('backend.api.endpoints.results.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = mock_image_record
            
            response = client.get("/api/v1/results/detailed/test-image-id")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["image_id"] == "test-image-id"
            assert "segments" in data
            assert "statistics" in data
            assert "image_url" in data
    
    def test_get_detailed_results_not_completed(self, client, mock_image_record):
        """Test getting detailed results for incomplete processing."""
        mock_image_record.status = ProcessingStatus.SEGMENTING
        
        with patch('backend.api.endpoints.results.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = mock_image_record
            
            response = client.get("/api/v1/results/detailed/test-image-id")
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_submit_feedback_success(self, client):
        """Test submitting human feedback on classification."""
        feedback_data = {
            "classification_id": "test-classification-id",
            "correct_label": "corrected_apple",
            "confidence": 0.95,
            "notes": "This is definitely an apple"
        }
        
        with patch('backend.api.endpoints.results.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock classification record
            mock_classification = Mock()
            mock_db.query.return_value.filter.return_value.first.return_value = mock_classification
            
            response = client.post("/api/v1/results/feedback", json=feedback_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["corrected_label"] == "corrected_apple"
    
    def test_export_results_json(self, client, mock_image_record):
        """Test exporting results in JSON format."""
        with patch('backend.api.endpoints.results.get_db') as mock_get_db:
            with patch('backend.api.endpoints.results.get_storage_service') as mock_storage:
                
                mock_db = Mock()
                mock_get_db.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = mock_image_record
                
                mock_storage_service = AsyncMock()
                mock_storage_service.export_to_json.return_value = "/test/export.json"
                mock_storage.return_value = mock_storage_service
                
                response = client.get("/api/v1/results/export/test-image-id?format=json")
                
                # Should return a file response
                mock_storage_service.export_to_json.assert_called_once()
    
    def test_get_analytics_overview(self, client):
        """Test getting analytics overview."""
        with patch('backend.api.endpoints.results.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock database queries
            mock_db.query.return_value.filter.return_value.count.return_value = 10
            mock_db.query.return_value.filter.return_value.first.return_value = Mock(avg_time=30.5)
            mock_db.query.return_value.join.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
            mock_db.query.return_value.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = []
            
            response = client.get("/api/v1/results/analytics/overview")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "summary" in data
            assert "top_classifications" in data
            assert "daily_processing_counts" in data


# =============================================================================
# TRAINING ENDPOINTS TESTS
# =============================================================================

class TestTrainingEndpoints:
    """Test suite for training-related API endpoints."""
    
    def test_start_training_success(self, client):
        """Test starting a new training run."""
        training_config = {
            "run_name": "Test Training",
            "model_type": "random_forest",
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 10
        }
        
        with patch('backend.api.endpoints.training.get_db') as mock_get_db:
            with patch('backend.api.endpoints.training.get_training_service') as mock_training:
                
                mock_db = Mock()
                mock_get_db.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = None  # No active training
                
                mock_training_service = AsyncMock()
                mock_training_service.get_dataset_info.return_value = {"ready_for_training": True, "total_samples": 150}
                mock_training.return_value = mock_training_service
                
                # Mock TrainingRun creation
                with patch('backend.api.endpoints.training.TrainingRun') as mock_training_run_class:
                    mock_training_run = Mock()
                    mock_training_run.id = "test-training-id"
                    mock_training_run_class.return_value = mock_training_run
                    
                    response = client.post("/api/v1/training/start", json=training_config)
                    
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert data["run_name"] == "Test Training"
                    assert data["status"] == TrainingStatus.PENDING.value
    
    def test_start_training_already_in_progress(self, client, mock_training_run):
        """Test starting training when another run is already in progress."""
        mock_training_run.status = TrainingStatus.IN_PROGRESS
        
        with patch('backend.api.endpoints.training.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = mock_training_run
            
            training_config = {"run_name": "Test Training"}
            response = client.post("/api/v1/training/start", json=training_config)
            
            assert response.status_code == status.HTTP_409_CONFLICT
    
    def test_list_training_runs(self, client, mock_training_run):
        """Test listing training runs."""
        with patch('backend.api.endpoints.training.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock query chain
            mock_query = Mock()
            mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_training_run]
            mock_db.query.return_value = mock_query
            
            response = client.get("/api/v1/training/runs")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) >= 0  # Should return a list
    
    def test_get_training_run_details(self, client, mock_training_run):
        """Test getting detailed information about a training run."""
        with patch('backend.api.endpoints.training.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = mock_training_run
            
            response = client.get("/api/v1/training/runs/test-training-id")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["id"] == "test-training-id"
            assert data["run_name"] == "Test Training Run"
    
    def test_get_training_progress(self, client, mock_training_run):
        """Test getting training progress."""
        mock_training_run.status = TrainingStatus.IN_PROGRESS
        mock_training_run.current_epoch = 5
        mock_training_run.total_epochs = 10
        
        with patch('backend.api.endpoints.training.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = mock_training_run
            
            response = client.get("/api/v1/training/progress/test-training-id")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["training_run_id"] == "test-training-id"
            assert data["progress_percentage"] == 50.0  # 5/10 epochs
    
    def test_get_dataset_summary(self, client):
        """Test getting dataset summary."""
        with patch('backend.api.endpoints.training.get_training_service') as mock_training:
            mock_training_service = AsyncMock()
            mock_training_service.get_dataset_info.return_value = {
                "total_samples": 150,
                "samples_by_source": {"openai": 100, "human": 50},
                "samples_by_label": {"apple": 75, "orange": 75},
                "human_verified_samples": 50,
                "ready_for_training": True,
                "min_samples_needed": 100
            }
            mock_training.return_value = mock_training_service
            
            response = client.get("/api/v1/training/dataset")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_samples"] == 150
            assert data["ready_for_training"] is True
    
    def test_pause_training(self, client, mock_training_run):
        """Test pausing a training run."""
        mock_training_run.status = TrainingStatus.IN_PROGRESS
        
        with patch('backend.api.endpoints.training.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = mock_training_run
            
            response = client.post("/api/v1/training/runs/test-training-id/pause")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True


# =============================================================================
# HEALTH ENDPOINTS TESTS
# =============================================================================

class TestHealthEndpoints:
    """Test suite for health check and monitoring endpoints."""
    
    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "VisionFlow AI"
    
    def test_detailed_health_check_healthy(self, client):
        """Test detailed health check when all systems are healthy."""
        with patch('backend.api.endpoints.health.db_manager') as mock_db:
            with patch('backend.api.endpoints.health.get_sam_service') as mock_sam:
                with patch('backend.api.endpoints.health.get_openai_service') as mock_openai:
                    
                    # Mock healthy responses
                    mock_db.health_check.return_value = {"status": "healthy"}
                    
                    mock_sam_service = AsyncMock()
                    mock_sam_service.health_check.return_value = {"status": "healthy"}
                    mock_sam.return_value = mock_sam_service
                    
                    mock_openai_service = AsyncMock()
                    mock_openai_service.test_connection.return_value = {"status": "success"}
                    mock_openai.return_value = mock_openai_service
                    
                    response = client.get("/api/v1/health/detailed")
                    
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert data["overall_status"] == "healthy"
                    assert "components" in data
                    assert "metrics" in data
    
    def test_detailed_health_check_degraded(self, client):
        """Test detailed health check when some systems are unhealthy."""
        with patch('backend.api.endpoints.health.db_manager') as mock_db:
            with patch('backend.api.endpoints.health.get_sam_service') as mock_sam:
                with patch('backend.api.endpoints.health.get_openai_service') as mock_openai:
                    
                    # Mock mixed health responses
                    mock_db.health_check.return_value = {"status": "healthy"}
                    
                    mock_sam_service = AsyncMock()
                    mock_sam_service.health_check.return_value = {"status": "unhealthy", "error": "Model not loaded"}
                    mock_sam.return_value = mock_sam_service
                    
                    mock_openai_service = AsyncMock()
                    mock_openai_service.test_connection.return_value = {"status": "success"}
                    mock_openai.return_value = mock_openai_service
                    
                    response = client.get("/api/v1/health/detailed")
                    
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert data["overall_status"] == "degraded"
    
    def test_database_health_check(self, client):
        """Test database-specific health check."""
        with patch('backend.api.endpoints.health.db_manager') as mock_db:
            with patch('backend.api.endpoints.health.get_table_counts') as mock_counts:
                with patch('backend.api.endpoints.health.get_processing_statistics') as mock_stats:
                    with patch('backend.api.endpoints.health.get_database_size') as mock_size:
                        
                        mock_db.health_check.return_value = {"status": "healthy"}
                        mock_counts.return_value = {"images": 10, "segments": 50}
                        mock_stats.return_value = {"total_processed": 10}
                        mock_size.return_value = {"size_mb": 100}
                        
                        response = client.get("/api/v1/health/database")
                        
                        assert response.status_code == status.HTTP_200_OK
                        data = response.json()
                        assert data["status"] == "healthy"
                        assert "table_counts" in data
    
    def test_get_recent_errors(self, client):
        """Test getting recent error logs."""
        with patch('backend.api.endpoints.health.get_db') as mock_get_db:
            mock_db = Mock()
            mock_get_db.return_value = mock_db
            
            # Mock error logs query
            mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
            mock_db.query.return_value.filter.return_value.group_by.return_value.all.return_value = []
            
            response = client.get("/api/v1/health/errors/recent")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "errors" in data
            assert "error_statistics" in data


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test suite for API error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent-endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    def test_422_validation_error(self, client):
        """Test validation error handling."""
        # Send invalid JSON to an endpoint that expects specific format
        invalid_data = {"invalid": "data"}
        
        response = client.post("/api/v1/training/start", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_internal_server_error_handling(self, client):
        """Test internal server error handling."""
        with patch('backend.api.endpoints.health.get_app_info') as mock_info:
            # Force an exception
            mock_info.side_effect = Exception("Simulated error")
            
            response = client.get("/api/v1/health/")
            
            # Should handle the error gracefully
            assert response.status_code >= 400


# =============================================================================
# AUTHENTICATION AND AUTHORIZATION TESTS
# =============================================================================

class TestAuthentication:
    """Test suite for authentication and authorization (when implemented)."""
    
    @pytest.mark.skip(reason="Authentication not yet implemented")
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication."""
        # This test would verify that protected endpoints require authentication
        response = client.get("/api/v1/admin/users")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.skip(reason="Authentication not yet implemented")
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/admin/users", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test suite for API performance characteristics."""
    
    def test_health_check_response_time(self, client):
        """Test that health check responds quickly."""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/health/")
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        assert (end_time - start_time) < 1.0  # Should respond in under 1 second
    
    def test_concurrent_requests(self, client):
        """Test handling multiple concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            start_time = time.time()
            response = client.get("/api/v1/health/")
            end_time = time.time()
            results.append({
                "status_code": response.status_code,
                "response_time": end_time - start_time
            })
        
        # Create multiple threads to simulate concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        assert len(results) == 5
        assert all(result["status_code"] == 200 for result in results)
        assert all(result["response_time"] < 2.0 for result in results)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])