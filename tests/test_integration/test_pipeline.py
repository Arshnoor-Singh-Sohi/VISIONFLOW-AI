"""
VisionFlow AI - Integration Pipeline Tests
==========================================

These tests verify the entire image processing pipeline works correctly
from image upload through segmentation, classification, and training.
Think of these as "end-to-end" tests that simulate real user workflows.
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import numpy as np

# Import the components we need to test
from backend.main import app
from backend.database import init_database, reset_database, db_manager
from backend.models.database_models import ImageRecord, ProcessingStatus
from backend.services.sam_service import SAMService, SegmentResult, SAMProcessingResult
from backend.services.openai_service import OpenAIService, ClassificationResult
from backend.config import get_settings
from fastapi.testclient import TestClient


# =============================================================================
# TEST FIXTURES AND SETUP
# =============================================================================

@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing environment."""
    with patch('backend.config.get_settings') as mock_settings:
        settings = Mock()
        settings.debug = True
        settings.database_url = "sqlite:///:memory:"
        settings.upload_path = "/tmp/visionflow_test/uploads"
        settings.segments_path = "/tmp/visionflow_test/segments"
        settings.results_path = "/tmp/visionflow_test/results"
        settings.models_path = "/tmp/visionflow_test/models"
        settings.sam_service_url = "http://localhost:8001"
        settings.openai_api_key = "test-key-12345"
        settings.openai_model = "gpt-4-vision-preview"
        settings.max_file_size = 10 * 1024 * 1024  # 10MB
        settings.min_training_samples = 5
        settings.enable_training = True
        
        # Create test directories
        for path in [settings.upload_path, settings.segments_path, 
                    settings.results_path, settings.models_path]:
            os.makedirs(path, exist_ok=True)
        
        mock_settings.return_value = settings
        yield settings


@pytest.fixture(scope="session")
def test_database(test_settings):
    """Set up test database with clean state."""
    # Initialize test database
    init_database()
    
    yield
    
    # Cleanup after all tests
    try:
        db_manager.close()
    except:
        pass


@pytest.fixture
def test_client(test_database):
    """Create test client for API testing."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample test image for processing."""
    # Create a simple test image (red square on white background)
    image = Image.new('RGB', (400, 300), color='white')
    
    # Add a red square in the center
    pixels = image.load()
    for x in range(150, 250):
        for y in range(100, 200):
            pixels[x, y] = (255, 0, 0)  # Red color
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    image.save(temp_file.name, 'JPEG')
    
    yield temp_file.name
    
    # Cleanup
    try:
        os.unlink(temp_file.name)
    except:
        pass


@pytest.fixture
def mock_sam_response():
    """Mock response from SAM service."""
    return {
        'segments': [
            {
                'id': 0,
                'bbox': [150, 100, 100, 100],  # x, y, width, height
                'area': 10000,
                'confidence': 0.95,
                'segment_image_path': '/tmp/segment_0.jpg'
            },
            {
                'id': 1,
                'bbox': [0, 0, 400, 300],  # Background
                'area': 120000,
                'confidence': 0.85,
                'segment_image_path': '/tmp/segment_1.jpg'
            }
        ],
        'model_info': {
            'model_type': 'SAM',
            'version': '1.0'
        }
    }


@pytest.fixture
def mock_openai_response():
    """Mock response from OpenAI service."""
    return {
        'choices': [{
            'message': {
                'content': json.dumps({
                    'primary_item': 'red square',
                    'confidence': 0.92,
                    'alternatives': [
                        {'item': 'geometric shape', 'confidence': 0.78}
                    ],
                    'category': 'geometric object'
                })
            }
        }],
        'usage': {
            'total_tokens': 150
        },
        'model': 'gpt-4-vision-preview'
    }


# =============================================================================
# BASIC PIPELINE TESTS
# =============================================================================

class TestBasicPipeline:
    """Test the basic image processing pipeline functionality."""
    
    def test_health_check(self, test_client):
        """Test that the API is healthy and responding."""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "components" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint provides correct information."""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Welcome to VisionFlow AI"
        assert "version" in data
        assert "features" in data
        assert isinstance(data["features"], list)


class TestImageUpload:
    """Test image upload functionality and validation."""
    
    def test_successful_image_upload(self, test_client, sample_image):
        """Test uploading a valid image file."""
        with open(sample_image, 'rb') as f:
            files = {'image': ('test_image.jpg', f, 'image/jpeg')}
            data = {'config': '{"min_area": 1000}'}
            
            response = test_client.post("/api/v1/images/upload", files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["success"] is True
        assert "image_id" in result
        assert result["filename"] == "test_image.jpg"
        assert result["processing_status"] == "uploaded"
    
    def test_invalid_file_upload(self, test_client):
        """Test uploading an invalid file type."""
        # Create a text file instead of an image
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not an image")
            f.flush()
            
            with open(f.name, 'rb') as file_handle:
                files = {'image': ('test.txt', file_handle, 'text/plain')}
                response = test_client.post("/api/v1/images/upload", files=files)
        
        assert response.status_code == 422
        assert "must be an image" in response.json()["detail"]
        
        # Cleanup
        os.unlink(f.name)
    
    def test_empty_file_upload(self, test_client):
        """Test uploading an empty file."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            # File is empty
            pass
            
        with open(f.name, 'rb') as file_handle:
            files = {'image': ('empty.jpg', file_handle, 'image/jpeg')}
            response = test_client.post("/api/v1/images/upload", files=files)
        
        assert response.status_code == 422
        assert "Empty file" in response.json()["detail"]
        
        # Cleanup
        os.unlink(f.name)


class TestProcessingPipeline:
    """Test the complete image processing pipeline with mocked services."""
    
    @patch('backend.services.sam_service.SAMService._call_sam_service')
    @patch('backend.services.openai_service.OpenAIService._call_openai_vision')
    async def test_complete_pipeline_success(self, mock_openai, mock_sam, 
                                           test_client, sample_image, 
                                           mock_sam_response, mock_openai_response):
        """Test the complete pipeline from upload to completion."""
        
        # Setup mocks
        mock_sam.return_value = mock_sam_response
        mock_openai.return_value = Mock(**mock_openai_response)
        
        # Upload image
        with open(sample_image, 'rb') as f:
            files = {'image': ('test_pipeline.jpg', f, 'image/jpeg')}
            data = {'config': json.dumps({
                'min_area': 1000,
                'max_segments': 10,
                'confidence_threshold': 0.7,
                'enable_training': True
            })}
            
            response = test_client.post("/api/v1/images/upload", files=files, data=data)
        
        assert response.status_code == 200
        image_id = response.json()["image_id"]
        
        # Wait a moment for background processing to start
        await asyncio.sleep(0.1)
        
        # Check processing status
        status_response = test_client.get(f"/api/v1/images/status/{image_id}")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["image_id"] == image_id
        assert status_data["status"] in ["uploaded", "segmenting", "classifying", "training", "completed"]
        assert "progress_percentage" in status_data
        
        # Verify the image was stored in database
        with db_manager.get_session_context() as db:
            image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
            assert image_record is not None
            assert image_record.filename == "test_pipeline.jpg"
            assert image_record.status in [ProcessingStatus.UPLOADED, ProcessingStatus.SEGMENTING, 
                                         ProcessingStatus.CLASSIFYING, ProcessingStatus.TRAINING, 
                                         ProcessingStatus.COMPLETED]
    
    def test_image_list_functionality(self, test_client):
        """Test listing uploaded images with pagination."""
        response = test_client.get("/api/v1/images/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "images" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data
        assert isinstance(data["images"], list)
    
    def test_image_download(self, test_client, sample_image):
        """Test downloading an uploaded image."""
        # First upload an image
        with open(sample_image, 'rb') as f:
            files = {'image': ('download_test.jpg', f, 'image/jpeg')}
            response = test_client.post("/api/v1/images/upload", files=files)
        
        image_id = response.json()["image_id"]
        
        # Try to download it
        download_response = test_client.get(f"/api/v1/images/download/{image_id}")
        assert download_response.status_code == 200
        assert download_response.headers["content-type"].startswith("image/")


# =============================================================================
# SERVICE INTEGRATION TESTS
# =============================================================================

class TestServiceIntegration:
    """Test integration between different services."""
    
    @patch('aiohttp.ClientSession.post')
    async def test_sam_service_integration(self, mock_post, mock_sam_response):
        """Test integration with SAM service."""
        
        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_sam_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Test SAM service
        sam_service = SAMService()
        
        # Create a temporary image for testing
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name, 'JPEG')
            
            result = await sam_service.segment_image(f.name)
            
            assert isinstance(result, SAMProcessingResult)
            assert len(result.segments) == 2
            assert result.total_segments_found == 2
            
            # Check segment details
            first_segment = result.segments[0]
            assert isinstance(first_segment, SegmentResult)
            assert first_segment.bbox == (150, 100, 100, 100)
            assert first_segment.area == 10000
            
        # Cleanup
        os.unlink(f.name)
    
    @patch('backend.services.openai_service.AsyncOpenAI')
    async def test_openai_service_integration(self, mock_client, mock_openai_response):
        """Test integration with OpenAI service."""
        
        # Setup mock client
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(return_value=Mock(**mock_openai_response))
        mock_client.return_value = mock_instance
        
        # Test OpenAI service
        openai_service = OpenAIService()
        
        # Create a temporary image for testing
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(f.name, 'JPEG')
            
            result = await openai_service.classify_image_segment(f.name)
            
            assert isinstance(result, ClassificationResult)
            assert result.primary_label == "red square"
            assert result.confidence_score == 0.92
            assert result.model_used == "gpt-4-vision-preview"
            assert result.tokens_used == 150
            
        # Cleanup
        os.unlink(f.name)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling throughout the pipeline."""
    
    def test_nonexistent_image_status(self, test_client):
        """Test requesting status for non-existent image."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = test_client.get(f"/api/v1/images/status/{fake_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_invalid_image_id_format(self, test_client):
        """Test using invalid UUID format for image ID."""
        response = test_client.get("/api/v1/images/status/invalid-id")
        # This might return 422 for invalid UUID format, depending on FastAPI validation
        assert response.status_code in [404, 422]
    
    def test_file_too_large(self, test_client):
        """Test uploading a file that exceeds size limits."""
        # Create a large dummy file (simulate)
        large_data = b"x" * (11 * 1024 * 1024)  # 11MB, exceeds 10MB limit
        
        files = {'image': ('large_image.jpg', large_data, 'image/jpeg')}
        response = test_client.post("/api/v1/images/upload", files=files)
        
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance characteristics of the pipeline."""
    
    @pytest.mark.asyncio
    async def test_concurrent_uploads(self, test_client):
        """Test handling multiple concurrent image uploads."""
        
        async def upload_image(client, image_data, filename):
            """Helper function to upload a single image."""
            files = {'image': (filename, image_data, 'image/jpeg')}
            return client.post("/api/v1/images/upload", files=files)
        
        # Create test image data
        img = Image.new('RGB', (200, 200), color='green')
        img_data = []
        
        for i in range(5):  # Test with 5 concurrent uploads
            with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
                img.save(f.name, 'JPEG')
                with open(f.name, 'rb') as file_handle:
                    img_data.append(file_handle.read())
        
        # Perform concurrent uploads
        tasks = []
        for i, data in enumerate(img_data):
            tasks.append(upload_image(test_client, data, f"concurrent_{i}.jpg"))
        
        # Note: For a real async test, you'd use asyncio.gather with an async client
        # This is a simplified version for demonstration
        responses = [test_client.post("/api/v1/images/upload", 
                                    files={'image': (f"test_{i}.jpg", data, 'image/jpeg')})
                    for i, data in enumerate(img_data)]
        
        # Check that all uploads succeeded
        successful_uploads = sum(1 for r in responses if r.status_code == 200)
        assert successful_uploads >= 3  # At least 3 out of 5 should succeed


# =============================================================================
# TRAINING PIPELINE TESTS
# =============================================================================

class TestTrainingPipeline:
    """Test the model training pipeline functionality."""
    
    def test_dataset_info_endpoint(self, test_client):
        """Test getting information about the training dataset."""
        response = test_client.get("/api/v1/training/dataset")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_samples" in data
        assert "ready_for_training" in data
        assert "samples_by_source" in data
        assert "samples_by_label" in data
    
    def test_training_runs_list(self, test_client):
        """Test listing training runs."""
        response = test_client.get("/api/v1/training/runs")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # Might be empty for a fresh test database
    
    @patch('backend.services.training_service.TrainingService.get_dataset_info')
    async def test_training_trigger_conditions(self, mock_dataset_info, test_client):
        """Test conditions that trigger automatic training."""
        
        # Mock insufficient data scenario
        mock_dataset_info.return_value = {
            'total_samples': 3,
            'ready_for_training': False,
            'issues': ['Insufficient samples: 3 < 5']
        }
        
        response = test_client.get("/api/v1/training/dataset")
        data = response.json()
        assert data['ready_for_training'] is False
        assert 'Insufficient samples' in data['issues'][0]


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Test data validation and integrity throughout the pipeline."""
    
    def test_image_metadata_extraction(self, sample_image):
        """Test that image metadata is correctly extracted."""
        from backend.utils.image_processing import get_image_metadata
        
        metadata = get_image_metadata(sample_image)
        
        assert metadata['width'] == 400
        assert metadata['height'] == 300
        assert metadata['channels'] == 3
        assert metadata['format'] == 'JPEG'
        assert 'file_size' in metadata
        assert metadata['file_size'] > 0
    
    def test_image_validation(self, sample_image):
        """Test image file validation."""
        from backend.utils.image_processing import validate_image_file
        
        result = validate_image_file(sample_image)
        
        assert result['valid'] is True
        assert result['dimensions'] == (400, 300)
        assert result['mime_type'].startswith('image/')
        assert 'metadata' in result


# =============================================================================
# CLEANUP AND UTILITIES
# =============================================================================

class TestCleanupAndMaintenance:
    """Test cleanup and maintenance operations."""
    
    def test_image_deletion(self, test_client, sample_image):
        """Test deleting an image and its associated data."""
        # First upload an image
        with open(sample_image, 'rb') as f:
            files = {'image': ('to_delete.jpg', f, 'image/jpeg')}
            response = test_client.post("/api/v1/images/upload", files=files)
        
        image_id = response.json()["image_id"]
        
        # Delete the image
        delete_response = test_client.delete(f"/api/v1/images/{image_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["success"] is True
        
        # Verify it's gone
        status_response = test_client.get(f"/api/v1/images/status/{image_id}")
        assert status_response.status_code == 404


# =============================================================================
# BENCHMARK TESTS (OPTIONAL)
# =============================================================================

@pytest.mark.benchmark
class TestBenchmarks:
    """Performance benchmark tests (marked as optional)."""
    
    def test_image_upload_speed(self, test_client, sample_image, benchmark):
        """Benchmark image upload speed."""
        
        def upload_image():
            with open(sample_image, 'rb') as f:
                files = {'image': ('benchmark.jpg', f, 'image/jpeg')}
                return test_client.post("/api/v1/images/upload", files=files)
        
        result = benchmark(upload_image)
        assert result.status_code == 200


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    pytest.main([__file__, "-v", "--tb=short"])