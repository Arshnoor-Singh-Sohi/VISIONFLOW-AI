"""
VisionFlow AI - Backend Services Tests
======================================

This module contains comprehensive tests for all backend services.
Think of these tests as a quality assurance team that checks every
part of your system to make sure it works correctly and handles
edge cases gracefully.

These tests cover:
- OpenAI service integration and error handling
- SAM service communication and response parsing
- Storage service file operations and cleanup
- Training service model management and execution
- Database operations and data integrity
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from datetime import datetime, timezone

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.services.openai_service import OpenAIService, ClassificationResult
from backend.services.sam_service import SAMService, SegmentResult, SAMProcessingResult
from backend.services.storage_service import StorageService
from backend.services.training_service import TrainingService
from backend.config import get_settings
from backend.models.database_models import ImageRecord, ProcessingStatus


# =============================================================================
# TEST FIXTURES AND UTILITIES
# =============================================================================

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    # return Mock(
    #     openai_api_key="test-api-key",
    #     openai_model="gpt-4-vision-preview",
    #     openai_max_tokens=1000,
    #     openai_temperature=0.1,
    #     sam_service_url="http://localhost:8001",
    #     upload_path="./test_data/uploads",
    #     segments_path="./test_data/segments",
    #     results_path="./test_data/results",
    #     models_path="./test_data/models",
    #     max_file_size=10485760,
    #     log_file="./test_data/logs/test.log"
    # )
    """Mock settings for testing."""
    settings = Mock()
    settings.sam_service_url = "http://localhost:8001"
    settings.sam_model_type = "vit_h"  # String value, not Mock
    settings.sam_device = "cpu"         # String value, not Mock
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4-vision-preview"
    return settings


@pytest.fixture
def sample_image_path():
    """Create a sample image file for testing."""
    # Create a minimal test image
    from PIL import Image
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(f.name, 'JPEG')
        
        yield f.name
        
        # Cleanup
        try:
            os.unlink(f.name)
        except OSError:
            pass


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "primary_item": "apple",
                    "confidence": 0.95,
                    "alternatives": [
                        {"item": "red apple", "confidence": 0.85},
                        {"item": "fruit", "confidence": 0.80}
                    ],
                    "category": "fruit"
                })
            }
        }],
        "usage": {
            "total_tokens": 150
        },
        "model": "gpt-4-vision-preview"
    }


@pytest.fixture
def mock_sam_response():
    """Create a mock SAM service response."""
    return {
        "segments": [
            {
                "id": 0,
                "bbox": [10, 10, 50, 50],
                "area": 2500,
                "confidence": 0.9,
                "segment_image_path": "/test/segment_0.jpg"
            },
            {
                "id": 1,
                "bbox": [60, 60, 30, 30],
                "area": 900,
                "confidence": 0.8,
                "segment_image_path": "/test/segment_1.jpg"
            }
        ],
        "model_info": {
            "model_type": "vit_h",
            "device": "cpu"
        }
    }


# =============================================================================
# OPENAI SERVICE TESTS
# =============================================================================

class TestOpenAIService:
    """Test suite for OpenAI service functionality."""
    
    @pytest.fixture
    def openai_service(self, mock_settings):
        """Create OpenAI service instance for testing."""
        with patch('backend.services.openai_service.get_settings', return_value=mock_settings):
            return OpenAIService()
    
    @pytest.mark.asyncio
    async def test_classify_image_segment_success(self, openai_service, sample_image_path, mock_openai_response):
        """Test successful image classification."""
        # Mock the OpenAI client
        with patch.object(openai_service.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            # Configure mock to return our test response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = mock_openai_response["choices"][0]["message"]["content"]
            mock_response.usage.total_tokens = mock_openai_response["usage"]["total_tokens"]
            mock_response.model = mock_openai_response["model"]
            mock_create.return_value = mock_response
            
            # Test the classification
            result = await openai_service.classify_image_segment(sample_image_path)
            
            # Verify the result
            assert isinstance(result, ClassificationResult)
            assert result.primary_label == "apple"
            assert result.confidence_score == 0.95
            assert len(result.alternative_labels) == 2
            assert result.model_used == "gpt-4-vision-preview"
            assert result.tokens_used == 150
            
            # Verify the API was called correctly
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_classify_image_segment_file_not_found(self, openai_service):
        """Test classification with non-existent image file."""
        with pytest.raises(Exception) as exc_info:
            await openai_service.classify_image_segment("/nonexistent/file.jpg")
        
        assert "not found" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_classify_image_segment_invalid_json(self, openai_service, sample_image_path):
        """Test classification with invalid JSON response."""
        with patch.object(openai_service.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            # Configure mock to return invalid JSON
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "This is not valid JSON"
            mock_response.usage.total_tokens = 100
            mock_response.model = "gpt-4-vision-preview"
            mock_create.return_value = mock_response
            
            # Test the classification
            result = await openai_service.classify_image_segment(sample_image_path)
            
            # Should handle invalid JSON gracefully
            assert isinstance(result, ClassificationResult)
            assert result.primary_label == "This is not valid JSON"
            assert result.confidence_score == 0.8  # Default confidence
    
    @pytest.mark.asyncio
    async def test_batch_classify_segments(self, openai_service, mock_openai_response):
        """Test batch classification of multiple segments."""
        # Create multiple test image files
        test_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=f'_test_{i}.jpg', delete=False) as f:
                from PIL import Image
                img = Image.new('RGB', (50, 50), color='blue')
                img.save(f.name, 'JPEG')
                test_files.append(f.name)
        
        try:
            with patch.object(openai_service.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
                # Configure mock response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = mock_openai_response["choices"][0]["message"]["content"]
                mock_response.usage.total_tokens = 150
                mock_response.model = "gpt-4-vision-preview"
                mock_create.return_value = mock_response
                
                # Test batch classification
                results = await openai_service.batch_classify_segments(test_files)
                
                # Verify results
                assert len(results) == 3
                assert all(isinstance(r, ClassificationResult) for r in results)
                assert all(r.primary_label == "apple" for r in results)
                
                # Verify API was called for each image
                assert mock_create.call_count == 3
        
        finally:
            # Cleanup test files
            for file_path in test_files:
                try:
                    os.unlink(file_path)
                except OSError:
                    pass
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, openai_service):
        """Test successful connection test."""
        with patch.object(openai_service.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_response = Mock()
            mock_response.model = "gpt-3.5-turbo"
            mock_response.usage.total_tokens = 10
            mock_create.return_value = mock_response
            
            result = await openai_service.test_connection()
            
            assert result["status"] == "success"
            assert result["model_available"] == "gpt-3.5-turbo"
            assert result["tokens_used"] == 10
    
    def test_get_usage_stats(self, openai_service):
        """Test usage statistics tracking."""
        # Simulate some usage
        openai_service.total_requests = 10
        openai_service.failed_requests = 2
        openai_service.total_tokens_used = 1500
        
        stats = openai_service.get_usage_stats()
        
        assert stats["total_requests"] == 10
        assert stats["failed_requests"] == 2
        assert stats["success_rate"] == 0.8
        assert stats["total_tokens_used"] == 1500
        assert stats["average_tokens_per_request"] == 150
        assert "estimated_cost_usd" in stats


# =============================================================================
# SAM SERVICE TESTS
# =============================================================================

class TestSAMService:
    """Test suite for SAM service functionality."""
    
    @pytest.fixture
    def sam_service(self, mock_settings):
        """Create SAM service instance for testing."""
        with patch('backend.services.sam_service.get_settings', return_value=mock_settings):
            return SAMService()
    
    @pytest.mark.asyncio
    async def test_segment_image_success(self, sam_service, sample_image_path, mock_sam_response):
        """Test successful image segmentation."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Configure mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_sam_response
            
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Test segmentation
            result = await sam_service.segment_image(sample_image_path)
            
            # Verify result
            assert isinstance(result, SAMProcessingResult)
            assert len(result.segments) == 2
            assert result.total_segments_found == 2
            
            # Check individual segments
            segment1 = result.segments[0]
            assert isinstance(segment1, SegmentResult)
            assert segment1.bbox == (10, 10, 50, 50)
            assert segment1.area == 2500
            assert segment1.confidence_score == 0.9
    
    @pytest.mark.asyncio
    async def test_segment_image_file_not_found(self, sam_service):
        """Test segmentation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            await sam_service.segment_image("/nonexistent/file.jpg")
    
    @pytest.mark.asyncio
    async def test_segment_image_service_error(self, sam_service, sample_image_path):
        """Test segmentation with SAM service error."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Configure mock to return error
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Test segmentation
            with pytest.raises(Exception) as exc_info:
                await sam_service.segment_image(sample_image_path)
            
            assert "SAM service error 500" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, sam_service):
        """Test successful health check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "model_loaded": True,
                "gpu_available": False
            }
            
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await sam_service.health_check()
            
            assert result["status"] == "healthy"
            assert result["model_loaded"] is True
            assert result["gpu_available"] is False
    
    @pytest.mark.asyncio
    async def test_health_check_unreachable(self, sam_service):
        """Test health check when service is unreachable."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            
            result = await sam_service.health_check()
            
            assert result["status"] == "unreachable"
            assert "Connection refused" in result["error"]
    
    def test_get_usage_stats(self, sam_service):
        """Test usage statistics tracking."""
        # Simulate some usage
        sam_service.total_requests = 5
        sam_service.failed_requests = 1
        sam_service.total_processing_time = 120.5
        
        stats = sam_service.get_usage_stats()
        
        assert stats["total_requests"] == 5
        assert stats["failed_requests"] == 1
        assert stats["success_rate"] == 0.8
        assert stats["total_processing_time_seconds"] == 120.5
        assert stats["average_processing_time_seconds"] == 24.1


# =============================================================================
# STORAGE SERVICE TESTS
# =============================================================================

class TestStorageService:
    """Test suite for storage service functionality."""
    
    @pytest.fixture
    def storage_service(self, mock_settings):
        """Create storage service instance for testing."""
        with patch('backend.services.storage_service.get_settings', return_value=mock_settings):
            return StorageService()
    
    @pytest.mark.asyncio
    async def test_create_thumbnail(self, storage_service, sample_image_path):
        """Test thumbnail creation."""
        # Create temporary thumbnails directory
        thumbs_dir = Path(tempfile.mkdtemp()) / "thumbnails"
        thumbs_dir.mkdir(exist_ok=True)
        
        with patch.object(storage_service.settings, 'results_path', str(thumbs_dir.parent)):
            thumbnail_path = await storage_service.create_thumbnail(sample_image_path, size=50)
            
            # Verify thumbnail was created
            assert os.path.exists(thumbnail_path)
            assert thumbnail_path.endswith('_thumb_50.jpg')
            
            # Verify thumbnail size
            from PIL import Image
            with Image.open(thumbnail_path) as img:
                assert max(img.size) <= 50
        
        # Cleanup
        try:
            os.unlink(thumbnail_path)
            thumbs_dir.rmdir()
            thumbs_dir.parent.rmdir()
        except OSError:
            pass
    
    @pytest.mark.asyncio
    async def test_export_to_json(self, storage_service):
        """Test JSON export functionality."""
        # Mock database session and image record
        mock_image_record = Mock()
        mock_image_record.id = "test-image-id"
        mock_image_record.filename = "test.jpg"
        mock_image_record.file_size = 1024
        mock_image_record.width = 100
        mock_image_record.height = 100
        mock_image_record.channels = 3
        mock_image_record.mime_type = "image/jpeg"
        mock_image_record.created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        mock_image_record.status = ProcessingStatus.COMPLETED
        mock_image_record.processing_started_at = None
        mock_image_record.processing_completed_at = None
        mock_image_record.processing_config = {}
        mock_image_record.segments = []
        mock_image_record.classifications = []
        
        # Create temporary export directory
        export_dir = Path(tempfile.mkdtemp()) / "exports"
        export_dir.mkdir(exist_ok=True)
        
        with patch.object(storage_service.settings, 'results_path', str(export_dir.parent)):
            with patch('backend.services.storage_service.db_manager') as mock_db:
                mock_db.get_session_context.return_value.__enter__.return_value.query.return_value.filter.return_value.first.return_value = mock_image_record
                
                export_path = await storage_service.export_to_json("test-image-id")
                
                # Verify export file was created
                assert os.path.exists(export_path)
                assert export_path.endswith('_export.json')
                
                # Verify export content
                with open(export_path, 'r') as f:
                    export_data = json.load(f)
                
                assert export_data["export_info"]["format"] == "VisionFlow JSON Export"
                assert export_data["image_info"]["filename"] == "test.jpg"
        
        # Cleanup
        try:
            os.unlink(export_path)
            export_dir.rmdir()
            export_dir.parent.rmdir()
        except OSError:
            pass
    
    def test_get_storage_stats(self, storage_service):
        """Test storage statistics calculation."""
        # Mock the directory analysis
        with patch.object(storage_service, '_analyze_directory') as mock_analyze:
            mock_analyze.return_value = {
                "path": "/test/path",
                "file_count": 10,
                "size_bytes": 1024000,
                "size_mb": 1000,
                "size_gb": 1.0
            }
            
            stats = storage_service.get_storage_stats()
            
            assert "directories" in stats
            assert "totals" in stats
            assert "operations" in stats
            assert stats["totals"]["total_files"] >= 0
            assert stats["totals"]["total_size_bytes"] >= 0


# =============================================================================
# TRAINING SERVICE TESTS
# =============================================================================

class TestTrainingService:
    """Test suite for training service functionality."""
    
    @pytest.fixture
    def training_service(self, mock_settings):
        """Create training service instance for testing."""
        with patch('backend.services.training_service.get_settings', return_value=mock_settings):
            return TrainingService()
    
    @pytest.mark.asyncio
    async def test_get_dataset_info_empty(self, training_service):
        """Test dataset info with no training samples."""
        with patch('backend.services.training_service.db_manager') as mock_db:
            mock_db.get_session_context.return_value.__enter__.return_value.query.return_value.all.return_value = []
            
            dataset_info = await training_service.get_dataset_info()
            
            assert dataset_info["total_samples"] == 0
            assert dataset_info["ready_for_training"] is False
            assert "No training samples available" in dataset_info["issues"]
    
    @pytest.mark.asyncio
    async def test_get_dataset_info_ready(self, training_service):
        """Test dataset info with sufficient training samples."""
        # Mock training samples
        mock_samples = []
        for i in range(150):  # Above minimum threshold
            sample = Mock()
            sample.label_source = "openai" if i % 2 == 0 else "human"
            sample.ground_truth_label = f"label_{i % 10}"  # 10 different labels
            sample.created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)  # Add this line
            mock_samples.append(sample)
        
        with patch('backend.services.training_service.db_manager') as mock_db:
            mock_db.get_session_context.return_value.__enter__.return_value.query.return_value.all.return_value = mock_samples
            
            dataset_info = await training_service.get_dataset_info()
            
            assert dataset_info["total_samples"] == 150
            assert dataset_info["unique_labels"] == 10
            assert dataset_info["ready_for_training"] is True
            assert dataset_info["min_samples_per_class"] == 15
    
    @pytest.mark.asyncio
    async def test_should_trigger_training_no_previous(self, training_service):
        """Test training trigger with no previous training runs."""
        with patch('backend.services.training_service.db_manager') as mock_db:
            # Mock no active training
            mock_db.get_session_context.return_value.__enter__.return_value.query.return_value.filter.return_value.first.return_value = None
            
            # Mock sufficient samples
            mock_db.get_session_context.return_value.__enter__.return_value.query.return_value.filter.return_value.count.return_value = 150
            
            # Mock dataset info
            with patch.object(training_service, 'get_dataset_info') as mock_dataset:
                mock_dataset.return_value = {"ready_for_training": True}
                
                should_train = await training_service.should_trigger_training()
                
                assert should_train is True
    
    def test_extract_features(self, training_service):
        """Test feature extraction from classification and segment data."""
        # Mock classification object
        mock_classification = Mock()
        mock_classification.confidence_score = 0.85
        mock_classification.tokens_used = 120
        mock_classification.human_verified = True
        mock_classification.primary_label = "test_label"
        mock_classification.alternative_labels = [{"alt": 0.7}]
        
        # Mock segment object
        mock_segment = Mock()
        mock_segment.area = 1500
        mock_segment.bbox_width = 50
        mock_segment.bbox_height = 30
        mock_segment.confidence_score = 0.9
        
        features = training_service._extract_features(mock_classification, mock_segment)
        
        # Verify feature extraction
        assert len(features) == 10  # Expected number of features
        assert features[0] == 1500  # area
        assert features[1] == 50    # width
        assert features[2] == 30    # height
        assert features[3] == 50/30 # aspect ratio
        assert features[4] == 0.9   # segment confidence
        assert features[5] == 0.85  # classification confidence
    
    def test_get_training_stats(self, training_service):
        """Test training statistics tracking."""
        # Simulate some training activity
        training_service.models_trained = 3
        training_service.total_training_time = 1800  # 30 minutes
        training_service.best_accuracy = 0.92
        
        stats = training_service.get_training_stats()
        
        assert stats["models_trained"] == 3
        assert stats["total_training_time_seconds"] == 1800
        assert stats["average_training_time_seconds"] == 600  # 10 minutes per model
        assert stats["best_accuracy_achieved"] == 0.92
        assert len(stats["supported_model_types"]) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestServiceIntegration:
    """Test integration between different services."""
    
    @pytest.mark.asyncio
    async def test_full_processing_pipeline_mock(self, mock_settings):
        """Test a complete processing pipeline with mocked services."""
        # This test verifies that services can work together
        
        with patch('backend.services.openai_service.get_settings', return_value=mock_settings):
            with patch('backend.services.sam_service.get_settings', return_value=mock_settings):
                openai_service = OpenAIService()
                sam_service = SAMService()
                
                # Mock successful SAM response
                mock_sam_result = SAMProcessingResult(
                    image_path="test.jpg",
                    segments=[
                        SegmentResult(0, (10, 10, 50, 50), 2500, 0.9, segment_image_path="seg1.jpg"),
                        SegmentResult(1, (60, 60, 30, 30), 900, 0.8, segment_image_path="seg2.jpg")
                    ],
                    processing_time_seconds=5.2,
                    model_info={"model": "vit_h"},
                    total_segments_found=2
                )
                
                # Mock successful OpenAI responses
                mock_classification = ClassificationResult(
                    primary_label="apple",
                    confidence_score=0.95,
                    alternative_labels=[{"red apple": 0.85}],
                    raw_response={"test": "data"},
                    model_used="gpt-4-vision",
                    tokens_used=150,
                    processing_time_seconds=2.1
                )
                
                with patch.object(sam_service, 'segment_image', return_value=mock_sam_result):
                    with patch.object(openai_service, 'classify_image_segment', return_value=mock_classification):
                        
                        # Test the pipeline
                        sam_result = await sam_service.segment_image("test.jpg")
                        assert len(sam_result.segments) == 2
                        
                        # Classify each segment
                        classifications = []
                        for segment in sam_result.segments:
                            if segment.segment_image_path:
                                result = await openai_service.classify_image_segment(segment.segment_image_path)
                                classifications.append(result)
                        
                        assert len(classifications) == 2
                        assert all(c.primary_label == "apple" for c in classifications)


# =============================================================================
# TEST UTILITIES AND HELPERS
# =============================================================================

def create_test_image(width: int = 100, height: int = 100, color: str = 'red') -> str:
    """Create a test image file and return its path."""
    from PIL import Image
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img = Image.new('RGB', (width, height), color=color)
        img.save(f.name, 'JPEG')
        return f.name


def cleanup_test_files(*file_paths):
    """Clean up test files after tests complete."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except OSError:
            pass


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestServicePerformance:
    """Test service performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_openai_service_batch_performance(self, mock_settings):
        """Test OpenAI batch processing performance."""
        with patch('backend.services.openai_service.get_settings', return_value=mock_settings):
            service = OpenAIService()
            
            # Create multiple test images
            test_images = [create_test_image() for _ in range(5)]
            
            try:
                with patch.object(service, 'classify_image_segment') as mock_classify:
                    # Mock fast responses
                    mock_classify.return_value = ClassificationResult(
                        primary_label="test",
                        confidence_score=0.9,
                        alternative_labels=[],
                        raw_response={},
                        model_used="test",
                        tokens_used=50,
                        processing_time_seconds=0.1
                    )
                    
                    import time
                    start_time = time.time()
                    
                    # Test batch processing
                    results = await service.batch_classify_segments(test_images, max_concurrent=3)
                    
                    end_time = time.time()
                    
                    # Verify performance characteristics
                    assert len(results) == 5
                    assert end_time - start_time < 2.0  # Should complete quickly with mocking
                    assert mock_classify.call_count == 5
            
            finally:
                cleanup_test_files(*test_images)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])