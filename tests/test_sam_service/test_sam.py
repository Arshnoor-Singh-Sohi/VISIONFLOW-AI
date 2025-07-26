"""
VisionFlow AI - SAM Service Tests
=================================

This module contains comprehensive tests for the SAM (Segment Anything Model) service.
These tests verify the integration with the SAM microservice, error handling,
response parsing, and various edge cases.

The SAM service is critical for the image segmentation pipeline, so we need to
ensure it's robust and reliable under various conditions.
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from PIL import Image
import numpy as np
import aiohttp
from aiohttp import ClientError, ClientTimeout

# Import SAM service components
from backend.services.sam_service import (
    SAMService, 
    SegmentResult, 
    SAMProcessingResult,
    SAMServiceError,
    SAMProcessingError,
    SAMResponseError,
    get_sam_service,
    filter_segments_by_area,
    merge_overlapping_segments,
    rank_segments_by_importance
)
from backend.config import get_settings


# =============================================================================
# TEST FIXTURES AND SETUP
# =============================================================================

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.sam_service_url = "http://localhost:8001"
    settings.sam_model_type = "vit_h"
    settings.sam_device = "cpu"
    return settings


@pytest.fixture
def sam_service(mock_settings):
    """Create SAM service instance with mocked settings."""
    with patch('backend.services.sam_service.get_settings', return_value=mock_settings):
        service = SAMService()
        yield service


@pytest.fixture
def sample_image_file():
    """Create a temporary test image file."""
    # Create a simple test image
    image = Image.new('RGB', (640, 480), color='white')
    
    # Add some colored rectangles to create distinct segments
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Red rectangle
    draw.rectangle([100, 100, 200, 200], fill='red')
    # Blue circle (approximated with rectangle for simplicity)
    draw.rectangle([300, 150, 400, 250], fill='blue')
    # Green triangle (approximated)
    draw.rectangle([500, 200, 580, 280], fill='green')
    
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
def valid_sam_response():
    """Valid SAM service response for testing."""
    return {
        'segments': [
            {
                'id': 0,
                'bbox': [100, 100, 100, 100],  # Red rectangle
                'area': 10000,
                'confidence': 0.95,
                'segment_image_path': '/tmp/segment_0.jpg'
            },
            {
                'id': 1,
                'bbox': [300, 150, 100, 100],  # Blue rectangle
                'area': 10000,
                'confidence': 0.88,
                'segment_image_path': '/tmp/segment_1.jpg'
            },
            {
                'id': 2,
                'bbox': [500, 200, 80, 80],    # Green rectangle
                'area': 6400,
                'confidence': 0.92,
                'segment_image_path': '/tmp/segment_2.jpg'
            },
            {
                'id': 3,
                'bbox': [0, 0, 640, 480],      # Background
                'area': 307200,
                'confidence': 0.75,
                'segment_image_path': '/tmp/segment_3.jpg'
            }
        ],
        'model_info': {
            'model_type': 'SAM',
            'model_version': 'vit_h',
            'device': 'cpu',
            'processing_time': 2.5
        }
    }


@pytest.fixture
def invalid_sam_response():
    """Invalid SAM service response for testing error handling."""
    return {
        'error': 'Model not loaded',
        'status': 'failed'
    }


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestSAMServiceBasics:
    """Test basic SAM service functionality and initialization."""
    
    def test_sam_service_initialization(self, mock_settings):
        """Test that SAM service initializes correctly."""
        with patch('backend.services.sam_service.get_settings', return_value=mock_settings):
            service = SAMService()
            
            assert service.sam_service_url == "http://localhost:8001"
            assert service.total_requests == 0
            assert service.failed_requests == 0
            assert service.total_processing_time == 0.0
    
    def test_singleton_service_factory(self, mock_settings):
        """Test that get_sam_service returns singleton instance."""
        with patch('backend.services.sam_service.get_settings', return_value=mock_settings):
            service1 = get_sam_service()
            service2 = get_sam_service()
            
            # Should be the same instance
            assert service1 is service2
    
    def test_service_statistics_tracking(self, sam_service):
        """Test that service tracks usage statistics correctly."""
        stats = sam_service.get_usage_stats()
        
        assert 'total_requests' in stats
        assert 'failed_requests' in stats
        assert 'success_rate' in stats
        assert 'total_processing_time_seconds' in stats
        assert 'average_processing_time_seconds' in stats
        assert 'service_url' in stats
        
        # Initial values should be zero
        assert stats['total_requests'] == 0
        assert stats['failed_requests'] == 0


# =============================================================================
# IMAGE SEGMENTATION TESTS
# =============================================================================

class TestImageSegmentation:
    """Test image segmentation functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_image_segmentation(self, sam_service, sample_image_file, valid_sam_response):
        """Test successful image segmentation with valid response."""
        
        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=valid_sam_response)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await sam_service.segment_image(
                sample_image_file,
                min_area=1000,
                max_segments=10,
                confidence_threshold=0.7
            )
            
            # Verify result structure
            assert isinstance(result, SAMProcessingResult)
            assert result.image_path == sample_image_file
            assert len(result.segments) == 4
            assert result.total_segments_found == 4
            assert result.processing_time_seconds > 0
            
            # Verify segment details
            first_segment = result.segments[0]
            assert isinstance(first_segment, SegmentResult)
            assert first_segment.segment_id == 0
            assert first_segment.bbox == (100, 100, 100, 100)
            assert first_segment.area == 10000
            assert first_segment.confidence_score == 0.95
            
            # Verify statistics were updated
            assert sam_service.total_requests == 1
            assert sam_service.failed_requests == 0
    
    @pytest.mark.asyncio
    async def test_image_segmentation_with_filters(self, sam_service, sample_image_file, valid_sam_response):
        """Test image segmentation with filtering parameters."""
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=valid_sam_response)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await sam_service.segment_image(
                sample_image_file,
                min_area=8000,  # Filter out smaller segments
                max_segments=3,
                confidence_threshold=0.8
            )
            
            # Should still get all segments since filtering happens after processing
            assert len(result.segments) == 4
            
            # But we can test the filter functions separately
            filtered_segments = filter_segments_by_area(result.segments, min_area=8000)
            assert len(filtered_segments) == 3  # Should exclude the 6400 area segment
    
    @pytest.mark.asyncio
    async def test_file_not_found_error(self, sam_service):
        """Test handling of non-existent image file."""
        
        with pytest.raises(FileNotFoundError):
            await sam_service.segment_image("nonexistent_file.jpg")
    
    @pytest.mark.asyncio
    async def test_sam_service_http_error(self, sam_service, sample_image_file):
        """Test handling of HTTP errors from SAM service."""
        
        # Mock HTTP 500 error
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(SAMServiceError) as exc_info:
                await sam_service.segment_image(sample_image_file)
            
            assert "SAM service error 500" in str(exc_info.value)
            assert sam_service.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_sam_service_timeout(self, sam_service, sample_image_file):
        """Test handling of timeout errors."""
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(SAMServiceError) as exc_info:
                await sam_service.segment_image(sample_image_file)
            
            assert "Request timed out" in str(exc_info.value)
            assert sam_service.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_sam_service_network_error(self, sam_service, sample_image_file):
        """Test handling of network errors."""
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = ClientError("Connection failed")
            
            with pytest.raises(SAMServiceError) as exc_info:
                await sam_service.segment_image(sample_image_file)
            
            assert "Network error" in str(exc_info.value)
            assert sam_service.failed_requests == 1


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================

class TestBatchProcessing:
    """Test batch image processing functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_segment_images_success(self, sam_service, valid_sam_response):
        """Test successful batch processing of multiple images."""
        
        # Create multiple test images
        image_paths = []
        for i in range(3):
            img = Image.new('RGB', (100, 100), color=['red', 'green', 'blue'][i])
            temp_file = tempfile.NamedTemporaryFile(suffix=f'_batch_{i}.jpg', delete=False)
            img.save(temp_file.name, 'JPEG')
            image_paths.append(temp_file.name)
        
        try:
            # Mock successful responses for all images
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=valid_sam_response)
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value = mock_response
                
                results = await sam_service.batch_segment_images(
                    image_paths,
                    min_area=1000
                )
                
                assert len(results) == 3
                for result in results:
                    assert isinstance(result, SAMProcessingResult)
                    assert len(result.segments) == 4
                
                # Check that total requests were tracked
                assert sam_service.total_requests == 3
                
        finally:
            # Cleanup
            for path in image_paths:
                try:
                    os.unlink(path)
                except:
                    pass
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_failures(self, sam_service, valid_sam_response):
        """Test batch processing where some images fail."""
        
        # Create test images (one will be invalid)
        image_paths = []
        
        # Valid image
        img = Image.new('RGB', (100, 100), color='red')
        temp_file = tempfile.NamedTemporaryFile(suffix='_valid.jpg', delete=False)
        img.save(temp_file.name, 'JPEG')
        image_paths.append(temp_file.name)
        
        # Invalid path
        image_paths.append("nonexistent_file.jpg")
        
        try:
            # Mock response for the valid image
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=valid_sam_response)
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value = mock_response
                
                results = await sam_service.batch_segment_images(image_paths)
                
                # Should only get result for the valid image
                assert len(results) == 1
                assert isinstance(results[0], SAMProcessingResult)
                
        finally:
            # Cleanup
            try:
                os.unlink(image_paths[0])
            except:
                pass
    
    @pytest.mark.asyncio
    async def test_empty_batch_processing(self, sam_service):
        """Test batch processing with empty list."""
        
        results = await sam_service.batch_segment_images([])
        assert results == []


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================

class TestHealthCheck:
    """Test SAM service health check functionality."""
    
    @pytest.mark.asyncio
    async def test_healthy_service_check(self, sam_service):
        """Test health check when service is healthy."""
        
        mock_health_response = {
            'status': 'healthy',
            'model_loaded': True,
            'gpu_available': False,
            'response_time_ms': 50
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_health_response)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            health_result = await sam_service.health_check()
            
            assert health_result['status'] == 'healthy'
            assert health_result['service_url'] == sam_service.sam_service_url
            assert health_result['model_loaded'] is True
            assert health_result['gpu_available'] is False
            assert 'response_time_ms' in health_result
    
    @pytest.mark.asyncio
    async def test_unhealthy_service_check(self, sam_service):
        """Test health check when service is unhealthy."""
        
        mock_response = AsyncMock()
        mock_response.status = 503
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            health_result = await sam_service.health_check()
            
            assert health_result['status'] == 'unhealthy'
            assert 'HTTP 503' in health_result['error']
    
    @pytest.mark.asyncio
    async def test_unreachable_service_check(self, sam_service):
        """Test health check when service is unreachable."""
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = ClientError("Connection refused")
            
            health_result = await sam_service.health_check()
            
            assert health_result['status'] == 'unreachable'
            assert 'Connection refused' in health_result['error']


# =============================================================================
# RESPONSE PARSING TESTS
# =============================================================================

class TestResponseParsing:
    """Test parsing of SAM service responses."""
    
    @pytest.mark.asyncio
    async def test_valid_response_parsing(self, sam_service, sample_image_file, valid_sam_response):
        """Test parsing of valid SAM response."""
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=valid_sam_response)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await sam_service.segment_image(sample_image_file)
            
            # Verify all fields were parsed correctly
            assert result.model_info['model_type'] == 'SAM'
            assert result.model_info['model_version'] == 'vit_h'
            assert result.model_info['device'] == 'cpu'
            
            # Check segment parsing
            for i, segment in enumerate(result.segments):
                expected = valid_sam_response['segments'][i]
                assert segment.segment_id == expected['id']
                assert segment.bbox == tuple(expected['bbox'])
                assert segment.area == expected['area']
                assert segment.confidence_score == expected['confidence']
    
    @pytest.mark.asyncio
    async def test_malformed_response_parsing(self, sam_service, sample_image_file):
        """Test handling of malformed SAM response."""
        
        # Response missing required fields
        malformed_response = {
            'segments': [
                {
                    'id': 0,
                    # Missing bbox, area, confidence
                }
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=malformed_response)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(SAMResponseError):
                await sam_service.segment_image(sample_image_file)
    
    @pytest.mark.asyncio
    async def test_empty_segments_response(self, sam_service, sample_image_file):
        """Test handling of response with no segments."""
        
        empty_response = {
            'segments': [],
            'model_info': {'model_type': 'SAM'}
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=empty_response)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await sam_service.segment_image(sample_image_file)
            
            assert len(result.segments) == 0
            assert result.total_segments_found == 0


# =============================================================================
# SEGMENT POST-PROCESSING TESTS
# =============================================================================

class TestSegmentPostProcessing:
    """Test segment post-processing utility functions."""
    
    def test_filter_segments_by_area(self):
        """Test filtering segments based on area constraints."""
        
        segments = [
            SegmentResult(0, (0, 0, 10, 10), 100, 0.9),      # Small area
            SegmentResult(1, (0, 0, 50, 50), 2500, 0.9),     # Medium area
            SegmentResult(2, (0, 0, 100, 100), 10000, 0.9),  # Large area
            SegmentResult(3, (0, 0, 200, 200), 40000, 0.9),  # Very large area
        ]
        
        # Filter out segments smaller than 1000 pixels
        filtered = filter_segments_by_area(segments, min_area=1000)
        assert len(filtered) == 3  # Should keep segments with areas: 2500, 10000, 40000
        assert all(seg.area >= 1000 for seg in filtered)
        
        # Filter with both min and max area
        filtered_range = filter_segments_by_area(segments, min_area=1000, max_area=20000)
        assert len(filtered_range) == 2  # Should keep medium (2500) and large (10000)
        assert filtered_range[0].area == 2500
        assert filtered_range[1].area == 10000
    
    def test_merge_overlapping_segments(self):
        """Test merging of overlapping segments."""
        
        segments = [
            SegmentResult(0, (10, 10, 50, 50), 2500, 0.9),   # Base segment
            SegmentResult(1, (15, 15, 50, 50), 2500, 0.8),   # Overlapping
            SegmentResult(2, (100, 100, 30, 30), 900, 0.7),  # Non-overlapping
        ]
        
        merged = merge_overlapping_segments(segments, iou_threshold=0.3)
        
        # Should merge the first two overlapping segments
        assert len(merged) == 2
        # The larger segment should be kept
        assert merged[0].area == 2500  # First segment (larger confidence)
        assert merged[1].area == 900   # Non-overlapping segment
    
    def test_rank_segments_by_importance(self):
        """Test ranking segments by importance score."""
        
        segments = [
            SegmentResult(0, (0, 0, 10, 10), 100, 0.5),      # Small, low confidence
            SegmentResult(1, (150, 100, 100, 100), 10000, 0.9),  # Center, high confidence
            SegmentResult(2, (300, 250, 50, 50), 2500, 0.8), # Off-center, medium confidence
        ]
        
        # Rank with image center at (200, 150)
        ranked = rank_segments_by_importance(segments, image_center=(200, 150))
        
        # The center segment with high confidence should be ranked highest
        assert ranked[0].segment_id == 1
        assert hasattr(ranked[0], 'importance_score')
        assert ranked[0].importance_score > ranked[1].importance_score


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_image_too_large_error(self, sam_service, sample_image_file):
        """Test handling of image too large error."""
        
        mock_response = AsyncMock()
        mock_response.status = 413
        mock_response.text = AsyncMock(return_value="Image too large for processing")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(SAMServiceError) as exc_info:
                await sam_service.segment_image(sample_image_file)
            
            assert "Image too large" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_invalid_parameters_error(self, sam_service, sample_image_file):
        """Test handling of invalid parameters error."""
        
        mock_response = AsyncMock()
        mock_response.status = 422
        mock_response.text = AsyncMock(return_value="Invalid min_area parameter")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(SAMServiceError) as exc_info:
                await sam_service.segment_image(sample_image_file, min_area=-1)
            
            assert "Invalid request parameters" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_service_unavailable_error(self, sam_service, sample_image_file):
        """Test handling of service unavailable error."""
        
        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.text = AsyncMock(return_value="SAM model not loaded")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(SAMServiceError) as exc_info:
                await sam_service.segment_image(sample_image_file)
            
            assert "SAM service temporarily unavailable" in str(exc_info.value)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance characteristics and resource usage."""
    
    @pytest.mark.asyncio
    async def test_large_image_processing(self, sam_service):
        """Test processing of large images."""
        
        # Create a large test image
        large_image = Image.new('RGB', (2048, 1536), color='white')
        
        # Add some content
        from PIL import ImageDraw
        draw = ImageDraw.Draw(large_image)
        draw.rectangle([500, 500, 1000, 1000], fill='red')
        
        temp_file = tempfile.NamedTemporaryFile(suffix='_large.jpg', delete=False)
        large_image.save(temp_file.name, 'JPEG')
        
        try:
            # Mock a response that includes processing time
            mock_response_data = {
                'segments': [
                    {
                        'id': 0,
                        'bbox': [500, 500, 500, 500],
                        'area': 250000,
                        'confidence': 0.9,
                        'segment_image_path': '/tmp/large_segment.jpg'
                    }
                ],
                'model_info': {
                    'model_type': 'SAM',
                    'processing_time': 5.2  # Longer processing time for large image
                }
            }
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value = mock_response
                
                result = await sam_service.segment_image(temp_file.name)
                
                assert isinstance(result, SAMProcessingResult)
                assert result.processing_time_seconds > 0
                
        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def test_memory_usage_tracking(self, sam_service):
        """Test that memory usage is tracked in statistics."""
        
        # Process several mock requests to build up statistics
        sam_service.total_requests = 10
        sam_service.total_processing_time = 25.5
        sam_service.failed_requests = 1
        
        stats = sam_service.get_usage_stats()
        
        assert stats['total_requests'] == 10
        assert stats['failed_requests'] == 1
        assert stats['success_rate'] == 0.9
        assert stats['average_processing_time_seconds'] == 2.55
        assert stats['total_processing_time_seconds'] == 25.5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test integration with other system components."""
    
    @pytest.mark.asyncio
    async def test_image_metadata_integration(self, sam_service, sample_image_file, valid_sam_response):
        """Test integration with image metadata extraction."""
        
        # Test that image info is correctly extracted before processing
        from backend.utils.image_processing import get_image_metadata
        
        metadata = get_image_metadata(sample_image_file)
        assert metadata['width'] > 0
        assert metadata['height'] > 0
        
        # Mock successful SAM processing
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=valid_sam_response)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await sam_service.segment_image(sample_image_file)
            
            # Verify that processing worked with the metadata
            assert result.image_path == sample_image_file
            assert len(result.segments) > 0


# =============================================================================
# BENCHMARK TESTS (OPTIONAL)
# =============================================================================

@pytest.mark.benchmark
class TestBenchmarks:
    """Performance benchmark tests (run with pytest-benchmark if available)."""
    
    @pytest.mark.asyncio
    async def test_segmentation_speed_benchmark(self, sam_service, sample_image_file, valid_sam_response):
        """Benchmark segmentation processing speed."""
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=valid_sam_response)
        
        async def run_segmentation():
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value.__aenter__.return_value = mock_response
                return await sam_service.segment_image(sample_image_file)
        
        # This would work with pytest-benchmark plugin
        # result = await benchmark(run_segmentation)
        result = await run_segmentation()
        assert isinstance(result, SAMProcessingResult)


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestUtilities:
    """Test utility functions and helper methods."""
    
    def test_segment_result_to_dict(self):
        """Test conversion of SegmentResult to dictionary."""
        
        segment = SegmentResult(
            segment_id=1,
            bbox=(10, 20, 100, 200),
            area=20000,
            confidence_score=0.85,
            segment_image_path='/tmp/segment.jpg'
        )
        
        segment_dict = segment.to_dict()
        
        assert segment_dict['segment_id'] == 1
        assert segment_dict['bbox'] == [10, 20, 100, 200]
        assert segment_dict['area'] == 20000
        assert segment_dict['confidence_score'] == 0.85
        assert segment_dict['segment_image_path'] == '/tmp/segment.jpg'
    
    def test_sam_processing_result_to_dict(self, valid_sam_response):
        """Test conversion of SAMProcessingResult to dictionary."""
        
        # Create segments from response
        segments = []
        for seg_data in valid_sam_response['segments']:
            segment = SegmentResult(
                segment_id=seg_data['id'],
                bbox=tuple(seg_data['bbox']),
                area=seg_data['area'],
                confidence_score=seg_data['confidence'],
                segment_image_path=seg_data['segment_image_path']
            )
            segments.append(segment)
        
        result = SAMProcessingResult(
            image_path='/test/image.jpg',
            segments=segments,
            processing_time_seconds=2.5,
            model_info=valid_sam_response['model_info'],
            total_segments_found=len(segments)
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['image_path'] == '/test/image.jpg'
        assert result_dict['total_segments_found'] == 4
        assert result_dict['processing_time_seconds'] == 2.5
        assert len(result_dict['segments']) == 4
        assert 'model_info' in result_dict


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    pytest.main([__file__, "-v", "--tb=short"])