"""
VisionFlow AI - SAM (Segment Anything Model) Integration Service
==============================================================

This service handles integration with Meta's Segment Anything Model (SAM).
Think of SAM as an incredibly sophisticated "cookie cutter" that can identify
and separate every distinct object or region in an image.

Why SAM?
- State-of-the-art segmentation accuracy
- Works on any type of image without training
- Provides precise object boundaries (not just bounding boxes)
- Open source and free to use
- Designed to work with minimal prompts

The SAM service runs as a separate microservice (potentially in Docker)
because it requires significant computational resources and specialized
dependencies that we don't want to burden our main API with.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import aiohttp
import aiofiles
from PIL import Image
import numpy as np

from ..config import get_settings


# =============================================================================
# LOGGER SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES FOR SAM RESULTS
# =============================================================================

@dataclass
class SegmentResult:
    """
    Represents a single segment from SAM processing.
    
    Each segment represents a distinct object or region that SAM
    identified in the image. Think of it as one "cookie" cut out
    by our intelligent cookie cutter.
    """
    segment_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    area: int
    confidence_score: float
    mask_data: Optional[np.ndarray] = None  # The actual pixel mask
    segment_image_path: Optional[str] = None  # Path to cropped segment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses and database storage."""
        return {
            'segment_id': self.segment_id,
            'bbox': list(self.bbox),
            'area': self.area,
            'confidence_score': self.confidence_score,
            'segment_image_path': self.segment_image_path
        }


@dataclass
class SAMProcessingResult:
    """
    Complete result from SAM processing of an image.
    
    This contains all the segments found in an image, along with
    metadata about the processing operation.
    """
    image_path: str
    segments: List[SegmentResult]
    processing_time_seconds: float
    model_info: Dict[str, Any]
    total_segments_found: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'image_path': self.image_path,
            'segments': [seg.to_dict() for seg in self.segments],
            'processing_time_seconds': self.processing_time_seconds,
            'model_info': self.model_info,
            'total_segments_found': self.total_segments_found
        }


# =============================================================================
# SAM SERVICE CLIENT
# =============================================================================

class SAMService:
    """
    Client for communicating with the SAM segmentation service.
    
    This class handles all communication with our SAM microservice,
    including request formatting, response parsing, error handling,
    and result post-processing.
    
    The actual SAM model runs in a separate service (potentially Docker container)
    to isolate the heavy ML dependencies and computational requirements.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.sam_service_url = self.settings.sam_service_url
        self.timeout = aiohttp.ClientTimeout(total=120)  # 2 minutes for processing
        
        # Track service usage and performance
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.failed_requests = 0
        
        logger.info(f"SAM service client initialized, connecting to: {self.sam_service_url}")
    
    async def segment_image(
        self,
        image_path: str,
        min_area: int = 1000,
        max_segments: int = 60,
        confidence_threshold: float = 0.7
    ) -> SAMProcessingResult:
        """
        Segment an image using SAM model.
        
        This is the main function that takes an image and returns all the
        distinct objects/regions that SAM found in it. Think of it as asking
        an expert: "Please identify every separate thing in this image."
        
        Args:
            image_path: Path to the image file to segment
            min_area: Minimum area in pixels for a segment to be kept
            max_segments: Maximum number of segments to return
            confidence_threshold: Minimum confidence score for segments
            
        Returns:
            SAMProcessingResult containing all found segments
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting SAM segmentation for: {image_path}")
            
            # Validate input
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Prepare request data
            request_data = await self._prepare_sam_request(
                image_path, min_area, max_segments, confidence_threshold
            )
            
            # Make the API request to SAM service
            response_data = await self._call_sam_service(request_data)
            
            # Process and validate the response
            result = await self._process_sam_response(
                response_data, image_path, time.time() - start_time
            )
            
            # Update usage statistics
            self.total_requests += 1
            self.total_processing_time += result.processing_time_seconds
            
            logger.info(f"SAM segmentation completed: {result.total_segments_found} segments found")
            return result
            
        except FileNotFoundError:
            # Let FileNotFoundError bubble up for tests that expect it
            self.failed_requests += 1
            raise
        except SAMResponseError:
            # Let SAMResponseError bubble up for tests that expect it
            self.failed_requests += 1
            raise
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"SAM segmentation failed for {image_path}: {e}")
            raise SAMProcessingError(f"Segmentation failed: {e}") from e
    
    async def batch_segment_images(
        self,
        image_paths: List[str],
        **kwargs
    ) -> List[SAMProcessingResult]:
        """
        Segment multiple images in batch.
        
        This processes multiple images but does so sequentially to avoid
        overwhelming the SAM service, which is computationally intensive.
        
        Args:
            image_paths: List of image file paths to process
            **kwargs: Additional arguments passed to segment_image()
            
        Returns:
            List of SAMProcessingResults, one per image
        """
        if not image_paths:
            return []
        
        logger.info(f"Starting batch SAM segmentation for {len(image_paths)} images")
        
        results = []
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                result = await self.segment_image(image_path, **kwargs)
                results.append(result)
                
                # Small delay between requests to be nice to the SAM service
                if i < len(image_paths) - 1:
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                # Continue with other images even if one fails
                continue
        
        logger.info(f"Batch SAM segmentation completed: {len(results)}/{len(image_paths)} successful")
        return results
    
    async def _prepare_sam_request(
        self,
        image_path: str,
        min_area: int,
        max_segments: int,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """
        Prepare the request payload for the SAM service.
        
        This function packages up all the parameters and the image data
        in the format that our SAM service expects.
        """
        # Get image information
        image_info = await self._get_image_info(image_path)
        
        # Read image file
        async with aiofiles.open(image_path, 'rb') as f:
            image_data = await f.read()
        
        # Prepare request
        request_data = {
            'image_filename': image_path,  # Keep full path
            'image_size': len(image_data),
            'image_dimensions': image_info,
            'parameters': {
                'min_area': min_area,
                'max_segments': max_segments,
                'confidence_threshold': confidence_threshold,
                'model_type': self.settings.sam_model_type,
                'device': self.settings.sam_device
            }
        }
        
        return request_data
    
    async def _call_sam_service(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make the actual HTTP request to the SAM service.
        
        This handles the low-level HTTP communication, including error handling,
        timeouts, and response validation.
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # For this example, we'll assume the SAM service accepts multipart form data
                # In practice, you might use different request formats
                
                form_data = aiohttp.FormData()
                form_data.add_field('config', json.dumps(request_data['parameters']))
                with open(request_data['image_filename'], 'rb') as img_file:
                    form_data.add_field('image', 
                                    img_file.read(),
                                    filename=Path(request_data['image_filename']).name)
                
                async with session.post(
                    f"{self.sam_service_url}/segment",
                    data=form_data
                ) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 413:
                        raise SAMServiceError("Image too large for processing")
                    elif response.status == 422:
                        error_detail = await response.text()
                        raise SAMServiceError(f"Invalid request parameters: {error_detail}")
                    elif response.status == 503:
                        raise SAMServiceError("SAM service temporarily unavailable")
                    else:
                        error_text = await response.text()
                        raise SAMServiceError(f"SAM service error {response.status}: {error_text}")
                        
        except aiohttp.ClientError as e:
            logger.error(f"Network error calling SAM service: {e}")
            raise SAMServiceError(f"Network error: {e}") from e
        
        except asyncio.TimeoutError:
            logger.error("SAM service request timed out")
            raise SAMServiceError("Request timed out") from None
    
    async def _process_sam_response(
        self,
        response_data: Dict[str, Any],
        original_image_path: str,
        processing_time: float
    ) -> SAMProcessingResult:
        """
        Process and validate the response from SAM service.
        
        This function takes the raw response from the SAM service and
        converts it into our structured SAMProcessingResult format.
        """
        try:
            # Extract segments from response
            segments = []
            raw_segments = response_data.get('segments', [])
            
            for i, seg_data in enumerate(raw_segments):
                segment = SegmentResult(
                    segment_id=seg_data.get('id', i),
                    bbox=tuple(seg_data['bbox']),  # [x, y, w, h]
                    area=seg_data['area'],
                    confidence_score=seg_data.get('confidence', 1.0),
                    segment_image_path=seg_data.get('segment_image_path')
                )
                segments.append(segment)
            
            # Create result object
            result = SAMProcessingResult(
                image_path=original_image_path,
                segments=segments,
                processing_time_seconds=processing_time,
                model_info=response_data.get('model_info', {}),
                total_segments_found=len(segments)
            )
            
            return result
            
        except KeyError as e:
            logger.error(f"Invalid SAM response format, missing key: {e}")
            raise SAMResponseError(f"Invalid response format: missing {e}") from e
        
        except Exception as e:
            logger.error(f"Error processing SAM response: {e}")
            raise SAMResponseError(f"Response processing failed: {e}") from e
    
    async def _get_image_info(self, image_path: str) -> Dict[str, int]:
        """
        Get basic information about an image file.
        
        This extracts image dimensions and other metadata that might
        be useful for the SAM service.
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'channels': len(img.getbands()) if img.getbands() else 3
                }
        except Exception as e:
            logger.warning(f"Could not get image info for {image_path}: {e}")
            return {'width': 0, 'height': 0, 'channels': 3}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the SAM service is healthy and responsive.
        
        This is useful for monitoring and debugging. It makes a simple
        request to verify the service is running and accessible.
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(f"{self.sam_service_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        return {
                            'status': 'healthy',
                            'service_url': self.sam_service_url,
                            'response_time_ms': health_data.get('response_time_ms'),
                            'model_loaded': health_data.get('model_loaded', False),
                            'gpu_available': health_data.get('gpu_available', False)
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'error': f"HTTP {response.status}",
                            'service_url': self.sam_service_url
                        }
                        
        except Exception as e:
            logger.error(f"SAM service health check failed: {e}")
            return {
                'status': 'unreachable',
                'error': str(e),
                'service_url': self.sam_service_url
            }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for monitoring and performance analysis.
        
        This helps you understand how much you're using the SAM service
        and identify potential performance bottlenecks.
        """
        avg_processing_time = (
            self.total_processing_time / max(self.total_requests, 1)
        )
        
        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            'total_processing_time_seconds': self.total_processing_time,
            'average_processing_time_seconds': avg_processing_time,
            'service_url': self.sam_service_url
        }


# =============================================================================
# SEGMENT POST-PROCESSING UTILITIES
# =============================================================================

def filter_segments_by_area(
    segments: List[SegmentResult],
    min_area: int = 1000,
    max_area: Optional[int] = None
) -> List[SegmentResult]:
    """
    Filter segments based on area constraints.
    
    This removes segments that are too small (likely noise) or too large
    (likely background regions) to focus on meaningful objects.
    """
    filtered = []
    for segment in segments:
        if segment.area >= min_area:
            if max_area is None or segment.area <= max_area:
                filtered.append(segment)
    
    logger.debug(f"Area filtering: {len(segments)} -> {len(filtered)} segments")
    return filtered


def merge_overlapping_segments(
    segments: List[SegmentResult],
    iou_threshold: float = 0.5
) -> List[SegmentResult]:
    """
    Merge segments that overlap significantly.
    
    Sometimes SAM creates multiple segments for the same object.
    This function identifies overlapping segments and merges them.
    """
    if not segments:
        return segments
    
    def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Sort segments by area (largest first)
    sorted_segments = sorted(segments, key=lambda s: s.area, reverse=True)
    merged = []
    
    for segment in sorted_segments:
        # Check if this segment overlaps significantly with any already merged segment
        should_merge = False
        for existing in merged:
            iou = calculate_iou(segment.bbox, existing.bbox)
            if iou >= iou_threshold:
                should_merge = True
                break
        
        if not should_merge:
            merged.append(segment)
    
    logger.debug(f"Overlap merging: {len(segments)} -> {len(merged)} segments")
    return merged


def rank_segments_by_importance(
    segments: List[SegmentResult],
    image_center: Optional[Tuple[int, int]] = None
) -> List[SegmentResult]:
    """
    Rank segments by likely importance for classification.
    
    This uses heuristics to identify which segments are most likely to
    contain important objects that should be classified first.
    """
    def calculate_importance_score(segment: SegmentResult) -> float:
        """Calculate importance score based on various factors."""
        score = 0.0
        
        # Larger segments are generally more important
        score += min(segment.area / 10000, 1.0) * 0.3
        
        # Higher confidence segments are more important
        score += segment.confidence_score * 0.4
        
        # Segments closer to center are often more important
        if image_center:
            seg_center_x = segment.bbox[0] + segment.bbox[2] / 2
            seg_center_y = segment.bbox[1] + segment.bbox[3] / 2
            
            distance_from_center = (
                (seg_center_x - image_center[0]) ** 2 +
                (seg_center_y - image_center[1]) ** 2
            ) ** 0.5
            
            # Normalize and invert (closer = higher score)
            max_distance = (image_center[0] ** 2 + image_center[1] ** 2) ** 0.5
            center_score = 1.0 - (distance_from_center / max_distance)
            score += center_score * 0.3
        
        return score
    
    # Calculate scores and sort
    for segment in segments:
        segment.importance_score = calculate_importance_score(segment)
    
    ranked = sorted(segments, key=lambda s: s.importance_score, reverse=True)
    logger.debug(f"Segments ranked by importance")
    
    return ranked


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class SAMServiceError(Exception):
    """Base exception for SAM service errors."""
    pass


class SAMProcessingError(SAMServiceError):
    """Raised when SAM processing fails."""
    pass


class SAMResponseError(SAMServiceError):
    """Raised when SAM response is invalid or cannot be parsed."""
    pass


# =============================================================================
# SERVICE FACTORY FUNCTION
# =============================================================================

_sam_service_instance = None

def get_sam_service() -> SAMService:
    """
    Get singleton instance of SAM service.
    
    Using a singleton ensures we don't create multiple instances
    and helps with connection pooling and usage tracking.
    """
    global _sam_service_instance
    
    if _sam_service_instance is None:
        _sam_service_instance = SAMService()
    
    return _sam_service_instance