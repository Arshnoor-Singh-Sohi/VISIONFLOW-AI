"""
VisionFlow AI - SAM Model Processor
===================================

This module contains the core SAM (Segment Anything Model) processing logic.
It wraps the SAM model with additional functionality for our specific use case:
- Image preprocessing and optimization
- Segment filtering and post-processing
- Performance monitoring and statistics
- Error handling and recovery
"""

import os
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# =============================================================================
# LOGGER SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# SAM PROCESSOR CLASS
# =============================================================================

class SAMProcessor:
    """
    High-level interface for SAM model processing.
    
    This class encapsulates all the complexity of working with SAM,
    providing a clean, async-friendly interface for image segmentation.
    
    Features:
    - Automatic model loading and initialization
    - Image preprocessing and optimization
    - Configurable segmentation parameters
    - Performance monitoring and statistics
    - Error handling and recovery
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        model_type: str = 'vit_h',
        max_image_size: int = 2048
    ):
        """
        Initialize the SAM processor.
        
        Args:
            model_path: Path to the SAM model checkpoint file
            device: Device to run the model on ('cpu', 'cuda', 'mps')
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            max_image_size: Maximum image dimension for processing
        """
        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        self.max_image_size = max_image_size
        
        # Model components (loaded during initialization)
        self.sam_model = None
        self.mask_generator = None
        
        # Performance tracking
        self.total_images_processed = 0
        self.total_processing_time = 0.0
        self.total_segments_generated = 0
        self.failed_processing_count = 0
        
        # Initialize the model
        self._load_model()
        
        logger.info(f"SAM processor initialized: {model_type} on {device}")
    
    def _load_model(self) -> None:
        """
        Load the SAM model and initialize the mask generator.
        
        This is the most time and memory-intensive operation,
        typically taking 30-60 seconds depending on the model size.
        """
        try:
            logger.info(f"Loading SAM model from {self.model_path}")
            
            # Verify model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
            
            # Determine device
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = 'cpu'
            elif self.device == 'mps' and not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available, falling back to CPU")
                self.device = 'cpu'
            
            # Load the SAM model
            self.sam_model = sam_model_registry[self.model_type](
                checkpoint=self.model_path
            )
            self.sam_model.to(device=self.device)
            
            # Initialize the automatic mask generator with optimized parameters
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam_model,
                # Adjust these parameters based on your needs
                points_per_side=32,           # Number of points to sample along each side
                pred_iou_thresh=0.86,         # IoU threshold for mask prediction quality
                stability_score_thresh=0.92,  # Stability score threshold
                crop_n_layers=1,              # Number of layers for crop-based processing
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,     # Minimum area for a mask to be retained
            )
            
            logger.info("SAM model loaded successfully")
            
            # Log memory usage if on CUDA
            if self.device == 'cuda' and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                logger.info(f"GPU memory allocated: {memory_allocated:.1f} MB")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise RuntimeError(f"SAM model loading failed: {e}") from e
    
    async def process_image(
        self,
        image_path: str,
        min_area: int = 1000,
        max_segments: int = 60,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Process an image to generate segments.
        
        This is the main processing function that takes an image and returns
        all the distinct objects/regions found by SAM.
        
        Args:
            image_path: Path to the image file
            min_area: Minimum area in pixels for a segment to be kept
            max_segments: Maximum number of segments to return
            confidence_threshold: Minimum confidence score for segments
            
        Returns:
            Dictionary containing segments and processing metadata
        """
        start_time = time.time()
        
        try:
            # Load and preprocess the image
            image_rgb = self._load_and_preprocess_image(image_path)
            
            logger.debug(f"Processing image: {image_rgb.shape}")
            
            # Generate masks using SAM
            raw_masks = self._generate_masks(image_rgb)
            
            logger.debug(f"Generated {len(raw_masks)} raw masks")
            
            # Filter and process the masks
            processed_segments = self._process_masks(
                raw_masks,
                min_area=min_area,
                max_segments=max_segments,
                confidence_threshold=confidence_threshold
            )
            
            # Save segment crops if needed
            segment_paths = await self._save_segment_crops(
                image_rgb, processed_segments, image_path
            )
            
            # Add file paths to segments
            for i, segment in enumerate(processed_segments):
                if i < len(segment_paths):
                    segment['segment_image_path'] = segment_paths[i]
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.total_images_processed += 1
            self.total_processing_time += processing_time
            self.total_segments_generated += len(processed_segments)
            
            return {
                'segments': processed_segments,
                'processing_time_seconds': processing_time,
                'total_segments_found': len(processed_segments)
            }
            
        except Exception as e:
            self.failed_processing_count += 1
            logger.error(f"Image processing failed: {e}")
            raise RuntimeError(f"Processing failed: {e}") from e
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load an image file and preprocess it for SAM.
        
        SAM expects images in RGB format as numpy arrays.
        This function also handles resizing if the image is too large.
        """
        try:
            # Load image using PIL (handles more formats than OpenCV)
            with Image.open(image_path) as pil_image:
                # Convert to RGB if needed (removes alpha channel, handles CMYK, etc.)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Resize if too large (saves memory and processing time)
                width, height = pil_image.size
                max_dim = max(width, height)
                
                if max_dim > self.max_image_size:
                    scale_factor = self.max_image_size / max_dim
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    pil_image = pil_image.resize(
                        (new_width, new_height), 
                        Image.Resampling.LANCZOS
                    )
                    
                    logger.debug(f"Image resized from {width}x{height} to {new_width}x{new_height}")
                
                # Convert to numpy array (RGB format for SAM)
                image_rgb = np.array(pil_image)
                
                return image_rgb
                
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise ValueError(f"Image loading failed: {e}") from e
    
    def _generate_masks(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate masks using the SAM automatic mask generator.
        
        This is where the actual SAM processing happens. The model analyzes
        the image and identifies all the distinct objects/regions it can find.
        """
        try:
            # Generate masks using SAM
            masks = self.mask_generator.generate(image_rgb)
            
            logger.debug(f"SAM generated {len(masks)} masks")
            
            return masks
            
        except Exception as e:
            logger.error(f"SAM mask generation failed: {e}")
            raise RuntimeError(f"Mask generation failed: {e}") from e
    
    def _process_masks(
        self,
        raw_masks: List[Dict[str, Any]],
        min_area: int,
        max_segments: int,
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Process and filter the raw masks from SAM.
        
        SAM often generates many masks, including noise and overlapping regions.
        This function applies filtering and ranking to get the most useful segments.
        """
        processed_segments = []
        
        for i, mask_data in enumerate(raw_masks):
            try:
                # Extract mask information
                segmentation = mask_data['segmentation']
                bbox = mask_data['bbox']  # [x, y, width, height]
                area = mask_data['area']
                predicted_iou = mask_data.get('predicted_iou', 1.0)
                stability_score = mask_data.get('stability_score', 1.0)
                
                # Calculate composite confidence score
                confidence = (predicted_iou + stability_score) / 2.0
                
                # Apply filters
                if area < min_area:
                    continue
                
                if confidence < confidence_threshold:
                    continue
                
                # Create segment record
                segment = {
                    'id': i,
                    'bbox': [int(coord) for coord in bbox],  # Ensure integers
                    'area': int(area),
                    'confidence': float(confidence),
                    'predicted_iou': float(predicted_iou),
                    'stability_score': float(stability_score),
                    'mask_data': segmentation.tolist() if isinstance(segmentation, np.ndarray) else segmentation
                }
                
                processed_segments.append(segment)
                
            except Exception as e:
                logger.warning(f"Failed to process mask {i}: {e}")
                continue
        
        # Sort by area (largest first) and limit number of segments
        processed_segments.sort(key=lambda x: x['area'], reverse=True)
        processed_segments = processed_segments[:max_segments]
        
        # Re-assign sequential IDs after filtering
        for i, segment in enumerate(processed_segments):
            segment['id'] = i
        
        logger.debug(f"Processed {len(processed_segments)} segments after filtering")
        
        return processed_segments
    
    async def _save_segment_crops(
        self,
        image_rgb: np.ndarray,
        segments: List[Dict[str, Any]],
        original_image_path: str
    ) -> List[str]:
        """
        Save cropped images for each segment.
        
        This creates individual image files for each segment, which can then
        be sent to OpenAI for classification. Each crop contains just the
        pixels within the segment's bounding box.
        """
        segment_paths = []
        
        try:
            # Create output directory
            image_name = Path(original_image_path).stem
            output_dir = Path('/app/segments') / f"{image_name}_{int(time.time())}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for segment in segments:
                try:
                    # Extract bounding box coordinates
                    x, y, w, h = segment['bbox']
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, min(x, image_rgb.shape[1] - 1))
                    y = max(0, min(y, image_rgb.shape[0] - 1))
                    w = max(1, min(w, image_rgb.shape[1] - x))
                    h = max(1, min(h, image_rgb.shape[0] - y))
                    
                    # Crop the segment from the original image
                    crop = image_rgb[y:y+h, x:x+w]
                    
                    # Skip if crop is too small
                    if crop.size == 0:
                        logger.warning(f"Empty crop for segment {segment['id']}")
                        segment_paths.append(None)
                        continue
                    
                    # Save as JPEG
                    segment_filename = f"segment_{segment['id']:02d}.jpg"
                    segment_path = output_dir / segment_filename
                    
                    # Convert numpy array to PIL Image and save
                    crop_image = Image.fromarray(crop)
                    crop_image.save(segment_path, 'JPEG', quality=90)
                    
                    segment_paths.append(str(segment_path))
                    
                    logger.debug(f"Saved segment crop: {segment_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to save crop for segment {segment['id']}: {e}")
                    segment_paths.append(None)
            
            return segment_paths
            
        except Exception as e:
            logger.error(f"Failed to save segment crops: {e}")
            return [None] * len(segments)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics and performance metrics.
        
        This provides insights into how the SAM processor is performing,
        useful for monitoring and optimization.
        """
        avg_processing_time = (
            self.total_processing_time / max(self.total_images_processed, 1)
        )
        
        avg_segments_per_image = (
            self.total_segments_generated / max(self.total_images_processed, 1)
        )
        
        success_rate = (
            (self.total_images_processed - self.failed_processing_count) / 
            max(self.total_images_processed, 1)
        )
        
        stats = {
            'total_images_processed': self.total_images_processed,
            'total_processing_time_seconds': self.total_processing_time,
            'total_segments_generated': self.total_segments_generated,
            'failed_processing_count': self.failed_processing_count,
            'average_processing_time_seconds': avg_processing_time,
            'average_segments_per_image': avg_segments_per_image,
            'success_rate': success_rate,
            'model_info': {
                'model_type': self.model_type,
                'device': self.device,
                'max_image_size': self.max_image_size
            }
        }
        
        # Add GPU memory info if available
        if self.device == 'cuda' and torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2
            }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset all processing statistics."""
        self.total_images_processed = 0
        self.total_processing_time = 0.0
        self.total_segments_generated = 0
        self.failed_processing_count = 0
        
        # Clear GPU memory stats if available
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        logger.info("Processing statistics reset")
    
    def cleanup_gpu_memory(self) -> None:
        """
        Clean up GPU memory if using CUDA.
        
        This can be called periodically to free up unused GPU memory.
        """
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_sam_checkpoint(checkpoint_path: str, model_type: str = 'vit_h') -> bool:
    """
    Validate that a SAM checkpoint file is valid and compatible.
    
    This performs basic validation without loading the full model,
    which is useful for configuration validation.
    """
    try:
        if not os.path.exists(checkpoint_path):
            return False
        
        # Check file size (SAM models are large)
        file_size = os.path.getsize(checkpoint_path)
        if file_size < 100 * 1024 * 1024:  # Less than 100MB is suspicious
            logger.warning(f"SAM checkpoint file seems too small: {file_size} bytes")
            return False
        
        # Try to load just the state dict keys
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                return True
            # Some checkpoints store the state dict directly
            elif isinstance(checkpoint, dict) and len(checkpoint) > 10:
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Checkpoint validation failed: {e}")
        return False


def optimize_sam_parameters(image_dimensions: Tuple[int, int]) -> Dict[str, Any]:
    """
    Optimize SAM processing parameters based on image dimensions.
    
    Different image sizes benefit from different SAM parameters.
    This function provides intelligent defaults.
    """
    width, height = image_dimensions
    total_pixels = width * height
    
    # Adjust parameters based on image size
    if total_pixels < 500_000:  # Small images (less than ~700x700)
        return {
            'points_per_side': 16,
            'pred_iou_thresh': 0.88,
            'stability_score_thresh': 0.95,
            'min_mask_region_area': 50
        }
    elif total_pixels < 2_000_000:  # Medium images (less than ~1400x1400)
        return {
            'points_per_side': 32,
            'pred_iou_thresh': 0.86,
            'stability_score_thresh': 0.92,
            'min_mask_region_area': 100
        }
    else:  # Large images
        return {
            'points_per_side': 48,
            'pred_iou_thresh': 0.84,
            'stability_score_thresh': 0.90,
            'min_mask_region_area': 200
        }