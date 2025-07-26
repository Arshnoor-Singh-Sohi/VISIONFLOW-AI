"""
VisionFlow AI - Image Processing Utilities
==========================================

This module provides utility functions for image processing, validation,
metadata extraction, and format conversion. These utilities support
the core image processing pipeline and ensure reliable handling of
various image formats and conditions.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import mimetypes

import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags

# Import magic with graceful fallback for systems where libmagic isn't available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None


# =============================================================================
# LOGGER SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# IMAGE VALIDATION FUNCTIONS
# =============================================================================

def validate_image_file(file_path: str, max_size_mb: int = 50) -> Dict[str, Any]:
    """
    Validate an image file for processing compatibility.
    
    This performs comprehensive validation including format checking,
    size limits, corruption detection, and metadata extraction.
    
    Args:
        file_path: Path to the image file
        max_size_mb: Maximum allowed file size in megabytes
        
    Returns:
        Dictionary with validation results and metadata
    """
    try:
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            return {
                'valid': False,
                'error': 'File does not exist',
                'file_path': str(file_path)
            }
        
        # Check file size
        file_size = file_path.stat().st_size
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size == 0:
            return {
                'valid': False,
                'error': 'File is empty',
                'file_size': file_size
            }
        
        if file_size > max_size_bytes:
            return {
                'valid': False,
                'error': f'File too large ({file_size / (1024*1024):.1f}MB > {max_size_mb}MB)',
                'file_size': file_size
            }
        
        # Check MIME type
        # Check MIME type using multiple detection methods
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # If mimetypes didn't detect an image, try magic as fallback (if available)
        if not mime_type or not mime_type.startswith('image/'):
            if MAGIC_AVAILABLE:
                try:
                    mime_type = magic.from_file(str(file_path), mime=True)
                except Exception:
                    # If magic fails, we'll rely on PIL verification later
                    mime_type = 'application/octet-stream'
            else:
                # No magic available, we'll rely on PIL verification later
                mime_type = 'application/octet-stream'
        
        # For unknown MIME types, we'll let PIL try to validate the image
        # This handles cases where magic isn't available or MIME detection fails
        if mime_type.startswith('image/'):
            detected_as_image = True
        elif mime_type in ('application/octet-stream', 'unknown'):
            # Unknown type - we'll validate with PIL later
            detected_as_image = False
        else:
            # Definitely not an image
            return {
                'valid': False,
                'error': f'Not an image file (detected: {mime_type})',
                'mime_type': mime_type
            }
        
        # Try to open and validate with PIL
        try:
            with Image.open(file_path) as img:
                # Verify image can be loaded
                img.verify()
                
                # Reopen for metadata extraction (verify() closes the file)
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    format_name = img.format

                    # If we couldn't detect MIME type properly, use PIL's format detection
                    if not detected_as_image:
                        mime_type = f'image/{format_name.lower()}' if format_name else 'image/jpeg'
                    
                    # Check minimum dimensions
                    if width < 10 or height < 10:
                        return {
                            'valid': False,
                            'error': f'Image too small ({width}x{height})',
                            'dimensions': (width, height)
                        }
                    
                    # Check maximum dimensions
                    max_dimension = 8192
                    if width > max_dimension or height > max_dimension:
                        return {
                            'valid': False,
                            'error': f'Image too large ({width}x{height})',
                            'dimensions': (width, height)
                        }
                    
                    # Extract additional metadata
                    metadata = get_image_metadata(str(file_path))
                    
                    return {
                        'valid': True,
                        'file_path': str(file_path),
                        'file_size': file_size,
                        'mime_type': mime_type,
                        'format': format_name,
                        'dimensions': (width, height),
                        'mode': mode,
                        'metadata': metadata
                    }
                    
        except Exception as e:
            return {
                'valid': False,
                'error': f'Corrupted or invalid image file: {e}',
                'exception_type': type(e).__name__
            }
            
    except Exception as e:
        logger.error(f"Image validation failed for {file_path}: {e}")
        return {
            'valid': False,
            'error': f'Validation failed: {e}',
            'exception_type': type(e).__name__
        }


def get_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from an image file.
    
    This extracts EXIF data, color information, and other relevant
    metadata that might be useful for processing or analysis.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing extracted metadata
    """
    try:
        metadata = {
            'width': None,
            'height': None,
            'channels': None,
            'format': None,
            'mode': None,
            'has_transparency': False,
            'color_profile': None,
            'exif': {},
            'file_size': 0
        }
        
        # Basic file info
        file_path = Path(image_path)
        metadata['file_size'] = file_path.stat().st_size
        
        with Image.open(image_path) as img:
            # Basic image properties
            metadata['width'], metadata['height'] = img.size
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            metadata['channels'] = len(img.getbands()) if img.getbands() else 1
            metadata['has_transparency'] = 'transparency' in img.info or 'A' in img.mode
            
            # Color profile
            if hasattr(img, 'info') and 'icc_profile' in img.info:
                metadata['color_profile'] = 'ICC Profile present'
            
            # EXIF data extraction
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif_dict = img._getexif()
                
                for tag_id, value in exif_dict.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    
                    # Convert problematic types to strings
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)
                    elif isinstance(value, (tuple, list)) and len(value) > 10:
                        # Truncate very long sequences
                        value = f"[{len(value)} items]"
                    
                    metadata['exif'][tag] = value
            
            # Handle image orientation from EXIF
            orientation = metadata['exif'].get('Orientation', 1)
            if orientation and orientation != 1:
                metadata['needs_rotation'] = True
                metadata['orientation'] = orientation
            else:
                metadata['needs_rotation'] = False
        
        # Additional analysis with OpenCV
        try:
            cv_img = cv2.imread(image_path)
            if cv_img is not None:
                # Color statistics
                mean_color = cv_img.mean(axis=(0, 1))
                metadata['mean_color_bgr'] = [float(x) for x in mean_color]
                
                # Brightness analysis
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                metadata['mean_brightness'] = float(gray.mean())
                metadata['brightness_std'] = float(gray.std())
                
                # Check if image is likely grayscale
                if len(mean_color) == 3:
                    color_variance = np.var(mean_color)
                    metadata['is_likely_grayscale'] = color_variance < 10
        except Exception as cv_error:
            logger.debug(f"OpenCV analysis failed for {image_path}: {cv_error}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to extract metadata from {image_path}: {e}")
        return {
            'error': str(e),
            'width': 0,
            'height': 0,
            'channels': 3
        }


# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================

def resize_image_if_needed(
    image_path: str,
    max_dimension: int = 2048,
    quality: int = 90,
    output_path: Optional[str] = None
) -> str:
    """
    Resize an image if it exceeds the maximum dimension.
    
    This maintains aspect ratio while ensuring the image fits within
    the specified maximum dimension. Useful for reducing processing
    time and memory usage.
    
    Args:
        image_path: Path to the input image
        max_dimension: Maximum width or height
        quality: JPEG quality for output (1-100)
        output_path: Path for output image (defaults to overwriting input)
        
    Returns:
        Path to the processed image
    """
    try:
        output_path = output_path or image_path
        
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            
            # Check if resizing is needed
            if max(original_width, original_height) <= max_dimension:
                # Copy to output path if different
                if output_path != image_path:
                    img.save(output_path, quality=quality, optimize=True)
                return output_path
            
            # Calculate new dimensions
            if original_width > original_height:
                new_width = max_dimension
                new_height = int(original_height * (max_dimension / original_width))
            else:
                new_height = max_dimension
                new_width = int(original_width * (max_dimension / original_height))
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Handle different formats
            if img.mode in ('RGBA', 'LA', 'P'):
                # Convert to RGB for JPEG output
                rgb_img = Image.new('RGB', resized_img.size, (255, 255, 255))
                if resized_img.mode == 'P':
                    resized_img = resized_img.convert('RGBA')
                rgb_img.paste(resized_img, mask=resized_img.split()[-1] if 'A' in resized_img.mode else None)
                resized_img = rgb_img
            
            # Save resized image
            resized_img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            logger.info(f"Image resized from {original_width}x{original_height} to {new_width}x{new_height}")
            return output_path
            
    except Exception as e:
        logger.error(f"Failed to resize image {image_path}: {e}")
        raise RuntimeError(f"Image resizing failed: {e}")


def correct_image_orientation(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Correct image orientation based on EXIF data.
    
    Many images from mobile devices have EXIF orientation data that
    needs to be applied to display the image correctly.
    
    Args:
        image_path: Path to the input image
        output_path: Path for output image (defaults to overwriting input)
        
    Returns:
        Path to the corrected image
    """
    try:
        output_path = output_path or image_path
        
        with Image.open(image_path) as img:
            # Use PIL's built-in orientation correction
            corrected_img = ImageOps.exif_transpose(img)
            
            # Save corrected image
            if corrected_img != img:  # Only save if changes were made
                corrected_img.save(output_path, quality=90, optimize=True)
                logger.info(f"Image orientation corrected: {image_path}")
            elif output_path != image_path:
                # Copy to output path if different and no correction needed
                img.save(output_path, quality=90, optimize=True)
            
            return output_path
            
    except Exception as e:
        logger.error(f"Failed to correct orientation for {image_path}: {e}")
        # If correction fails, just return the original path
        return image_path


def normalize_image_format(
    image_path: str,
    target_format: str = 'JPEG',
    quality: int = 90,
    output_path: Optional[str] = None
) -> str:
    """
    Convert an image to a standardized format.
    
    This ensures all images are in a consistent format for processing,
    which can help avoid compatibility issues downstream.
    
    Args:
        image_path: Path to the input image
        target_format: Target format ('JPEG', 'PNG', etc.)
        quality: Quality for lossy formats (1-100)
        output_path: Path for output image
        
    Returns:
        Path to the normalized image
    """
    try:
        if not output_path:
            base_path = Path(image_path)
            if target_format.upper() == 'JPEG':
                output_path = base_path.with_suffix('.jpg')
            else:
                output_path = base_path.with_suffix(f'.{target_format.lower()}')
        
        with Image.open(image_path) as img:
            # Handle transparency for JPEG conversion
            if target_format.upper() == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
                img = rgb_img
            
            # Save in target format
            save_kwargs = {'optimize': True}
            if target_format.upper() in ('JPEG', 'WEBP'):
                save_kwargs['quality'] = quality
            
            img.save(str(output_path), target_format, **save_kwargs)
            
            logger.info(f"Image format normalized to {target_format}: {output_path}")
            return str(output_path)
            
    except Exception as e:
        logger.error(f"Failed to normalize image format for {image_path}: {e}")
        raise RuntimeError(f"Format normalization failed: {e}")


def create_annotated_image(
    image_path: str,
    annotations: List[Dict[str, Any]],
    output_path: str,
    font_size: int = 12,
    line_thickness: int = 2
) -> str:
    """
    Create an annotated version of an image with bounding boxes and labels.
    
    This overlays bounding boxes and text labels on an image for
    visualization of detection/classification results.
    
    Args:
        image_path: Path to the input image
        annotations: List of annotation dictionaries with bbox and label info
        output_path: Path for the annotated output image
        font_size: Size of text labels
        line_thickness: Thickness of bounding box lines
        
    Returns:
        Path to the annotated image
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Color palette for different labels
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue (BGR format)
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Light Green
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (255, 0, 128),  # Pink
        ]
        
        # Track labels for consistent coloring
        label_colors = {}
        color_index = 0
        
        for annotation in annotations:
            bbox = annotation.get('bbox', [])
            label = annotation.get('label', 'Unknown')
            confidence = annotation.get('confidence', 0.0)
            
            if len(bbox) != 4:
                continue
            
            x, y, w, h = map(int, bbox)
            
            # Get color for this label
            if label not in label_colors:
                label_colors[label] = colors[color_index % len(colors)]
                color_index += 1
            
            color = label_colors[label]
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, line_thickness)
            
            # Prepare label text
            if confidence > 0:
                text = f"{label} ({confidence:.2f})"
            else:
                text = label
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = font_size / 20.0  # Scale factor for font size
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            
            # Position text above bounding box, or below if not enough space
            text_x = x
            text_y = y - 10 if y - 10 > text_size[1] else y + h + text_size[1] + 10
            
            # Draw text background rectangle
            bg_x1, bg_y1 = text_x - 2, text_y - text_size[1] - 2
            bg_x2, bg_y2 = text_x + text_size[0] + 2, text_y + 2
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(image, text, (text_x, text_y), font, font_scale, color, 1)
        
        # Save annotated image
        cv2.imwrite(output_path, image)
        
        logger.info(f"Annotated image created: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create annotated image: {e}")
        raise RuntimeError(f"Annotation creation failed: {e}")


# =============================================================================
# IMAGE ANALYSIS FUNCTIONS
# =============================================================================

def analyze_image_quality(image_path: str) -> Dict[str, Any]:
    """
    Analyze image quality metrics.
    
    This provides insights into image quality that can be useful
    for filtering or preprocessing decisions.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness statistics
        brightness_mean = float(gray.mean())
        brightness_std = float(gray.std())
        
        # Contrast (using standard deviation)
        contrast = brightness_std
        
        # Exposure analysis
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Check for over/under exposure
        total_pixels = gray.shape[0] * gray.shape[1]
        dark_pixels = histogram[:50].sum() / total_pixels  # Very dark pixels
        bright_pixels = histogram[200:].sum() / total_pixels  # Very bright pixels
        
        # Color analysis
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            color_variance = np.var([b.mean(), g.mean(), r.mean()])
        else:
            color_variance = 0.0
        
        # Noise estimation (using high-frequency content)
        noise_estimate = float(cv2.Laplacian(gray, cv2.CV_64F).std())
        
        quality_metrics = {
            'sharpness': float(laplacian_var),
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'contrast': contrast,
            'dark_pixels_ratio': float(dark_pixels),
            'bright_pixels_ratio': float(bright_pixels),
            'color_variance': float(color_variance),
            'noise_estimate': noise_estimate,
            'is_likely_blurry': laplacian_var < 100,
            'is_likely_overexposed': bright_pixels > 0.1,
            'is_likely_underexposed': dark_pixels > 0.3,
            'is_likely_low_contrast': contrast < 30
        }
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Failed to analyze image quality for {image_path}: {e}")
        return {
            'error': str(e),
            'sharpness': 0.0,
            'brightness_mean': 0.0,
            'contrast': 0.0
        }


def extract_color_features(image_path: str) -> Dict[str, Any]:
    """
    Extract color-based features from an image.
    
    This analyzes color distribution and properties that might
    be useful for classification or analysis.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with color features
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract channel statistics
        bgr_means = [float(image[:, :, i].mean()) for i in range(3)]
        bgr_stds = [float(image[:, :, i].std()) for i in range(3)]
        
        hsv_means = [float(hsv[:, :, i].mean()) for i in range(3)]
        hsv_stds = [float(hsv[:, :, i].std()) for i in range(3)]
        
        # Dominant colors (simplified)
        pixels = image.reshape((-1, 3))
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Get top 5 most common colors
        top_indices = np.argsort(counts)[-5:][::-1]
        dominant_colors = [
            {
                'color_bgr': [int(c) for c in unique_colors[i]],
                'count': int(counts[i]),
                'percentage': float(counts[i] / len(pixels) * 100)
            }
            for i in top_indices
        ]
        
        color_features = {
            'bgr_means': bgr_means,
            'bgr_stds': bgr_stds,
            'hsv_means': hsv_means,
            'hsv_stds': hsv_stds,
            'dominant_colors': dominant_colors,
            'color_richness': len(unique_colors),
            'color_diversity': float(len(unique_colors) / len(pixels))
        }
        
        return color_features
        
    except Exception as e:
        logger.error(f"Failed to extract color features from {image_path}: {e}")
        return {
            'error': str(e),
            'bgr_means': [0.0, 0.0, 0.0],
            'hsv_means': [0.0, 0.0, 0.0]
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about an image file.
    
    This combines validation, metadata extraction, and quality analysis
    into a single comprehensive report.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with complete image information
    """
    try:
        # Basic validation
        validation_result = validate_image_file(image_path)
        
        if not validation_result['valid']:
            return validation_result
        
        # Get detailed metadata
        metadata = get_image_metadata(image_path)
        
        # Quality analysis
        quality_metrics = analyze_image_quality(image_path)
        
        # Combine all information
        image_info = {
            'validation': validation_result,
            'metadata': metadata,
            'quality': quality_metrics,
            'recommendations': []
        }
        
        # Add processing recommendations
        if quality_metrics.get('is_likely_blurry', False):
            image_info['recommendations'].append('Image appears blurry - may affect classification accuracy')
        
        if quality_metrics.get('is_likely_overexposed', False):
            image_info['recommendations'].append('Image may be overexposed')
        
        if quality_metrics.get('is_likely_underexposed', False):
            image_info['recommendations'].append('Image may be underexposed')
        
        if metadata.get('needs_rotation', False):
            image_info['recommendations'].append('Image orientation should be corrected')
        
        if not image_info['recommendations']:
            image_info['recommendations'].append('Image looks good for processing')
        
        return image_info
        
    except Exception as e:
        logger.error(f"Failed to get image info for {image_path}: {e}")
        return {
            'valid': False,
            'error': str(e),
            'file_path': image_path
        }


def convert_image_format(
    input_path: str,
    output_path: str,
    target_format: str = 'JPEG',
    **kwargs
) -> bool:
    """
    Convert an image from one format to another.
    
    This is a convenience function that handles format conversion
    with appropriate settings for different target formats.
    
    Args:
        input_path: Path to input image
        output_path: Path for output image
        target_format: Target format ('JPEG', 'PNG', 'WEBP', etc.)
        **kwargs: Additional arguments for format-specific options
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        with Image.open(input_path) as img:
            # Handle transparency for formats that don't support it
            if target_format.upper() in ('JPEG', 'BMP') and img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Set default quality for lossy formats
            save_kwargs = kwargs.copy()
            if target_format.upper() in ('JPEG', 'WEBP') and 'quality' not in save_kwargs:
                save_kwargs['quality'] = 90
            
            if 'optimize' not in save_kwargs:
                save_kwargs['optimize'] = True
            
            img.save(output_path, target_format, **save_kwargs)
            
        logger.info(f"Image converted from {input_path} to {output_path} ({target_format})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert image {input_path} to {target_format}: {e}")
        return False


def batch_process_images(
    image_paths: List[str],
    output_dir: str,
    processing_func: callable,
    **kwargs
) -> Dict[str, Any]:
    """
    Process multiple images in batch with a given processing function.
    
    This provides a convenient way to apply the same processing
    to multiple images with progress tracking and error handling.
    
    Args:
        image_paths: List of input image paths
        output_dir: Directory for output images
        processing_func: Function to apply to each image
        **kwargs: Additional arguments for processing function
        
    Returns:
        Dictionary with processing results and statistics
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'total_images': len(image_paths),
            'successful': 0,
            'failed': 0,
            'results': [],
            'errors': []
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                # Generate output path
                input_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{input_name}_processed.jpg")
                
                # Apply processing function
                result = processing_func(image_path, output_path, **kwargs)
                
                results['results'].append({
                    'input_path': image_path,
                    'output_path': output_path,
                    'success': True,
                    'result': result
                })
                results['successful'] += 1
                
            except Exception as e:
                error_info = {
                    'input_path': image_path,
                    'error': str(e),
                    'index': i
                }
                results['errors'].append(error_info)
                results['failed'] += 1
                
                logger.error(f"Failed to process {image_path}: {e}")
        
        success_rate = (results['successful'] / results['total_images']) * 100
        logger.info(f"Batch processing completed: {success_rate:.1f}% success rate")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return {
            'total_images': len(image_paths),
            'successful': 0,
            'failed': len(image_paths),
            'error': str(e)
        }