"""
VisionFlow AI - Storage Service
===============================

This service handles all file storage, management, and processing operations:
- Image file storage and organization
- Thumbnail generation
- Annotated image creation
- Export functionality in various formats
- File cleanup and maintenance
- Storage analytics and monitoring

Think of this as the "file manager" of the system that ensures all data
is properly stored, accessible, and organized.
"""

import os
import json
import csv
import asyncio
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import aiofiles

from ..config import get_settings
from ..database import db_manager
from ..models.database_models import ImageRecord, ImageSegment, Classification


# =============================================================================
# LOGGER SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# STORAGE SERVICE CLASS
# =============================================================================

class StorageService:
    """
    Comprehensive file storage and management service.
    
    This service handles all aspects of file storage including:
    - Organizing uploaded files
    - Creating derived content (thumbnails, annotations)
    - Exporting data in various formats
    - Cleanup and maintenance operations
    - Storage analytics and reporting
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Ensure all storage directories exist
        self._ensure_directories()
        
        # Track storage statistics
        self.thumbnails_created = 0
        self.annotations_created = 0
        self.exports_created = 0
        
        logger.info("Storage service initialized")
    
    def _ensure_directories(self):
        """Create all necessary storage directories."""
        directories = [
            self.settings.upload_path,
            self.settings.segments_path,
            self.settings.results_path,
            self.settings.models_path,
            os.path.join(self.settings.results_path, "thumbnails"),
            os.path.join(self.settings.results_path, "annotated"),
            os.path.join(self.settings.results_path, "exports"),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
    
    async def create_thumbnail(
        self, 
        image_path: str, 
        size: int = 300,
        quality: int = 85
    ) -> str:
        """
        Create a thumbnail version of an image.
        
        This generates a smaller version of the image for quick previews
        in the frontend. Thumbnails are cached to avoid regeneration.
        
        Args:
            image_path: Path to the original image
            size: Maximum dimension for the thumbnail
            quality: JPEG quality (1-100)
            
        Returns:
            Path to the created thumbnail
        """
        try:
            # Generate thumbnail filename
            original_name = Path(image_path).stem
            thumbnail_dir = os.path.join(self.settings.results_path, "thumbnails")
            thumbnail_path = os.path.join(thumbnail_dir, f"{original_name}_thumb_{size}.jpg")
            
            # Return existing thumbnail if it exists and is newer than original
            if os.path.exists(thumbnail_path):
                thumb_mtime = os.path.getmtime(thumbnail_path)
                orig_mtime = os.path.getmtime(image_path)
                if thumb_mtime > orig_mtime:
                    return thumbnail_path
            
            # Create thumbnail
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Calculate new dimensions maintaining aspect ratio
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                
                # Save thumbnail
                img.save(thumbnail_path, 'JPEG', quality=quality, optimize=True)
            
            self.thumbnails_created += 1
            logger.debug(f"Thumbnail created: {thumbnail_path}")
            
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail for {image_path}: {e}")
            raise RuntimeError(f"Thumbnail creation failed: {e}")
    
    async def create_annotated_image(
        self,
        image_id: str,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_bbox: bool = True,
        font_size: int = 12,
        bbox_thickness: int = 2
    ) -> str:
        """
        Create an annotated version of an image with bounding boxes and labels.
        
        This overlays the AI processing results on the original image
        for visual review and presentation.
        
        Args:
            image_id: ID of the image to annotate
            show_labels: Whether to show classification labels
            show_confidence: Whether to show confidence scores
            show_bbox: Whether to show bounding boxes
            font_size: Size of text labels
            bbox_thickness: Thickness of bounding box lines
            
        Returns:
            Path to the annotated image
        """
        try:
            # Get image data from database
            with db_manager.get_session_context() as db:
                image_record = db.query(ImageRecord).filter(
                    ImageRecord.id == image_id
                ).first()
                
                if not image_record:
                    raise ValueError(f"Image not found: {image_id}")
                
                if not os.path.exists(image_record.file_path):
                    raise ValueError(f"Image file not found: {image_record.file_path}")
                
                # Load image
                image = cv2.imread(image_record.file_path)
                if image is None:
                    raise ValueError(f"Cannot read image: {image_record.file_path}")
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create PIL image for text rendering
                pil_image = Image.fromarray(image_rgb)
                draw = ImageDraw.Draw(pil_image)
                
                # Try to load a font
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except (OSError, IOError):
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                    except (OSError, IOError):
                        font = ImageFont.load_default()
                
                # Color palette for different labels
                colors = [
                    (255, 0, 0),    # Red
                    (0, 255, 0),    # Green
                    (0, 0, 255),    # Blue
                    (255, 255, 0),  # Yellow
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Cyan
                    (255, 128, 0),  # Orange
                    (128, 0, 255),  # Purple
                    (0, 128, 255),  # Light Blue
                    (255, 128, 128) # Pink
                ]
                
                # Track used labels for consistent coloring
                label_colors = {}
                color_index = 0
                
                # Process each segment
                for segment in image_record.segments:
                    x, y, w, h = segment.bbox_x, segment.bbox_y, segment.bbox_width, segment.bbox_height
                    
                    # Find classification for this segment
                    classification = None
                    for cls in image_record.classifications:
                        if cls.segment_id == segment.id:
                            classification = cls
                            break
                    
                    # Get label and color
                    label = classification.primary_label if classification else "Unknown"
                    
                    if label not in label_colors:
                        label_colors[label] = colors[color_index % len(colors)]
                        color_index += 1
                    
                    color = label_colors[label]
                    
                    # Draw bounding box
                    if show_bbox:
                        draw.rectangle(
                            [(x, y), (x + w, y + h)],
                            outline=color,
                            width=bbox_thickness
                        )
                    
                    # Prepare label text
                    text_parts = []
                    if show_labels and classification:
                        text_parts.append(label)
                    if show_confidence and classification:
                        confidence_text = f"{classification.confidence_score:.2f}"
                        text_parts.append(confidence_text)
                    
                    if text_parts:
                        text = " | ".join(text_parts)
                        
                        # Calculate text position
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        # Position text above bounding box, or below if not enough space
                        text_x = x
                        text_y = y - text_height - 5 if y - text_height - 5 > 0 else y + h + 5
                        
                        # Draw text background
                        background_padding = 2
                        draw.rectangle(
                            [
                                (text_x - background_padding, text_y - background_padding),
                                (text_x + text_width + background_padding, text_y + text_height + background_padding)
                            ],
                            fill=(0, 0, 0, 128)  # Semi-transparent black
                        )
                        
                        # Draw text
                        draw.text((text_x, text_y), text, fill=color, font=font)
                
                # Convert back to RGB array
                annotated_array = np.array(pil_image)
                
                # Save annotated image
                annotated_dir = os.path.join(self.settings.results_path, "annotated")
                annotated_filename = f"{Path(image_record.filename).stem}_annotated.jpg"
                annotated_path = os.path.join(annotated_dir, annotated_filename)
                
                annotated_image = Image.fromarray(annotated_array)
                annotated_image.save(annotated_path, 'JPEG', quality=90)
                
                self.annotations_created += 1
                logger.info(f"Annotated image created: {annotated_path}")
                
                return annotated_path
                
        except Exception as e:
            logger.error(f"Failed to create annotated image for {image_id}: {e}")
            raise RuntimeError(f"Annotation creation failed: {e}")
    
    async def export_to_json(self, image_id: str) -> str:
        """
        Export processing results to JSON format.
        
        This creates a comprehensive JSON export containing all
        processing results, metadata, and annotations.
        """
        try:
            with db_manager.get_session_context() as db:
                image_record = db.query(ImageRecord).filter(
                    ImageRecord.id == image_id
                ).first()
                
                if not image_record:
                    raise ValueError(f"Image not found: {image_id}")
                
                # Build export data
                export_data = {
                    "export_info": {
                        "format": "VisionFlow JSON Export",
                        "version": "1.0",
                        "exported_at": datetime.now(timezone.utc).isoformat(),
                        "image_id": str(image_record.id)
                    },
                    "image_info": {
                        "filename": image_record.filename,
                        "file_size": image_record.file_size,
                        "width": image_record.width,
                        "height": image_record.height,
                        "channels": image_record.channels,
                        "mime_type": image_record.mime_type,
                        "uploaded_at": image_record.created_at.isoformat(),
                        "processing_config": image_record.processing_config
                    },
                    "processing_info": {
                        "status": image_record.status.value,
                        "started_at": image_record.processing_started_at.isoformat() if image_record.processing_started_at else None,
                        "completed_at": image_record.processing_completed_at.isoformat() if image_record.processing_completed_at else None,
                        "processing_time_seconds": (
                            (image_record.processing_completed_at - image_record.processing_started_at).total_seconds()
                            if image_record.processing_started_at and image_record.processing_completed_at
                            else None
                        )
                    },
                    "segments": [],
                    "classifications": []
                }
                
                # Add segments
                for segment in image_record.segments:
                    segment_data = {
                        "id": str(segment.id),
                        "index": segment.segment_index,
                        "bbox": {
                            "x": segment.bbox_x,
                            "y": segment.bbox_y,
                            "width": segment.bbox_width,
                            "height": segment.bbox_height
                        },
                        "area": segment.area,
                        "confidence_score": segment.confidence_score,
                        "segment_path": segment.segment_path
                    }
                    export_data["segments"].append(segment_data)
                
                # Add classifications
                for classification in image_record.classifications:
                    classification_data = {
                        "id": str(classification.id),
                        "segment_id": str(classification.segment_id),
                        "primary_label": classification.primary_label,
                        "confidence_score": classification.confidence_score,
                        "alternative_labels": classification.alternative_labels,
                        "model_used": classification.model_used,
                        "tokens_used": classification.tokens_used,
                        "human_verified": classification.human_verified,
                        "human_label": classification.human_label,
                        "human_feedback_notes": classification.human_feedback_notes
                    }
                    export_data["classifications"].append(classification_data)
                
                # Save to file
                export_dir = os.path.join(self.settings.results_path, "exports")
                export_filename = f"{Path(image_record.filename).stem}_export.json"
                export_path = os.path.join(export_dir, export_filename)
                
                async with aiofiles.open(export_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(export_data, indent=2, ensure_ascii=False))
                
                self.exports_created += 1
                logger.info(f"JSON export created: {export_path}")
                
                return export_path
                
        except Exception as e:
            logger.error(f"Failed to export to JSON for {image_id}: {e}")
            raise RuntimeError(f"JSON export failed: {e}")
    
    async def export_to_csv(self, image_id: str) -> str:
        """
        Export processing results to CSV format.
        
        This creates a tabular export suitable for analysis
        in spreadsheet applications or data analysis tools.
        """
        try:
            with db_manager.get_session_context() as db:
                image_record = db.query(ImageRecord).filter(
                    ImageRecord.id == image_id
                ).first()
                
                if not image_record:
                    raise ValueError(f"Image not found: {image_id}")
                
                # Prepare CSV data
                csv_data = []
                
                for segment in image_record.segments:
                    # Find corresponding classification
                    classification = None
                    for cls in image_record.classifications:
                        if cls.segment_id == segment.id:
                            classification = cls
                            break
                    
                    row = {
                        'image_id': str(image_record.id),
                        'image_filename': image_record.filename,
                        'segment_id': str(segment.id),
                        'segment_index': segment.segment_index,
                        'bbox_x': segment.bbox_x,
                        'bbox_y': segment.bbox_y,
                        'bbox_width': segment.bbox_width,
                        'bbox_height': segment.bbox_height,
                        'area': segment.area,
                        'segment_confidence': segment.confidence_score,
                        'primary_label': classification.primary_label if classification else '',
                        'classification_confidence': classification.confidence_score if classification else 0.0,
                        'model_used': classification.model_used if classification else '',
                        'human_verified': classification.human_verified if classification else False,
                        'human_label': classification.human_label if classification else '',
                        'processing_time_seconds': (
                            (image_record.processing_completed_at - image_record.processing_started_at).total_seconds()
                            if image_record.processing_started_at and image_record.processing_completed_at
                            else None
                        )
                    }
                    csv_data.append(row)
                
                # Save to CSV file
                export_dir = os.path.join(self.settings.results_path, "exports")
                export_filename = f"{Path(image_record.filename).stem}_export.csv"
                export_path = os.path.join(export_dir, export_filename)
                
                if csv_data:
                    fieldnames = csv_data[0].keys()
                    
                    async with aiofiles.open(export_path, 'w', encoding='utf-8', newline='') as f:
                        # Write CSV header and data
                        content = []
                        
                        # Header
                        content.append(','.join(fieldnames))
                        
                        # Data rows
                        for row in csv_data:
                            values = []
                            for field in fieldnames:
                                value = row[field]
                                if value is None:
                                    values.append('')
                                elif isinstance(value, str) and (',' in value or '"' in value):
                                    values.append(f'"{value.replace('"', '""')}"')
                                else:
                                    values.append(str(value))
                            content.append(','.join(values))
                        
                        await f.write('\n'.join(content))
                else:
                    # Empty CSV with headers only
                    headers = ['image_id', 'image_filename', 'message']
                    async with aiofiles.open(export_path, 'w', encoding='utf-8', newline='') as f:
                        await f.write(','.join(headers) + '\n')
                        await f.write(f'{image_id},{image_record.filename},No segments found\n')
                
                self.exports_created += 1
                logger.info(f"CSV export created: {export_path}")
                
                return export_path
                
        except Exception as e:
            logger.error(f"Failed to export to CSV for {image_id}: {e}")
            raise RuntimeError(f"CSV export failed: {e}")
    
    async def export_to_coco(self, image_id: str) -> str:
        """
        Export processing results to COCO format.
        
        This creates an export compatible with the COCO dataset format,
        useful for computer vision research and training.
        """
        try:
            with db_manager.get_session_context() as db:
                image_record = db.query(ImageRecord).filter(
                    ImageRecord.id == image_id
                ).first()
                
                if not image_record:
                    raise ValueError(f"Image not found: {image_id}")
                
                # Create COCO format data
                coco_data = {
                    "info": {
                        "description": "VisionFlow AI Export - COCO Format",
                        "version": "1.0",
                        "year": datetime.now().year,
                        "contributor": "VisionFlow AI",
                        "date_created": datetime.now(timezone.utc).isoformat()
                    },
                    "images": [
                        {
                            "id": 1,
                            "width": image_record.width,
                            "height": image_record.height,
                            "file_name": image_record.filename,
                            "date_captured": image_record.created_at.isoformat()
                        }
                    ],
                    "annotations": [],
                    "categories": []
                }
                
                # Collect unique categories
                categories = set()
                for classification in image_record.classifications:
                    categories.add(classification.primary_label)
                
                # Add categories to COCO data
                category_map = {}
                for i, category in enumerate(sorted(categories), 1):
                    coco_data["categories"].append({
                        "id": i,
                        "name": category,
                        "supercategory": "object"
                    })
                    category_map[category] = i
                
                # Add annotations
                annotation_id = 1
                for segment in image_record.segments:
                    # Find classification
                    classification = None
                    for cls in image_record.classifications:
                        if cls.segment_id == segment.id:
                            classification = cls
                            break
                    
                    if classification:
                        annotation = {
                            "id": annotation_id,
                            "image_id": 1,
                            "category_id": category_map.get(classification.primary_label, 1),
                            "bbox": [
                                segment.bbox_x,
                                segment.bbox_y,
                                segment.bbox_width,
                                segment.bbox_height
                            ],
                            "area": segment.area,
                            "iscrowd": 0,
                            "confidence": classification.confidence_score
                        }
                        coco_data["annotations"].append(annotation)
                        annotation_id += 1
                
                # Save to file
                export_dir = os.path.join(self.settings.results_path, "exports")
                export_filename = f"{Path(image_record.filename).stem}_coco.json"
                export_path = os.path.join(export_dir, export_filename)
                
                async with aiofiles.open(export_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(coco_data, indent=2))
                
                self.exports_created += 1
                logger.info(f"COCO export created: {export_path}")
                
                return export_path
                
        except Exception as e:
            logger.error(f"Failed to export to COCO for {image_id}: {e}")
            raise RuntimeError(f"COCO export failed: {e}")
    
    async def export_to_yolo(self, image_id: str) -> str:
        """
        Export processing results to YOLO format.
        
        This creates an export compatible with YOLO training format.
        """
        try:
            with db_manager.get_session_context() as db:
                image_record = db.query(ImageRecord).filter(
                    ImageRecord.id == image_id
                ).first()
                
                if not image_record:
                    raise ValueError(f"Image not found: {image_id}")
                
                # Collect unique labels for class mapping
                labels = set()
                for classification in image_record.classifications:
                    labels.add(classification.primary_label)
                
                label_to_id = {label: i for i, label in enumerate(sorted(labels))}
                
                # Create YOLO format annotations
                yolo_lines = []
                image_width = image_record.width or 1
                image_height = image_record.height or 1
                
                for segment in image_record.segments:
                    # Find classification
                    classification = None
                    for cls in image_record.classifications:
                        if cls.segment_id == segment.id:
                            classification = cls
                            break
                    
                    if classification:
                        # Convert bbox to YOLO format (normalized center x, center y, width, height)
                        x = segment.bbox_x
                        y = segment.bbox_y
                        w = segment.bbox_width
                        h = segment.bbox_height
                        
                        center_x = (x + w / 2) / image_width
                        center_y = (y + h / 2) / image_height
                        norm_width = w / image_width
                        norm_height = h / image_height
                        
                        class_id = label_to_id.get(classification.primary_label, 0)
                        
                        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                        yolo_lines.append(yolo_line)
                
                # Save annotations file
                export_dir = os.path.join(self.settings.results_path, "exports")
                export_filename = f"{Path(image_record.filename).stem}_yolo.txt"
                export_path = os.path.join(export_dir, export_filename)
                
                async with aiofiles.open(export_path, 'w', encoding='utf-8') as f:
                    await f.write('\n'.join(yolo_lines))
                
                # Also save class names file
                classes_filename = f"{Path(image_record.filename).stem}_classes.txt"
                classes_path = os.path.join(export_dir, classes_filename)
                
                async with aiofiles.open(classes_path, 'w', encoding='utf-8') as f:
                    await f.write('\n'.join(sorted(labels)))
                
                self.exports_created += 1
                logger.info(f"YOLO export created: {export_path}")
                
                return export_path
                
        except Exception as e:
            logger.error(f"Failed to export to YOLO for {image_id}: {e}")
            raise RuntimeError(f"YOLO export failed: {e}")
    
    async def export_batch_to_json(self, image_ids: List[str]) -> str:
        """
        Export multiple images to a single JSON file.
        
        This creates a batch export containing all specified images
        in a single JSON document.
        """
        try:
            batch_data = {
                "export_info": {
                    "format": "VisionFlow Batch JSON Export",
                    "version": "1.0",
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "image_count": len(image_ids)
                },
                "images": []
            }
            
            # Export each image
            for image_id in image_ids:
                # Get individual export data
                temp_export_path = await self.export_to_json(image_id)
                
                # Read the export data
                async with aiofiles.open(temp_export_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    image_data = json.loads(content)
                
                batch_data["images"].append(image_data)
                
                # Clean up temporary file
                os.remove(temp_export_path)
            
            # Save batch export
            export_dir = os.path.join(self.settings.results_path, "exports")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"batch_export_{len(image_ids)}_images_{timestamp}.json"
            export_path = os.path.join(export_dir, export_filename)
            
            async with aiofiles.open(export_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(batch_data, indent=2, ensure_ascii=False))
            
            logger.info(f"Batch JSON export created: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to create batch JSON export: {e}")
            raise RuntimeError(f"Batch JSON export failed: {e}")
    
    async def export_batch_to_csv(self, image_ids: List[str]) -> str:
        """
        Export multiple images to a single CSV file.
        
        This creates a batch export containing all specified images
        in a single CSV document.
        """
        try:
            all_rows = []
            
            # Collect data from all images
            for image_id in image_ids:
                # Get individual CSV data
                temp_export_path = await self.export_to_csv(image_id)
                
                # Read CSV data (skip header for all but first file)
                async with aiofiles.open(temp_export_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    lines = content.strip().split('\n')
                    
                    if image_id == image_ids[0]:
                        # Include header for first file
                        all_rows.extend(lines)
                    else:
                        # Skip header for subsequent files
                        all_rows.extend(lines[1:])
                
                # Clean up temporary file
                os.remove(temp_export_path)
            
            # Save batch export
            export_dir = os.path.join(self.settings.results_path, "exports")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"batch_export_{len(image_ids)}_images_{timestamp}.csv"
            export_path = os.path.join(export_dir, export_filename)
            
            async with aiofiles.open(export_path, 'w', encoding='utf-8') as f:
                await f.write('\n'.join(all_rows))
            
            logger.info(f"Batch CSV export created: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to create batch CSV export: {e}")
            raise RuntimeError(f"Batch CSV export failed: {e}")
    
    async def cleanup_image_files(self, image_id: str):
        """
        Clean up all files associated with an image.
        
        This removes thumbnails, annotations, exports, and segment files
        when an image is deleted from the system.
        """
        try:
            # Clean up thumbnails
            thumbnail_dir = os.path.join(self.settings.results_path, "thumbnails")
            for file in os.listdir(thumbnail_dir):
                if image_id in file:
                    file_path = os.path.join(thumbnail_dir, file)
                    os.remove(file_path)
                    logger.debug(f"Removed thumbnail: {file_path}")
            
            # Clean up annotated images
            annotated_dir = os.path.join(self.settings.results_path, "annotated")
            for file in os.listdir(annotated_dir):
                if image_id in file:
                    file_path = os.path.join(annotated_dir, file)
                    os.remove(file_path)
                    logger.debug(f"Removed annotated image: {file_path}")
            
            # Clean up exports
            export_dir = os.path.join(self.settings.results_path, "exports")
            for file in os.listdir(export_dir):
                if image_id in file:
                    file_path = os.path.join(export_dir, file)
                    os.remove(file_path)
                    logger.debug(f"Removed export: {file_path}")
            
            # Clean up segments
            segments_base = Path(self.settings.segments_path)
            for segment_dir in segments_base.glob(f"*{image_id}*"):
                if segment_dir.is_dir():
                    shutil.rmtree(segment_dir)
                    logger.debug(f"Removed segment directory: {segment_dir}")
            
            logger.info(f"Cleanup completed for image: {image_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup files for image {image_id}: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.
        
        This provides insights into disk usage and file counts
        for monitoring and capacity planning.
        """
        try:
            stats = {
                "directories": {},
                "totals": {
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "total_size_gb": 0.0
                },
                "operations": {
                    "thumbnails_created": self.thumbnails_created,
                    "annotations_created": self.annotations_created,
                    "exports_created": self.exports_created
                }
            }
            
            # Analyze each directory
            directories = [
                ("uploads", self.settings.upload_path),
                ("segments", self.settings.segments_path),
                ("results", self.settings.results_path),
                ("models", self.settings.models_path)
            ]
            
            for name, path in directories:
                if os.path.exists(path):
                    dir_stats = self._analyze_directory(path)
                    stats["directories"][name] = dir_stats
                    stats["totals"]["total_files"] += dir_stats["file_count"]
                    stats["totals"]["total_size_bytes"] += dir_stats["size_bytes"]
            
            stats["totals"]["total_size_gb"] = round(
                stats["totals"]["total_size_bytes"] / (1024**3), 2
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    def _analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze a directory and return size/file statistics."""
        total_size = 0
        file_count = 0
        
        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                    except (OSError, IOError):
                        # Skip files that can't be accessed
                        continue
            
            return {
                "path": directory_path,
                "file_count": file_count,
                "size_bytes": total_size,
                "size_mb": round(total_size / (1024**2), 2),
                "size_gb": round(total_size / (1024**3), 2)
            }
            
        except Exception as e:
            return {
                "path": directory_path,
                "error": str(e),
                "file_count": 0,
                "size_bytes": 0
            }


# =============================================================================
# SERVICE FACTORY FUNCTION
# =============================================================================

_storage_service_instance = None

def get_storage_service() -> StorageService:
    """
    Get singleton instance of storage service.
    
    Using a singleton ensures consistent file management
    and helps with resource tracking.
    """
    global _storage_service_instance
    
    if _storage_service_instance is None:
        _storage_service_instance = StorageService()
    
    return _storage_service_instance