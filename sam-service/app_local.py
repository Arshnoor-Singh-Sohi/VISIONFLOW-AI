"""
VisionFlow AI - SAM Processing Service (Local Development Version)
================================================================

This is a standalone FastAPI service that runs the Segment Anything Model (SAM)
for local development on Windows. It's adapted from the Docker version to work
with local file paths and development workflows.
"""

import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from io import BytesIO

import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# =============================================================================
# LOCAL DEVELOPMENT CONFIGURATION
# =============================================================================

# Get the project root directory (parent of sam-service)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Local development paths (Windows-compatible)
MODEL_PATH = str(DATA_DIR / "models" / "sam_vit_h_4b8939.pth")
DEVICE = os.getenv('DEVICE', 'cpu')  # 'cpu', 'cuda', or 'mps'
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2048'))
TEMP_DIR = DATA_DIR / "temp"
SEGMENTS_DIR = DATA_DIR / "segments"

# Create necessary directories
TEMP_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup basic logging for development
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# SIMPLIFIED SAM PROCESSOR FOR LOCAL DEVELOPMENT
# =============================================================================

class LocalSAMProcessor:
    """Simplified SAM processor for local development and testing."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.sam_model = None
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=16,  # Reduced from 32
            pred_iou_thresh=0.88,
            stability_score_thresh=0.94,
            crop_n_layers=0,  # Reduced from 1
            min_mask_region_area=500,  # Increased to filter out smaller segments
        )
        self.stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'segments_generated': 0
        }
        
        self._load_model()
    
    # def _load_model(self):
    #     """Load the SAM model with error handling."""
    #     try:
    #         logger.info(f"Loading SAM model from {self.model_path}")
            
    #         if not os.path.exists(self.model_path):
    #             raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
    #         # Import SAM components
    #         from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
    #         # Determine device
    #         if self.device == 'cuda' and not torch.cuda.is_available():
    #             logger.warning("CUDA requested but not available, using CPU")
    #             self.device = 'cpu'
            
    #         # Load model
    #         self.sam_model = sam_model_registry['vit_h'](checkpoint=self.model_path)
    #         self.sam_model.to(device=self.device)
            
    #         # Create mask generator
    #         self.mask_generator = SamAutomaticMaskGenerator(
    #             model=self.sam_model,
    #             points_per_side=32,
    #             pred_iou_thresh=0.86,
    #             stability_score_thresh=0.92,
    #             crop_n_layers=1,
    #             min_mask_region_area=100,
    #         )
            
    #         logger.info(f"SAM model loaded successfully on {self.device}")
            
    #     except Exception as e:
    #         logger.error(f"Failed to load SAM model: {e}")
    #         raise
    
    def _load_model(self):
        """Skip model loading for mock testing."""
        logger.info("Skipping SAM model loading - using mocks")
        self.sam_model = None
        self.mask_generator = None

    async def process_image(self, image_path: str, min_area: int = 1000, 
                          max_segments: int = 60, confidence_threshold: float = 0.7):
        """Process an image and return segments."""
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate masks
            masks = self.mask_generator.generate(image_rgb)
            
            # Process masks into segments
            segments = []
            for i, mask_data in enumerate(masks):
                if i >= max_segments:
                    break
                
                area = mask_data['area']
                if area < min_area:
                    continue
                
                bbox = mask_data['bbox']
                confidence = (mask_data.get('predicted_iou', 1.0) + 
                            mask_data.get('stability_score', 1.0)) / 2.0
                
                if confidence < confidence_threshold:
                    continue
                
                segment = {
                    'id': len(segments),
                    'bbox': [int(x) for x in bbox],
                    'area': int(area),
                    'confidence': float(confidence),
                    'segment_image_path': None  # We'll add this if needed
                }
                segments.append(segment)
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats['images_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['segments_generated'] += len(segments)
            
            return {
                'segments': segments,
                'processing_time_seconds': processing_time,
                'total_segments_found': len(segments)
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def get_stats(self):
        """Get processing statistics."""
        return self.stats.copy()

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="VisionFlow SAM Service (Local)",
    description="Local development version of SAM processing service",
    version="1.0.0-local"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance
sam_processor: Optional[LocalSAMProcessor] = None

# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize SAM model on startup."""
    global sam_processor
    
    logger.info("Starting VisionFlow SAM Service (Local Development)")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"SAM model file not found at: {MODEL_PATH}")
            logger.info("Please ensure the model file 'sam_vit_h_4b8939.pth' is in the data/models directory")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        # Initialize processor
        # sam_processor = LocalSAMProcessor(MODEL_PATH, DEVICE)
        sam_processor = "mock"
        logger.info("SAM service initialized successfully!")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = sam_processor is not None
    
    return {
        "status": "healthy" if model_loaded else "initializing",
        "model_loaded": model_loaded,
        "device": DEVICE,
        "model_path": MODEL_PATH,
        "project_root": str(PROJECT_ROOT),
        "response_time_ms": time.time() * 1000
    }

@app.post("/segment")
async def segment_image(
    image: UploadFile = File(...),
    config: str = Form(...)
):
    """Segment an uploaded image."""
    if sam_processor is None:
        raise HTTPException(status_code=503, detail="SAM model not loaded")
    
    temp_image_path = None
    
    try:
        # Parse config
        config_data = json.loads(config)
        
        # # Validate image
        # if not image.content_type or not image.content_type.startswith('image/'):
        #     raise HTTPException(status_code=422, detail="File must be an image")
        
        # Save temporary image
        temp_image_path = TEMP_DIR / f"temp_{int(time.time())}_{image.filename}"
        content = await image.read()
        
        with open(temp_image_path, 'wb') as f:
            f.write(content)
        
        try:
            logger.info(f"About to process image: {temp_image_path}")
            
            # Process with SAM
            # 
            
            logger.info("Using mock SAM response for testing")
            result = {
                'segments': [
                    {'id': 0, 'bbox': [10, 10, 100, 100], 'area': 10000, 'confidence': 0.9},
                    {'id': 1, 'bbox': [50, 50, 80, 80], 'area': 6400, 'confidence': 0.8},
                    {'id': 2, 'bbox': [120, 20, 60, 90], 'area': 5400, 'confidence': 0.85}
                ],
                'processing_time_seconds': 0.1,
                'total_segments_found': 3
            }
            logger.info("Mock SAM processing completed")

            
        except Exception as e:
            logger.error(f"SAM processing failed with error type: {type(e).__name__}")
            logger.error(f"Error message: {repr(e)}")
            logger.error(f"Error args: {e.args}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {repr(e)}")
        
        return {
            'success': True,
            'segments': result['segments'],
            'processing_time_seconds': result['processing_time_seconds'],
            'model_info': {
                'model_type': 'SAM',
                'device': DEVICE,
                'total_segments_found': result['total_segments_found']
            }
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON configuration")
    except Exception as e:
        logger.error(f"Segmentation failed: {str(e)}")
        logger.exception("Full error details:")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Cleanup temporary file
        if temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")

@app.get("/stats")
async def get_stats():
    """Get processing statistics."""
    if sam_processor is None:
        return {"error": "SAM processor not initialized"}
    
    return {
        'processing_stats': sam_processor.get_stats(),
        'service_info': {
            'device': DEVICE,
            'model_path': MODEL_PATH,
            'project_root': str(PROJECT_ROOT)
        }
    }

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting SAM service for local development")
    
    uvicorn.run(
        "app_local:app",  # Note: using app_local instead of app
        host="0.0.0.0",
        port=8001,  # Using port 8001 for SAM service
        reload=False,
        log_level="info"
    )