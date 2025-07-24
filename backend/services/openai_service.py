"""
VisionFlow AI - OpenAI Integration Service
==========================================

This service handles all interactions with OpenAI's API, specifically focused
on image analysis and classification. Think of this as your "AI expert consultant"
that can look at image segments and tell you what's in them.

Why OpenAI instead of Google Lens?
- More control over prompts and behavior
- Better integration with our pipeline
- Consistent API responses in structured format
- Ability to customize for specific domains (food, objects, etc.)
- No rate limiting issues like web scraping
- Better error handling and retry logic
"""

import base64
import logging
import asyncio
import time
from typing import Dict, List, Optional, Union, Any
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

import openai
from openai import AsyncOpenAI
from PIL import Image
import aiofiles
import aiohttp

from ..config import get_settings
from ..utils.image_processing import resize_image_if_needed, get_image_info


# =============================================================================
# LOGGER SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES FOR CLASSIFICATION RESULTS
# =============================================================================

@dataclass
class ClassificationResult:
    """
    Structured result from OpenAI image classification.
    
    This class ensures we always get consistent, typed results from our
    classification calls, making it easier to work with the data downstream.
    """
    primary_label: str
    confidence_score: float
    alternative_labels: List[Dict[str, float]]
    raw_response: Dict[str, Any]
    model_used: str
    tokens_used: int
    processing_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'primary_label': self.primary_label,
            'confidence_score': self.confidence_score,
            'alternative_labels': self.alternative_labels,
            'raw_response': self.raw_response,
            'model_used': self.model_used,
            'tokens_used': self.tokens_used,
            'processing_time_seconds': self.processing_time_seconds
        }


# =============================================================================
# OPENAI SERVICE CLASS
# =============================================================================

class OpenAIService:
    """
    Service class for OpenAI API interactions.
    
    This class encapsulates all OpenAI functionality, providing a clean
    interface for the rest of our application to use. It handles:
    - Image encoding and preparation
    - Prompt engineering for different use cases
    - Error handling and retries
    - Rate limiting and quota management
    - Response parsing and validation
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize async OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            timeout=60.0,  # 60 second timeout for image processing
            max_retries=3   # Automatic retries on transient failures
        )
        
        # Track usage for monitoring
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        logger.info("OpenAI service initialized successfully")
    
    async def classify_image_segment(
        self, 
        image_path: Union[str, Path],
        context: str = "food identification",
        custom_prompt: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a single image segment using OpenAI Vision API.
        
        This is the core function that takes an image segment and returns
        what OpenAI thinks it contains. Think of it as asking an expert:
        "What do you see in this picture?"
        
        Args:
            image_path: Path to the image file to classify
            context: Context for classification (e.g., "food", "objects", "clothing")
            custom_prompt: Optional custom prompt override
            
        Returns:
            ClassificationResult with structured classification data
        """
        start_time = time.time()
        
        try:
            # Prepare the image for OpenAI
            base64_image = await self._encode_image_to_base64(image_path)
            
            # Generate the prompt based on context
            prompt = custom_prompt or self._generate_classification_prompt(context)
            
            # Make the API call
            logger.debug(f"Classifying image: {image_path}")
            response = await self._call_openai_vision(base64_image, prompt)
            
            # Parse and structure the response
            result = await self._parse_classification_response(
                response, 
                time.time() - start_time
            )
            
            # Update usage tracking
            self.total_requests += 1
            self.total_tokens_used += result.tokens_used
            
            logger.info(f"Successfully classified image: {result.primary_label} "
                       f"(confidence: {result.confidence_score:.2f})")
            
            return result
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Failed to classify image {image_path}: {e}")
            raise OpenAIClassificationError(f"Classification failed: {e}") from e
    
    async def batch_classify_segments(
        self,
        image_paths: List[Union[str, Path]],
        context: str = "food identification",
        max_concurrent: int = 5
    ) -> List[ClassificationResult]:
        """
        Classify multiple image segments concurrently.
        
        This function processes multiple images at once, which is much faster
        than processing them one by one. It's like having multiple experts
        work on different images simultaneously.
        
        Args:
            image_paths: List of paths to images to classify
            context: Context for classification
            max_concurrent: Maximum number of concurrent API calls
            
        Returns:
            List of ClassificationResult objects
        """
        if not image_paths:
            return []
        
        logger.info(f"Starting batch classification of {len(image_paths)} images")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def classify_with_semaphore(path: Union[str, Path]) -> ClassificationResult:
            async with semaphore:
                return await self.classify_image_segment(path, context)
        
        # Process all images concurrently
        results = await asyncio.gather(
            *[classify_with_semaphore(path) for path in image_paths],
            return_exceptions=True
        )
        
        # Handle any exceptions that occurred
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to classify {image_paths[i]}: {result}")
                # You could create a "failed" result here instead of skipping
            else:
                successful_results.append(result)
        
        logger.info(f"Batch classification completed: {len(successful_results)}/{len(image_paths)} successful")
        return successful_results
    
    async def _encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 for OpenAI API.
        
        OpenAI's Vision API requires images to be base64 encoded. This function
        also optimizes the image size to reduce token usage and API costs.
        """
        try:
            # Convert to Path object for easier handling
            path = Path(image_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            
            # Read and potentially resize the image
            async with aiofiles.open(path, 'rb') as image_file:
                image_data = await image_file.read()
            
            # Resize if the image is too large (saves tokens and money)
            image_data = await self._optimize_image_for_api(image_data)
            
            # Encode to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            logger.debug(f"Image encoded: {len(base64_image)} base64 characters")
            return base64_image
            
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    async def _optimize_image_for_api(self, image_data: bytes) -> bytes:
        """
        Optimize image size and quality for OpenAI API.
        
        OpenAI charges based on image size, so we want to find the sweet spot
        between image quality (for accurate classification) and cost optimization.
        """
        try:
            # Open image with PIL
            image = Image.open(BytesIO(image_data))
            
            # Get current dimensions
            width, height = image.size
            max_dimension = 1024  # Good balance between quality and cost
            
            # Resize if too large
            if max(width, height) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            
            # Convert to RGB if necessary (removes alpha channel)
            if image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if 'A' in image.mode else None)
                image = rgb_image
            
            # Save optimized image to bytes
            output_buffer = BytesIO()
            image.save(output_buffer, format='JPEG', quality=85, optimize=True)
            optimized_data = output_buffer.getvalue()
            
            # Log optimization results
            original_size = len(image_data)
            optimized_size = len(optimized_data)
            savings_percent = ((original_size - optimized_size) / original_size) * 100
            
            logger.debug(f"Image optimized: {original_size} -> {optimized_size} bytes "
                        f"({savings_percent:.1f}% savings)")
            
            return optimized_data
            
        except Exception as e:
            logger.warning(f"Image optimization failed, using original: {e}")
            return image_data
    
    def _generate_classification_prompt(self, context: str) -> str:
        """
        Generate context-specific prompts for better classification.
        
        Different contexts (food, objects, clothing) need different prompts
        to get the best results from OpenAI. This function creates optimized
        prompts for each use case.
        """
        prompts = {
            "food identification": """
                Analyze this image and identify the food item(s) shown. Please provide your response in this exact JSON format:

                {
                    "primary_item": "most specific food name",
                    "confidence": 0.95,
                    "alternatives": [
                        {"item": "alternative name 1", "confidence": 0.80},
                        {"item": "alternative name 2", "confidence": 0.65}
                    ],
                    "category": "food category (fruit, vegetable, protein, etc.)",
                    "freshness": "assessment of freshness if visible",
                    "quantity": "estimated quantity or portion size"
                }

                Be as specific as possible (e.g., "Gala apple" instead of just "apple").
                Focus on identifying actual food items, not containers or packaging.
            """,
            
            "general objects": """
                Identify and describe the main object(s) in this image. Provide your response in this JSON format:

                {
                    "primary_object": "most specific object name",
                    "confidence": 0.95,
                    "alternatives": [
                        {"object": "alternative description 1", "confidence": 0.80},
                        {"object": "alternative description 2", "confidence": 0.65}
                    ],
                    "category": "general category",
                    "attributes": ["color", "material", "condition", "etc."]
                }
            """,
            
            "kitchen items": """
                Identify kitchen items, appliances, or food-related objects in this image. Format:

                {
                    "primary_item": "specific kitchen item name",
                    "confidence": 0.95,
                    "alternatives": [
                        {"item": "alternative name 1", "confidence": 0.80}
                    ],
                    "category": "appliance/utensil/food/container/etc.",
                    "condition": "new/used/clean/dirty/etc."
                }
            """
        }
        
        return prompts.get(context, prompts["general objects"])
    
    async def _call_openai_vision(self, base64_image: str, prompt: str) -> Any:
        """
        Make the actual API call to OpenAI Vision.
        
        This function handles the low-level API interaction, including
        error handling, retries, and response validation.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"  # Use high detail for better accuracy
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.settings.openai_max_tokens,
                temperature=self.settings.openai_temperature,
            )
            
            return response
            
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit: {e}")
            # Wait before retrying (exponential backoff)
            await asyncio.sleep(min(60, 2 ** self.failed_requests))
            raise
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise
    
    async def _parse_classification_response(
        self, 
        response: Any, 
        processing_time: float
    ) -> ClassificationResult:
        """
        Parse OpenAI response into structured ClassificationResult.
        
        This function extracts the useful information from OpenAI's response
        and structures it in a consistent format for our application.
        """
        try:
            # Extract basic response info
            message_content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            model_used = response.model
            
            # Try to parse JSON response
            import json
            try:
                parsed_json = json.loads(message_content)
                
                # Extract primary classification
                primary_label = parsed_json.get('primary_item') or parsed_json.get('primary_object', 'unknown')
                confidence_score = parsed_json.get('confidence', 0.0)
                
                # Extract alternatives
                alternatives = []
                alt_list = parsed_json.get('alternatives', [])
                for alt in alt_list:
                    if isinstance(alt, dict):
                        alt_name = alt.get('item') or alt.get('object', '')
                        alt_confidence = alt.get('confidence', 0.0)
                        if alt_name:
                            alternatives.append({alt_name: alt_confidence})
                
            except json.JSONDecodeError:
                # Fallback: treat the whole response as the primary label
                logger.warning("Could not parse JSON response, using raw text")
                primary_label = message_content.strip()
                confidence_score = 0.8  # Default confidence for non-JSON responses
                alternatives = []
            
            return ClassificationResult(
                primary_label=primary_label,
                confidence_score=confidence_score,
                alternative_labels=alternatives,
                raw_response=response.dict() if hasattr(response, 'dict') else str(response),
                model_used=model_used,
                tokens_used=tokens_used,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            raise OpenAIParsingError(f"Response parsing failed: {e}") from e
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for monitoring and cost tracking.
        
        This helps you understand how much you're using the OpenAI API
        and can help with budgeting and optimization.
        """
        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            'total_tokens_used': self.total_tokens_used,
            'average_tokens_per_request': self.total_tokens_used / max(self.total_requests, 1),
            'estimated_cost_usd': self._estimate_cost()
        }
    
    def _estimate_cost(self) -> float:
        """
        Estimate API costs based on token usage.
        
        OpenAI pricing changes over time, so this is just a rough estimate.
        Check current pricing at https://openai.com/pricing
        """
        # These are rough estimates - update with current pricing
        if 'gpt-4-vision' in self.settings.openai_model.lower():
            cost_per_1k_tokens = 0.01  # Approximate cost per 1K tokens
        else:
            cost_per_1k_tokens = 0.002  # Lower cost for non-vision models
            
        return (self.total_tokens_used / 1000) * cost_per_1k_tokens
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the OpenAI connection and API key.
        
        This is useful for system health checks and debugging configuration issues.
        """
        try:
            # Make a simple test request
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a cheaper model for testing
                messages=[{"role": "user", "content": "Hello, this is a connection test."}],
                max_tokens=10
            )
            
            return {
                "status": "success",
                "model_available": response.model,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class OpenAIServiceError(Exception):
    """Base exception for OpenAI service errors."""
    pass


class OpenAIClassificationError(OpenAIServiceError):
    """Raised when image classification fails."""
    pass


class OpenAIParsingError(OpenAIServiceError):
    """Raised when response parsing fails."""
    pass


# =============================================================================
# SERVICE FACTORY FUNCTION
# =============================================================================

_openai_service_instance = None

def get_openai_service() -> OpenAIService:
    """
    Get singleton instance of OpenAI service.
    
    Using a singleton ensures we don't create multiple instances
    and helps with resource management and usage tracking.
    """
    global _openai_service_instance
    
    if _openai_service_instance is None:
        _openai_service_instance = OpenAIService()
    
    return _openai_service_instance