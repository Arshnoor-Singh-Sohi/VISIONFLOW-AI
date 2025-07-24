"""
VisionFlow AI - Training Service
================================

This service handles all aspects of model training and management:
- Training data preparation and validation
- Model training orchestration
- Training progress monitoring
- Model evaluation and validation
- Training metrics collection and analysis
- Model versioning and deployment

Think of this as the "AI teacher" that continuously improves the system
by learning from accumulated data and human feedback.
"""

import os
import asyncio
import logging
import pickle
import joblib
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from ..config import get_settings
from ..database import db_manager
from ..models.database_models import (
    TrainingRun, TrainingSample, TrainingStatus, Classification, ImageSegment
)


# =============================================================================
# LOGGER SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# TRAINING SERVICE CLASS
# =============================================================================

class TrainingService:
    """
    Comprehensive model training and management service.
    
    This service orchestrates the entire machine learning pipeline:
    - Data preparation and feature engineering
    - Model training with various algorithms
    - Hyperparameter optimization
    - Model evaluation and validation
    - Training progress monitoring
    - Model deployment and versioning
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Training configuration
        self.supported_models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': None,  # Can add GradientBoostingClassifier
            'neural_network': None      # Can add MLPClassifier
        }
        
        # Feature extractors
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.label_encoder = LabelEncoder()
        
        # Training statistics
        self.models_trained = 0
        self.total_training_time = 0.0
        self.best_accuracy = 0.0
        
        logger.info("Training service initialized")
    
    async def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the training dataset.
        
        This analyzes the available training data to determine readiness
        for training and provide insights into data distribution.
        """
        try:
            with db_manager.get_session_context() as db:
                # Get all training samples
                samples = db.query(TrainingSample).all()
                
                if not samples:
                    return {
                        'total_samples': 0,
                        'samples_by_source': {},
                        'samples_by_label': {},
                        'human_verified_samples': 0,
                        'ready_for_training': False,
                        'min_samples_needed': self.settings.min_training_samples,
                        'issues': ['No training samples available']
                    }
                
                # Analyze sample distribution
                samples_by_source = {}
                samples_by_label = {}
                human_verified_count = 0
                
                for sample in samples:
                    # Count by source
                    source = sample.label_source
                    samples_by_source[source] = samples_by_source.get(source, 0) + 1
                    
                    # Count by label
                    label = sample.ground_truth_label
                    samples_by_label[label] = samples_by_label.get(label, 0) + 1
                    
                    # Count human verified
                    if sample.label_source == 'human':
                        human_verified_count += 1
                
                # Check training readiness
                total_samples = len(samples)
                unique_labels = len(samples_by_label)
                min_samples_per_class = min(samples_by_label.values()) if samples_by_label else 0
                
                issues = []
                ready_for_training = True
                
                if total_samples < self.settings.min_training_samples:
                    ready_for_training = False
                    issues.append(f"Insufficient samples: {total_samples} < {self.settings.min_training_samples}")
                
                if unique_labels < 2:
                    ready_for_training = False
                    issues.append(f"Need at least 2 classes, found {unique_labels}")
                
                if min_samples_per_class < 5:
                    ready_for_training = False
                    issues.append(f"Some classes have fewer than 5 samples")
                
                return {
                    'total_samples': total_samples,
                    'unique_labels': unique_labels,
                    'samples_by_source': samples_by_source,
                    'samples_by_label': samples_by_label,
                    'human_verified_samples': human_verified_count,
                    'min_samples_per_class': min_samples_per_class,
                    'ready_for_training': ready_for_training,
                    'min_samples_needed': self.settings.min_training_samples,
                    'issues': issues if issues else ['Dataset looks good for training']
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze dataset: {e}")
            return {
                'total_samples': 0,
                'error': str(e),
                'ready_for_training': False
            }
    
    async def should_trigger_training(self) -> bool:
        """
        Determine if automatic training should be triggered.
        
        This uses various heuristics to decide when it's beneficial
        to retrain the model based on new data accumulation.
        """
        try:
            # Check if training is enabled
            if not self.settings.enable_training:
                return False
            
            # Check if there's already training in progress
            with db_manager.get_session_context() as db:
                active_training = db.query(TrainingRun).filter(
                    TrainingRun.status.in_([
                        TrainingStatus.PENDING,
                        TrainingStatus.IN_PROGRESS,
                        TrainingStatus.PAUSED
                    ])
                ).first()
                
                if active_training:
                    logger.debug("Training already in progress, skipping auto-trigger")
                    return False
                
                # Get dataset info
                dataset_info = await self.get_dataset_info()
                
                if not dataset_info['ready_for_training']:
                    logger.debug("Dataset not ready for training")
                    return False
                
                # Check if enough new data has accumulated since last training
                last_training = db.query(TrainingRun).filter(
                    TrainingRun.status == TrainingStatus.COMPLETED
                ).order_by(TrainingRun.training_completed_at.desc()).first()
                
                if last_training:
                    # Check for new samples since last training
                    new_samples_count = db.query(TrainingSample).filter(
                        TrainingSample.created_at > last_training.training_completed_at,
                        TrainingSample.used_in_training == False
                    ).count()
                    
                    # Trigger if we have 20% more data than last training
                    trigger_threshold = max(50, int(last_training.num_samples * 0.2))
                    
                    if new_samples_count >= trigger_threshold:
                        logger.info(f"Auto-triggering training: {new_samples_count} new samples")
                        return True
                else:
                    # No previous training, trigger if we have enough data
                    if dataset_info['total_samples'] >= self.settings.min_training_samples:
                        logger.info("Auto-triggering first training run")
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to check training trigger: {e}")
            return False
    
    async def train_model(
        self,
        training_run_id: str,
        config: Dict[str, Any],
        resume: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train a machine learning model with the specified configuration.
        
        This is the main training orchestration function that handles
        the entire training pipeline from data preparation to model saving.
        
        Args:
            training_run_id: ID of the training run record
            config: Training configuration dictionary
            resume: Whether to resume from a previous checkpoint
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary with training results and metrics
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Starting model training for run: {training_run_id}")
            
            # Update training run status
            with db_manager.get_session_context() as db:
                training_run = db.query(TrainingRun).filter(
                    TrainingRun.id == training_run_id
                ).first()
                
                if not training_run:
                    raise ValueError(f"Training run not found: {training_run_id}")
                
                training_run.status = TrainingStatus.IN_PROGRESS
                training_run.training_started_at = start_time
                db.commit()
            
            if progress_callback:
                await progress_callback("started", {"message": "Training started"})
            
            # Step 1: Prepare training data
            logger.info("Preparing training data...")
            if progress_callback:
                await progress_callback("preparing_data", {"step": "Preparing training data"})
            
            X, y, feature_info = await self._prepare_training_data(config)
            
            # Step 2: Split data
            logger.info("Splitting data into train/validation sets...")
            train_test_split_ratio = config.get('train_test_split', 0.8)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=(1 - train_test_split_ratio),
                random_state=42,
                stratify=y
            )
            
            # Step 3: Train model
            logger.info("Training model...")
            if progress_callback:
                await progress_callback("training", {"step": "Training model"})
            
            model, training_metrics = await self._train_model_impl(
                X_train, y_train, X_test, y_test, config, progress_callback
            )
            
            # Step 4: Evaluate model
            logger.info("Evaluating model...")
            if progress_callback:
                await progress_callback("evaluating", {"step": "Evaluating model"})
            
            evaluation_metrics = await self._evaluate_model(
                model, X_test, y_test, feature_info
            )
            
            # Step 5: Save model
            logger.info("Saving model...")
            if progress_callback:
                await progress_callback("saving", {"step": "Saving model"})
            
            model_path, model_size = await self._save_model(
                model, training_run_id, feature_info
            )
            
            # Step 6: Update training run with results
            end_time = datetime.now(timezone.utc)
            duration = int((end_time - start_time).total_seconds())
            
            with db_manager.get_session_context() as db:
                training_run = db.query(TrainingRun).filter(
                    TrainingRun.id == training_run_id
                ).first()
                
                training_run.status = TrainingStatus.COMPLETED
                training_run.training_completed_at = end_time
                training_run.training_duration_seconds = duration
                training_run.train_accuracy = evaluation_metrics['train_accuracy']
                training_run.validation_accuracy = evaluation_metrics['test_accuracy']
                training_run.train_loss = training_metrics.get('train_loss', 0.0)
                training_run.validation_loss = training_metrics.get('validation_loss', 0.0)
                training_run.model_path = model_path
                training_run.model_size_bytes = model_size
                training_run.metrics_history = training_metrics.get('history', [])
                
                # Mark training samples as used
                training_samples = db.query(TrainingSample).filter(
                    TrainingSample.used_in_training == False
                ).all()
                
                for sample in training_samples:
                    sample.used_in_training = True
                    sample.training_run_id = training_run.id
                
                db.commit()
            
            # Update statistics
            self.models_trained += 1
            self.total_training_time += duration
            if evaluation_metrics['test_accuracy'] > self.best_accuracy:
                self.best_accuracy = evaluation_metrics['test_accuracy']
            
            if progress_callback:
                await progress_callback("completed", {
                    "message": "Training completed successfully",
                    "metrics": evaluation_metrics,
                    "duration_seconds": duration
                })
            
            logger.info(f"Training completed successfully for run: {training_run_id}")
            
            return {
                'success': True,
                'training_run_id': training_run_id,
                'duration_seconds': duration,
                'model_path': model_path,
                'metrics': evaluation_metrics,
                'model_size_bytes': model_size
            }
            
        except Exception as e:
            logger.error(f"Training failed for run {training_run_id}: {e}")
            
            # Update training run with error
            try:
                with db_manager.get_session_context() as db:
                    training_run = db.query(TrainingRun).filter(
                        TrainingRun.id == training_run_id
                    ).first()
                    
                    if training_run:
                        training_run.status = TrainingStatus.FAILED
                        training_run.error_message = str(e)
                        training_run.training_completed_at = datetime.now(timezone.utc)
                        db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update training run error: {db_error}")
            
            if progress_callback:
                await progress_callback("failed", {"error": str(e)})
            
            raise RuntimeError(f"Training failed: {e}") from e
    
    async def _prepare_training_data(self, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare training data by extracting features and labels.
        
        This converts the raw training samples into numerical features
        that can be used for machine learning.
        """
        try:
            with db_manager.get_session_context() as db:
                # Get training samples with related data
                samples_query = db.query(TrainingSample).join(
                    Classification, TrainingSample.segment_id == Classification.segment_id
                ).join(
                    ImageSegment, TrainingSample.segment_id == ImageSegment.id
                )
                
                # Filter by configuration
                if config.get('use_human_labels_only', False):
                    samples_query = samples_query.filter(
                        TrainingSample.label_source == 'human'
                    )
                
                samples = samples_query.all()
                
                if not samples:
                    raise ValueError("No training samples available")
                
                # Extract features and labels
                features_list = []
                labels_list = []
                
                for sample in samples:
                    # Get associated classification and segment
                    classification = None
                    segment = None
                    
                    for cls in sample.image.classifications:
                        if cls.segment_id == sample.segment_id:
                            classification = cls
                            break
                    
                    for seg in sample.image.segments:
                        if seg.id == sample.segment_id:
                            segment = seg
                            break
                    
                    if not classification or not segment:
                        continue
                    
                    # Extract numerical features
                    feature_vector = self._extract_features(classification, segment)
                    features_list.append(feature_vector)
                    labels_list.append(sample.ground_truth_label)
                
                if not features_list:
                    raise ValueError("No valid features could be extracted")
                
                # Convert to numpy arrays
                X = np.array(features_list)
                y = np.array(labels_list)
                
                # Encode labels
                y_encoded = self.label_encoder.fit_transform(y)
                
                feature_info = {
                    'feature_count': X.shape[1],
                    'sample_count': X.shape[0],
                    'label_classes': self.label_encoder.classes_.tolist(),
                    'feature_names': self._get_feature_names()
                }
                
                logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
                
                return X, y_encoded, feature_info
                
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def _extract_features(self, classification: Classification, segment: ImageSegment) -> List[float]:
        """
        Extract numerical features from classification and segment data.
        
        This converts the raw data into numerical features that can be
        used for machine learning algorithms.
        """
        features = []
        
        # Segment geometric features
        features.extend([
            float(segment.area),
            float(segment.bbox_width),
            float(segment.bbox_height),
            float(segment.bbox_width / segment.bbox_height) if segment.bbox_height > 0 else 1.0,  # aspect ratio
            float(segment.confidence_score)
        ])
        
        # Classification features
        features.extend([
            float(classification.confidence_score),
            float(classification.tokens_used) if classification.tokens_used else 0.0,
            1.0 if classification.human_verified else 0.0
        ])
        
        # Text features (simplified - in production you might use embeddings)
        label_length = len(classification.primary_label) if classification.primary_label else 0
        features.append(float(label_length))
        
        # Alternative labels count
        alt_count = len(classification.alternative_labels) if classification.alternative_labels else 0
        features.append(float(alt_count))
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Get descriptive names for the extracted features."""
        return [
            'segment_area',
            'bbox_width',
            'bbox_height',
            'aspect_ratio',
            'segment_confidence',
            'classification_confidence',
            'tokens_used',
            'human_verified',
            'label_length',
            'alternative_count'
        ]
    
    async def _train_model_impl(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the actual machine learning model.
        
        This handles the core training loop with progress monitoring
        and metrics collection.
        """
        try:
            model_type = config.get('model_type', 'random_forest')
            
            if model_type == 'random_forest':
                # Configure Random Forest
                model = RandomForestClassifier(
                    n_estimators=config.get('num_epochs', 100),
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Calculate training metrics
                train_accuracy = model.score(X_train, y_train)
                test_accuracy = model.score(X_test, y_test)
                
                training_metrics = {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_loss': 1.0 - train_accuracy,  # Simple loss approximation
                    'validation_loss': 1.0 - test_accuracy,
                    'history': [
                        {
                            'epoch': 1,
                            'train_accuracy': train_accuracy,
                            'validation_accuracy': test_accuracy,
                            'train_loss': 1.0 - train_accuracy,
                            'validation_loss': 1.0 - test_accuracy
                        }
                    ]
                }
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if progress_callback:
                await progress_callback("training_progress", {
                    "epoch": 1,
                    "total_epochs": 1,
                    "train_accuracy": training_metrics['train_accuracy'],
                    "validation_accuracy": training_metrics['test_accuracy']
                })
            
            return model, training_metrics
            
        except Exception as e:
            logger.error(f"Model training implementation failed: {e}")
            raise
    
    async def _evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model and calculate comprehensive metrics.
        
        This provides detailed evaluation metrics for model performance
        assessment and comparison.
        """
        try:
            # Predictions
            y_pred = model.predict(X_test)
            
            # Basic metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            train_accuracy = model.score(model.X_train_ if hasattr(model, 'X_train_') else X_test, 
                                       model.y_train_ if hasattr(model, 'y_train_') else y_test)
            
            # Classification report
            class_report = classification_report(
                y_test, y_pred,
                target_names=feature_info['label_classes'],
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importance_data = [
                    {
                        'feature': name,
                        'importance': float(importance)
                    }
                    for name, importance in zip(
                        feature_info['feature_names'],
                        model.feature_importances_
                    )
                ]
                feature_importance = sorted(
                    importance_data,
                    key=lambda x: x['importance'],
                    reverse=True
                )
            
            # Cross-validation score (on training data)
            cv_scores = cross_val_score(model, X_test, y_test, cv=min(5, len(X_test)//10))
            
            evaluation_metrics = {
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'cross_val_mean': float(cv_scores.mean()),
                'cross_val_std': float(cv_scores.std()),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'feature_importance': feature_importance,
                'test_samples': len(y_test),
                'num_classes': len(feature_info['label_classes'])
            }
            
            logger.info(f"Model evaluation completed - Test accuracy: {test_accuracy:.4f}")
            
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    async def _save_model(
        self,
        model: Any,
        training_run_id: str,
        feature_info: Dict[str, Any]
    ) -> Tuple[str, int]:
        """
        Save the trained model and associated metadata.
        
        This persists the model for future use and includes all
        necessary information for loading and using the model.
        """
        try:
            # Create model directory
            model_dir = Path(self.settings.models_path) / training_run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Save feature info and preprocessing objects
            metadata = {
                'feature_info': feature_info,
                'label_encoder': self.label_encoder,
                'model_type': type(model).__name__,
                'training_run_id': training_run_id,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'feature_names': feature_info['feature_names']
            }
            
            metadata_path = model_dir / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Calculate total size
            model_size = model_path.stat().st_size + metadata_path.stat().st_size
            
            logger.info(f"Model saved: {model_path} (size: {model_size} bytes)")
            
            return str(model_path), model_size
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    async def load_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a previously trained model.
        
        This loads both the model and its associated metadata
        for inference or further training.
        """
        try:
            model_file = Path(model_path)
            model_dir = model_file.parent
            metadata_path = model_dir / "metadata.pkl"
            
            # Load model
            model = joblib.load(model_file)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            logger.info(f"Model loaded: {model_path}")
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    async def predict(self, model_path: str, features: List[float]) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        This loads the model and makes predictions on new data,
        returning both the prediction and confidence scores.
        """
        try:
            model, metadata = await self.load_model(model_path)
            
            # Prepare features
            X = np.array([features])
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Get prediction probabilities (if available)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                confidence = float(probabilities.max())
                
                # Map probabilities to class names
                class_probabilities = {
                    class_name: float(prob)
                    for class_name, prob in zip(
                        metadata['feature_info']['label_classes'],
                        probabilities
                    )
                }
            else:
                confidence = 1.0  # Default confidence for non-probabilistic models
                class_probabilities = {}
            
            # Decode prediction
            predicted_label = metadata['label_encoder'].inverse_transform([prediction])[0]
            
            return {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'model_type': metadata['model_type'],
                'prediction_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training service statistics and performance metrics.
        
        This provides insights into training activity and performance
        for monitoring and optimization.
        """
        avg_training_time = (
            self.total_training_time / max(self.models_trained, 1)
        )
        
        return {
            'models_trained': self.models_trained,
            'total_training_time_seconds': self.total_training_time,
            'average_training_time_seconds': avg_training_time,
            'best_accuracy_achieved': self.best_accuracy,
            'supported_model_types': list(self.supported_models.keys()),
            'feature_count': len(self._get_feature_names())
        }


# =============================================================================
# SERVICE FACTORY FUNCTION
# =============================================================================

_training_service_instance = None

def get_training_service() -> TrainingService:
    """
    Get singleton instance of training service.
    
    Using a singleton ensures consistent training state
    and helps with resource management.
    """
    global _training_service_instance
    
    if _training_service_instance is None:
        _training_service_instance = TrainingService()
    
    return _training_service_instance