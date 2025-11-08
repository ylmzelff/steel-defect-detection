"""
API Module for Steel Defect Detection

This module provides REST API endpoints for steel defect detection using trained YOLOv8 models.
Includes FastAPI implementation with automatic documentation, batch processing, and monitoring.

Features:
- RESTful API with OpenAPI/Swagger documentation
- Image upload and batch processing
- Real-time defect detection and classification
- Configurable confidence and IoU thresholds
- Health monitoring and metrics
- CORS support for web integration
"""

from .main import app

__all__ = ["app"]

# API configuration
API_CONFIG = {
    "title": "Steel Defect Detection API",
    "version": "1.0.0",
    "description": "YOLOv8-based API for detecting surface defects in steel materials",
    "contact": {
        "name": "Steel Defect Detection Team",
        "url": "https://github.com/ylmzelff/steel-defect-detection-mlops",
    },
    "license": {
        "name": "MIT License",
        "url": "https://github.com/ylmzelff/steel-defect-detection-mlops/blob/main/LICENSE",
    }
}

# Supported image formats
SUPPORTED_FORMATS = [
    "image/jpeg",
    "image/jpg", 
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp"
]

# Default thresholds
DEFAULT_THRESHOLDS = {
    "confidence": 0.25,
    "iou": 0.45,
    "max_detections": 1000
}

# Rate limiting configuration
RATE_LIMITS = {
    "detect": "10/minute",
    "batch": "2/minute",
    "health": "60/minute"
}