"""
FastAPI application for steel defect detection inference.
Provides REST API endpoints with MLOps monitoring capabilities.
"""

import os
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import uvicorn

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    print("Prometheus client not available. Install with: pip install prometheus-client")
    PROMETHEUS_AVAILABLE = False

# Import model and monitoring
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Ultralytics not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from monitoring.drift_detector import ModelDriftDetector
    DRIFT_DETECTION_AVAILABLE = True
except ImportError:
    print("Drift detection not available")
    DRIFT_DETECTION_AVAILABLE = False

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    try:
        # Clear the entire registry to avoid duplicates
        from prometheus_client import REGISTRY, CollectorRegistry
        REGISTRY._collector_to_names.clear()
        REGISTRY._names_to_collectors.clear()
    except Exception as e:
        logger.warning(f"Could not clear existing metrics: {e}")
    
    REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
    REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
    PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['model_version'])
    MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time to load model')
    ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
    DRIFT_DETECTED = Gauge('model_drift_detected', 'Whether model drift was detected', ['drift_type'])

# Initialize FastAPI app
app = FastAPI(
    title="Steel Defect Detection API",
    description="YOLOv8-based steel surface defect detection service with MLOps monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_info = {}
drift_detector = ModelDriftDetector() if DRIFT_DETECTION_AVAILABLE else None

# Pydantic models
class PredictionResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    confidence_threshold: float
    processing_time: float
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
    timestamp: str
    version: str

class DriftResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    details: Dict[str, Any]
    timestamp: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize model and services on startup."""
    global model, model_info
    
    try:
        start_time = time.time()
        
        # Try to load the best model
        model_paths = [
            "models/best.pt",
            "runs/detect/steel_defect_colab_50_epochs/weights/best.pt",
            "runs/detect/steel_defect_colab_100_epochs/weights/best.pt"
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                logger.info(f"Found model at: {path}")
                break
        
        if not model_path:
            if YOLO_AVAILABLE:
                logger.warning("No trained model found. Using YOLOv8n pretrained model.")
                model_path = "yolov8n.pt"
            else:
                raise ImportError("No model found and YOLO not available")
        
        if YOLO_AVAILABLE:
            model = YOLO(model_path)
            load_time = time.time() - start_time
            
            if PROMETHEUS_AVAILABLE:
                MODEL_LOAD_TIME.set(load_time)
            
            model_info = {
                "model_path": str(model_path),
                "model_type": "YOLOv8",
                "load_time": load_time,
                "loaded_at": datetime.now().isoformat()
            }
            
            logger.info(f"Model loaded successfully from {model_path} in {load_time:.2f} seconds")
        else:
            raise ImportError("YOLO not available")
        
        # Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        model_info = {"error": str(e), "loaded_at": datetime.now().isoformat()}

@app.middleware("http")
async def monitor_requests(request, call_next):
    """Monitor all requests with Prometheus metrics."""
    start_time = time.time()
    
    if PROMETHEUS_AVAILABLE:
        ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.observe(time.time() - start_time)
        
        return response
    
    finally:
        if PROMETHEUS_AVAILABLE:
            ACTIVE_CONNECTIONS.dec()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Steel Defect Detection API",
        "version": "2.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancing."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_info=model_info,
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    if PROMETHEUS_AVAILABLE:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    else:
        return {"message": "Prometheus metrics not available"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_defects(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.25,
    background_tasks: BackgroundTasks = None
):
    """
    Detect steel surface defects in uploaded image.
    
    Args:
        file: Image file to analyze
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
        
    Returns:
        PredictionResponse with detection results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Run inference
        results = model(image, conf=confidence_threshold)
        
        # Process results
        predictions = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name (assuming standard defect classes)
                    class_names = {
                        0: 'crazing', 1: 'inclusion', 2: 'patches', 
                        3: 'pitted_surface', 4: 'rolled-in_scale', 5: 'scratches'
                    }
                    class_name = class_names.get(class_id, f'class_{class_id}')
                    
                    predictions.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': {
                            'x1': float(x1), 'y1': float(y1),
                            'x2': float(x2), 'y2': float(y2)
                        }
                    })
        
        processing_time = time.time() - start_time
        
        # Update metrics
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(model_version="1.0").inc()
        
        # Schedule drift detection in background
        if background_tasks and drift_detector:
            background_tasks.add_task(check_drift_background, [image])
        
        response = PredictionResponse(
            success=True,
            predictions=predictions,
            confidence_threshold=confidence_threshold,
            processing_time=processing_time,
            model_version="1.0",
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction completed: {len(predictions)} detections in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/detect-drift", response_model=DriftResponse)
async def detect_drift(files: List[UploadFile] = File(...)):
    """
    Detect data drift using uploaded images.
    
    Args:
        files: List of image files to analyze for drift
        
    Returns:
        DriftResponse with drift detection results
    """
    if not drift_detector:
        raise HTTPException(status_code=503, detail="Drift detection not available")
    
    try:
        images = []
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
                
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                images.append(image)
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Detect drift
        drift_result = drift_detector.detect_data_drift(images)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            DRIFT_DETECTED.labels(drift_type='data').set(1 if drift_result['drift_detected'] else 0)
        
        return DriftResponse(
            drift_detected=drift_result['drift_detected'],
            drift_score=drift_result['drift_score'],
            details=drift_result,
            timestamp=drift_result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Drift detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

async def check_drift_background(images: List[np.ndarray]):
    """Background task for drift detection."""
    if not drift_detector:
        return
        
    try:
        drift_result = drift_detector.detect_data_drift(images)
        if drift_result['drift_detected']:
            logger.warning(f"Data drift detected: {drift_result}")
            if PROMETHEUS_AVAILABLE:
                DRIFT_DETECTED.labels(drift_type='data').set(1)
    except Exception as e:
        logger.error(f"Background drift detection failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information."""
    return {
        "model_info": model_info,
        "model_loaded": model is not None,
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "defect_classes": [
            "crazing", "inclusion", "patches", 
            "pitted_surface", "rolled-in_scale", "scratches"
        ],
        "monitoring": {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "drift_detection_available": DRIFT_DETECTION_AVAILABLE,
            "yolo_available": YOLO_AVAILABLE
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info"
    )