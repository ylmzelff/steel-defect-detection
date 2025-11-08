# Steel Defect Detection API - FastAPI Implementation
# This would be a complete FastAPI application for model serving

"""
FastAPI application for steel defect detection inference.
Provides REST API endpoints for image upload and defect detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path
import os
from typing import List, Dict, Any
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Steel Defect Detection API",
    description="YOLOv8-based API for detecting surface defects in steel materials",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# Defect class names
DEFECT_CLASSES = [
    "crazing",
    "inclusion", 
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "1000"))

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found: {MODEL_PATH}")
            # For demo purposes, use a pre-trained model
            model = YOLO("yolov8n.pt")
            logger.warning("Using default YOLOv8n model for demo")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Steel Defect Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.get("/model/info")
async def model_info():
    """Get model information."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": MODEL_PATH,
        "classes": DEFECT_CLASSES,
        "num_classes": len(DEFECT_CLASSES),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "max_detections": MAX_DETECTIONS
    }

def process_image(image_bytes: bytes) -> np.ndarray:
    """Process uploaded image bytes to numpy array."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def format_detections(results) -> List[Dict[str, Any]]:
    """Format YOLO detection results."""
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                
                detection = {
                    "bbox": {
                        "x1": box[0],
                        "y1": box[1], 
                        "x2": box[2],
                        "y2": box[3],
                        "width": box[2] - box[0],
                        "height": box[3] - box[1]
                    },
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": DEFECT_CLASSES[class_id] if class_id < len(DEFECT_CLASSES) else f"unknown_{class_id}",
                    "area": (box[2] - box[0]) * (box[3] - box[1])
                }
                detections.append(detection)
    
    return detections

@app.post("/detect")
async def detect_defects(file: UploadFile = File(...)):
    """
    Detect defects in uploaded image.
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
    
    Returns:
        Detection results with bounding boxes and classifications
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Process image
        image_array = process_image(image_bytes)
        
        # Run inference
        results = model(
            image_array,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS
        )
        
        # Format results
        detections = format_detections(results)
        
        # Calculate summary statistics
        summary = {
            "total_detections": len(detections),
            "defect_types": list(set(d["class_name"] for d in detections)),
            "avg_confidence": sum(d["confidence"] for d in detections) / len(detections) if detections else 0,
            "image_size": {
                "width": image_array.shape[1],
                "height": image_array.shape[0],
                "channels": image_array.shape[2] if len(image_array.shape) > 2 else 1
            }
        }
        
        return {
            "filename": file.filename,
            "summary": summary,
            "detections": detections,
            "model_info": {
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD
            }
        }
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Detect defects in multiple uploaded images.
    
    Args:
        files: List of uploaded image files
    
    Returns:
        Batch detection results
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            # Process single file
            result = await detect_defects(file)
            results.append({
                "filename": file.filename,
                "status": "success",
                "result": result
            })
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": e.detail
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error", 
                "error": str(e)
            })
    
    return {
        "batch_size": len(files),
        "processed": len(results),
        "results": results
    }

if __name__ == "__main__":
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )