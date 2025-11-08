"""
Steel Defect Detection MLOps Pipeline
Source code package for steel surface defect detection using YOLOv8.
"""

__version__ = "1.0.0"
__author__ = "Steel Defect Detection Team"
__email__ = "your.email@example.com"

# Package metadata
__title__ = "steel-defect-detection-mlops"
__description__ = "MLOps pipeline for steel surface defect detection using YOLOv8"
__url__ = "https://github.com/ylmzelff/steel-defect-detection-mlops"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Steel Defect Detection Team"

# Supported defect classes
DEFECT_CLASSES = [
    "crazing",          # Surface cracking patterns
    "inclusion",        # Foreign material inclusions
    "patches",          # Irregular surface patches  
    "pitted_surface",   # Small surface pits and holes
    "rolled-in_scale",  # Scale defects from rolling process
    "scratches"         # Linear surface scratches
]

# Model variants
YOLO_MODELS = {
    "nano": "yolov8n.pt",      # 3.2M params - fastest
    "small": "yolov8s.pt",     # 11.2M params - balanced
    "medium": "yolov8m.pt",    # 25.9M params - accurate
    "large": "yolov8l.pt",     # 43.7M params - very accurate 
    "xlarge": "yolov8x.pt"     # 68.2M params - most accurate
}

# Default configuration
DEFAULT_CONFIG = {
    "dataset_path": "data/dataset",
    "config_path": "configs/neu_defect.yaml", 
    "model_name": "yolov8n.pt",
    "epochs": 50,
    "batch_size": 16,
    "image_size": 640,
    "num_classes": 6
}