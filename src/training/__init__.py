"""
Training Module for Steel Defect Detection

This module contains the YOLOv8 training pipeline for steel surface defect detection.
Includes model training, validation, hyperparameter optimization, and experiment tracking.

Features:
- Automated GPU/CPU detection and optimization
- Configurable training parameters  
- Early stopping and model checkpointing
- Comprehensive logging and metrics
- Multi-format model export
"""

from .train import YOLOv8Trainer

__all__ = [
    "YOLOv8Trainer"
]

# Model configurations
MODEL_CONFIGS = {
    "yolov8n": {
        "model": "yolov8n.pt",
        "params": "3.2M", 
        "speed": "fastest",
        "batch_size": 16
    },
    "yolov8s": {
        "model": "yolov8s.pt", 
        "params": "11.2M",
        "speed": "fast",
        "batch_size": 16
    },
    "yolov8m": {
        "model": "yolov8m.pt",
        "params": "25.9M", 
        "speed": "medium",
        "batch_size": 8
    },
    "yolov8l": {
        "model": "yolov8l.pt",
        "params": "43.7M",
        "speed": "slow", 
        "batch_size": 4
    },
    "yolov8x": {
        "model": "yolov8x.pt",
        "params": "68.2M",
        "speed": "slowest",
        "batch_size": 2
    }
}

# Default training configuration
DEFAULT_TRAIN_CONFIG = {
    "epochs": 50,
    "patience": 10,
    "save_best": True,
    "save_plots": True,
    "verbose": True,
    "augment": True,
    "optimizer": "AdamW",
    "lr0": 0.01,
    "weight_decay": 0.0005
}