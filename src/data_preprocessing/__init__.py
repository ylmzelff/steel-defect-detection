"""
Data Preprocessing Module for Steel Defect Detection

This module contains utilities for preprocessing the NEU Steel Surface Defect Database:
- XML to YOLO format conversion
- Dataset splitting for train/validation/test
- Data validation and augmentation utilities

Classes supported:
- crazing: Surface cracking patterns
- inclusion: Foreign material inclusions  
- patches: Irregular surface patches
- pitted_surface: Small surface pits and holes
- rolled-in_scale: Scale defects from rolling process
- scratches: Linear surface scratches
"""

from .xml_to_yolo import XMLToYOLOConverter
from .split_data import DatasetSplitter

__all__ = [
    "XMLToYOLOConverter",
    "DatasetSplitter"
]

# Default configuration for NEU dataset
NEU_CLASSES = [
    "crazing",
    "inclusion", 
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Default paths
DEFAULT_PATHS = {
    "neu_dataset": "data/NEU-DET",
    "labels_output": "data/labels", 
    "dataset_output": "data/dataset"
}

# Split ratios
DEFAULT_SPLIT_RATIOS = {
    "train": 0.7,
    "validation": 0.2, 
    "test": 0.1
}