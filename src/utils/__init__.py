"""
Utility Functions for Steel Defect Detection MLOps Pipeline

This module contains common utility functions used throughout the project:
- File and path management
- Configuration loading and validation
- Logging setup
- Performance metrics calculation
- Visualization utilities
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML configuration: {e}")

def create_directory_structure(base_path: str, directories: List[str]) -> None:
    """
    Create directory structure.
    
    Args:
        base_path: Base directory path
        directories: List of directories to create
    """
    base_path = Path(base_path)
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)

def validate_dataset_paths(dataset_config: Dict[str, Any]) -> bool:
    """
    Validate dataset paths in configuration.
    
    Args:
        dataset_config: Dataset configuration dictionary
        
    Returns:
        True if all paths exist, False otherwise
    """
    required_paths = ['train', 'val']
    
    for path_key in required_paths:
        if path_key in dataset_config:
            path = dataset_config[path_key]
            if not os.path.exists(path):
                logging.error(f"Dataset path not found: {path}")
                return False
    
    return True

def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    current_file = Path(__file__).resolve()
    # Go up from src/utils/ to project root
    return current_file.parent.parent.parent

def ensure_path_exists(path: str) -> Path:
    """
    Ensure path exists, create if necessary.
    
    Args:
        path: Path string
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

class MetricsCalculator:
    """Calculate and format performance metrics."""
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
        """
        Format metrics for display.
        
        Args:
            metrics: Dictionary of metric values
            precision: Number of decimal places
            
        Returns:
            Dictionary of formatted metric strings
        """
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted[key] = f"{value:.{precision}f}"
            else:
                formatted[key] = str(value)
        return formatted
    
    @staticmethod
    def calculate_improvement(old_value: float, new_value: float) -> Dict[str, Any]:
        """
        Calculate improvement percentage.
        
        Args:
            old_value: Previous metric value
            new_value: Current metric value
            
        Returns:
            Dictionary with improvement info
        """
        if old_value == 0:
            return {"improvement": float('inf'), "percentage": "N/A"}
        
        improvement = new_value - old_value
        percentage = (improvement / old_value) * 100
        
        return {
            "improvement": improvement,
            "percentage": f"{percentage:+.2f}%",
            "direction": "improved" if improvement > 0 else "decreased"
        }

__all__ = [
    "setup_logging",
    "load_config", 
    "create_directory_structure",
    "validate_dataset_paths",
    "get_project_root",
    "ensure_path_exists",
    "MetricsCalculator"
]