"""
YOLOv8 Training Module for Steel Defect Detection

This module provides a complete training pipeline for YOLOv8 models on steel surface
defect detection using the NEU dataset. Includes GPU optimization, hyperparameter tuning,
and comprehensive logging for MLOps workflows.

Features:
- Automatic GPU detection and configuration
- Configurable training parameters
- Early stopping to prevent overfitting
- Comprehensive logging and metrics tracking
- Model checkpointing and validation

Author: Steel Defect Detection Team
Date: 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLOv8Trainer:
    """YOLOv8 training pipeline for steel defect detection."""
    
    def __init__(self, 
                 model_name: str = 'yolov8n.pt',
                 data_config: str = 'configs/neu_defect.yaml',
                 project_dir: str = 'runs/detect',
                 run_name: str = 'steel_defect_experiment'):
        """
        Initialize YOLOv8 trainer.
        
        Args:
            model_name: YOLOv8 model variant (yolov8n.pt, yolov8s.pt, etc.)
            data_config: Path to dataset configuration YAML file
            project_dir: Directory to save training results
            run_name: Name for this training run
        """
        self.model_name = model_name
        self.data_config = data_config
        self.project_dir = project_dir
        self.run_name = run_name
        self.model = None
        
        # Default training parameters optimized for NEU dataset
        self.train_params = {
            'epochs': 50,
            'batch': 16,
            'imgsz': 640,
            'patience': 10,
            'save': True,
            'plots': True,
            'verbose': True,
            'workers': 4,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': True,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'augment': True,
            'agnostic_nms': False,
            'retina_masks': False,
            'embed': None,
        }
        
    def check_system_requirements(self) -> Dict[str, Any]:
        """
        Check system requirements and GPU availability.
        
        Returns:
            Dictionary containing system information
        """
        system_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device_name': None,
            'total_memory': None,
            'pytorch_version': torch.__version__,
            'recommended_device': 'cpu'
        }
        
        if system_info['cuda_available']:
            system_info['device_name'] = torch.cuda.get_device_name(0)
            system_info['total_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            system_info['recommended_device'] = 0
            
            logger.info(f"✅ CUDA GPU detected: {system_info['device_name']}")
            logger.info(f"   GPU Memory: {system_info['total_memory']:.1f} GB")
            
            # Adjust batch size based on GPU memory
            if system_info['total_memory'] < 6:  # Less than 6GB
                self.train_params['batch'] = 8
                logger.info("   Adjusted batch size to 8 for GPU with <6GB memory")
            elif system_info['total_memory'] < 12:  # Less than 12GB
                self.train_params['batch'] = 16
                logger.info("   Using batch size 16 for medium GPU memory")
            else:  # 12GB or more
                self.train_params['batch'] = 32
                logger.info("   Using batch size 32 for high-end GPU")
        else:
            logger.warning("⚠️ No CUDA GPU detected. Training will use CPU (slow)")
            system_info['recommended_device'] = 'cpu'
            self.train_params['batch'] = 4  # Small batch for CPU
            
        return system_info
    
    def validate_data_config(self) -> bool:
        """
        Validate dataset configuration file.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if not os.path.exists(self.data_config):
            logger.error(f"❌ Dataset config file not found: {self.data_config}")
            return False
            
        try:
            with open(self.data_config, 'r') as f:
                config = yaml.safe_load(f)
                
            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in config:
                    logger.error(f"❌ Missing required field '{field}' in {self.data_config}")
                    return False
                    
            # Check if paths exist
            for split in ['train', 'val']:
                if split in config:
                    path = config[split]
                    if not os.path.exists(path):
                        logger.error(f"❌ Dataset path not found: {path}")
                        return False
                        
            logger.info(f"✅ Dataset config validated: {self.data_config}")
            logger.info(f"   Classes: {config['nc']} ({config['names']})")
            logger.info(f"   Train path: {config.get('train', 'N/A')}")
            logger.info(f"   Val path: {config.get('val', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reading dataset config: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load YOLOv8 model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading YOLOv8 model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            # Log model info
            if hasattr(self.model.model, 'yaml'):
                logger.info(f"✅ Model loaded successfully")
                logger.info(f"   Architecture: {self.model_name}")
                # logger.info(f"   Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def update_training_params(self, **kwargs) -> None:
        """
        Update training parameters.
        
        Args:
            **kwargs: Training parameters to update
        """
        for key, value in kwargs.items():
            if key in self.train_params:
                old_value = self.train_params[key]
                self.train_params[key] = value
                logger.info(f"Updated {key}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")
    
    def train(self, device: Optional[str] = None) -> Optional[Dict]:
        """
        Execute training pipeline.
        
        Args:
            device: Training device ('0', 'cpu', etc.). If None, auto-detected.
            
        Returns:
            Training results dictionary or None if failed
        """
        logger.info("=== Starting YOLOv8 Training Pipeline ===")
        
        # Check system requirements
        system_info = self.check_system_requirements()
        
        # Set device
        if device is None:
            device = system_info['recommended_device']
        self.train_params['device'] = device
        
        # Validate configuration
        if not self.validate_data_config():
            return None
            
        # Load model
        if not self.load_model():
            return None
            
        # Create output directory
        output_dir = Path(self.project_dir) / self.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log training parameters
        logger.info("=== Training Configuration ===")
        for key, value in self.train_params.items():
            logger.info(f"   {key}: {value}")
            
        try:
            logger.info("🚀 Starting training...")
            
            # Start training
            results = self.model.train(
                data=self.data_config,
                name=self.run_name,
                project=self.project_dir,
                **self.train_params
            )
            
            # Log completion
            best_model_path = output_dir / 'weights' / 'best.pt'
            last_model_path = output_dir / 'weights' / 'last.pt'
            
            logger.info("=== Training Completed Successfully! ===")
            logger.info(f"✅ Results saved to: {output_dir}")
            logger.info(f"✅ Best model: {best_model_path}")
            logger.info(f"✅ Last model: {last_model_path}")
            
            # Log final metrics if available
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                if 'metrics/mAP50(B)' in metrics:
                    logger.info(f"✅ Final mAP50: {metrics['metrics/mAP50(B)']:.4f}")
                if 'metrics/mAP50-95(B)' in metrics:
                    logger.info(f"✅ Final mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error("💡 Suggestions:")
            logger.error("   - Try reducing batch size (--batch 8 or --batch 4)")
            logger.error("   - Check dataset paths in config file")
            logger.error("   - Ensure sufficient disk space")
            return None
    
    def evaluate(self, model_path: str, data_config: Optional[str] = None) -> Optional[Dict]:
        """
        Evaluate trained model.
        
        Args:
            model_path: Path to trained model weights
            data_config: Dataset config file (uses self.data_config if None)
            
        Returns:
            Evaluation results or None if failed
        """
        if data_config is None:
            data_config = self.data_config
            
        try:
            logger.info(f"Evaluating model: {model_path}")
            model = YOLO(model_path)
            results = model.val(data=data_config)
            
            logger.info("=== Evaluation Results ===")
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"   {key}: {value:.4f}")
                        
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return None


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for steel defect detection')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model variant')
    parser.add_argument('--data', type=str, default='configs/neu_defect.yaml',
                       help='Dataset configuration file')
    parser.add_argument('--name', type=str, default='steel_defect_experiment',
                       help='Experiment name')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size (auto-detected if not specified)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Training device (0, cpu, etc.)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Advanced parameters
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                       help='Warmup epochs')
    
    # Actions
    parser.add_argument('--evaluate', type=str, default=None,
                       help='Evaluate model at given path instead of training')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = YOLOv8Trainer(
        model_name=args.model,
        data_config=args.data,
        project_dir=args.project,
        run_name=args.name
    )
    
    # Update parameters
    update_params = {}
    if args.epochs != 50:
        update_params['epochs'] = args.epochs
    if args.batch is not None:
        update_params['batch'] = args.batch
    if args.imgsz != 640:
        update_params['imgsz'] = args.imgsz
    if args.patience != 10:
        update_params['patience'] = args.patience
    if args.lr0 != 0.01:
        update_params['lr0'] = args.lr0
    if args.weight_decay != 0.0005:
        update_params['weight_decay'] = args.weight_decay
    if args.warmup_epochs != 3:
        update_params['warmup_epochs'] = args.warmup_epochs
    
    if update_params:
        trainer.update_training_params(**update_params)
    
    # Execute action
    if args.evaluate:
        results = trainer.evaluate(args.evaluate)
    else:
        results = trainer.train(device=args.device)
    
    if results is None:
        sys.exit(1)
    
    logger.info("Pipeline completed successfully!")


if __name__ == '__main__':
    main()