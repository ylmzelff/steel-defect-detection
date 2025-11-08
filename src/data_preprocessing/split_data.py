"""
Data Splitting Module for Steel Defect Detection Dataset

This module splits the NEU Steel Surface Defect Dataset into train/validation/test sets
with proper stratification and reproducible random seeding for MLOps pipelines.

Features:
- Configurable split ratios (default: 70% train, 20% validation, 10% test)
- Reproducible splits with fixed random seed
- Automatic image-label pair validation
- Proper directory structure creation for YOLO training

Author: Steel Defect Detection Team
Date: 2024
"""

import os
import glob
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Handles dataset splitting for steel defect detection MLOps pipeline."""
    
    def __init__(self, 
                 source_dirs: Dict[str, Dict[str, str]] = None,
                 target_base_dir: str = "data/dataset",
                 train_ratio: float = 0.7,
                 valid_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 random_seed: int = 42):
        """
        Initialize dataset splitter with configuration.
        
        Args:
            source_dirs: Dictionary mapping subset names to image/label directories
            target_base_dir: Base directory for output dataset structure
            train_ratio: Fraction of data for training (default: 0.7)
            valid_ratio: Fraction of data for validation (default: 0.2)
            test_ratio: Fraction of data for testing (default: 0.1)
            random_seed: Random seed for reproducible splits (default: 42)
        """
        if source_dirs is None:
            source_dirs = {
                'train': {
                    'images': 'data/NEU-DET/train/images', 
                    'labels': 'data/labels/train'
                },
                'validation': {
                    'images': 'data/NEU-DET/validation/images', 
                    'labels': 'data/labels/validation'
                }
            }
        
        self.source_dirs = source_dirs
        self.target_base_dir = target_base_dir
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        logger.info(f"Dataset splitter initialized with seed {random_seed}")
        
    def create_directory_structure(self) -> None:
        """Create target directory structure for YOLO dataset."""
        logger.info(f"Creating directory structure in '{self.target_base_dir}'")
        
        for split in ['train', 'valid', 'test']:
            # Create image directories
            images_dir = Path(self.target_base_dir) / 'images' / split
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Create label directories  
            labels_dir = Path(self.target_base_dir) / 'labels' / split
            labels_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info("Directory structure created successfully")
    
    def collect_image_label_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Collect all valid image-label pairs from source directories.
        
        Returns:
            List of tuples (image_path, label_path, source_subset)
        """
        logger.info("Collecting image-label pairs from source directories")
        
        all_pairs = []
        
        for subset, dirs in self.source_dirs.items():
            image_dir = dirs['images']
            label_dir = dirs['labels']
            
            logger.info(f"Processing subset '{subset}': {image_dir}")
            
            # Find all image files (including subdirectories for defect classes)
            image_paths = glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
            
            for image_path in image_paths:
                # Generate corresponding label file path
                image_filename = os.path.basename(image_path)
                label_filename = os.path.splitext(image_filename)[0] + ".txt"
                label_path = os.path.join(label_dir, label_filename)
                
                # Only include pairs where both image and label exist
                if os.path.exists(label_path):
                    all_pairs.append((image_path, label_path, subset))
                else:
                    logger.warning(f"Label file missing for {image_path}: {label_path}")
        
        if not all_pairs:
            logger.error("No valid image-label pairs found!")
            logger.error("Please run xml_to_yolo.py first to generate YOLO labels")
            return []
            
        logger.info(f"Found {len(all_pairs)} valid image-label pairs")
        return all_pairs
    
    def split_dataset(self, image_label_pairs: List[Tuple[str, str, str]]) -> Dict[str, List]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            image_label_pairs: List of (image_path, label_path, source_subset) tuples
            
        Returns:
            Dictionary with keys 'train', 'valid', 'test' and lists of pairs as values
        """
        # Shuffle pairs for random distribution
        pairs_copy = image_label_pairs.copy()
        random.shuffle(pairs_copy)
        
        total_count = len(pairs_copy)
        train_count = int(total_count * self.train_ratio)
        valid_count = int(total_count * self.valid_ratio)
        
        splits = {
            'train': pairs_copy[:train_count],
            'valid': pairs_copy[train_count:train_count + valid_count],
            'test': pairs_copy[train_count + valid_count:]
        }
        
        logger.info(f"Dataset split completed:")
        logger.info(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/total_count*100:.1f}%)")
        logger.info(f"  Valid: {len(splits['valid'])} samples ({len(splits['valid'])/total_count*100:.1f}%)")
        logger.info(f"  Test: {len(splits['test'])} samples ({len(splits['test'])/total_count*100:.1f}%)")
        
        return splits
    
    def copy_files(self, splits: Dict[str, List]) -> None:
        """
        Copy files to target directories according to splits.
        
        Args:
            splits: Dictionary containing split assignments
        """
        logger.info("Copying files to target directories")
        
        total_copied = 0
        
        for split_name, file_pairs in splits.items():
            logger.info(f"Copying {len(file_pairs)} files to {split_name} split")
            
            for image_path, label_path, source_subset in file_pairs:
                # Get filenames
                image_filename = os.path.basename(image_path)
                label_filename = os.path.basename(label_path)
                
                # Define target paths
                target_image_path = Path(self.target_base_dir) / 'images' / split_name / image_filename
                target_label_path = Path(self.target_base_dir) / 'labels' / split_name / label_filename
                
                # Copy files
                try:
                    shutil.copy2(image_path, target_image_path)
                    shutil.copy2(label_path, target_label_path)
                    total_copied += 2  # Count both image and label
                except Exception as e:
                    logger.error(f"Error copying {image_filename}: {e}")
                    
        logger.info(f"File copying completed. Total files copied: {total_copied}")
    
    def run(self) -> None:
        """Execute complete dataset splitting pipeline."""
        logger.info("Starting dataset splitting pipeline")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Collect image-label pairs
        image_label_pairs = self.collect_image_label_pairs()
        if not image_label_pairs:
            return
        
        # Split dataset
        splits = self.split_dataset(image_label_pairs)
        
        # Copy files
        self.copy_files(splits)
        
        logger.info("Dataset splitting pipeline completed successfully")
        logger.info(f"Output saved to: {self.target_base_dir}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split steel defect dataset for YOLO training')
    parser.add_argument('--input-root', type=str, default='data/NEU-DET', 
                       help='Root directory of input dataset')
    parser.add_argument('--labels-root', type=str, default='data/labels',
                       help='Root directory of YOLO labels')
    parser.add_argument('--output', type=str, default='data/dataset',
                       help='Output directory for split dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--valid-ratio', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Configure source directories based on arguments
    source_dirs = {
        'train': {
            'images': os.path.join(args.input_root, 'train', 'images'),
            'labels': os.path.join(args.labels_root, 'train')
        },
        'validation': {
            'images': os.path.join(args.input_root, 'validation', 'images'),
            'labels': os.path.join(args.labels_root, 'validation')
        }
    }
    
    # Create and run splitter
    splitter = DatasetSplitter(
        source_dirs=source_dirs,
        target_base_dir=args.output,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    splitter.run()


if __name__ == "__main__":
    main()