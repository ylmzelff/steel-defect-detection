"""
XML to YOLO Format Converter for Steel Defect Detection

This module converts PASCAL VOC XML annotations to YOLO format for steel surface defect detection.
Supports the NEU Steel Surface Defect Database with 6 defect classes.

Classes:
- crazing: Surface cracking
- inclusion: Foreign material inclusion
- patches: Surface patches  
- pitted_surface: Pitted surface defects
- rolled-in_scale: Rolled-in scale defects
- scratches: Surface scratches

Author: Steel Defect Detection Team
Date: 2024
"""

import xml.etree.ElementTree as ET
import os
import glob
import argparse
from pathlib import Path
from typing import Tuple, Optional, List

# Steel defect classes based on NEU-DET dataset
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


class XMLToYOLOConverter:
    """Converts PASCAL VOC XML annotations to YOLO format."""
    
    def __init__(self, classes: List[str] = CLASSES):
        """
        Initialize converter with class definitions.
        
        Args:
            classes: List of class names for steel defect detection
        """
        self.classes = classes
        
    def convert_bbox_to_yolo(self, image_width: int, image_height: int, 
                           bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Convert PASCAL VOC bounding box format to YOLO format.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels  
            bbox: Bounding box as (xmin, ymin, xmax, ymax)
            
        Returns:
            YOLO format bbox as (x_center_norm, y_center_norm, w_norm, h_norm)
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Calculate box dimensions
        box_w = float(xmax) - float(xmin)
        box_h = float(ymax) - float(ymin)
        
        # Calculate center coordinates
        x_center = float(xmin) + (box_w / 2)
        y_center = float(ymin) + (box_h / 2)
        
        # Normalize coordinates (0-1 range)
        x_center_norm = x_center / float(image_width)
        y_center_norm = y_center / float(image_height)
        w_norm = box_w / float(image_width)
        h_norm = box_h / float(image_height)
        
        return (x_center_norm, y_center_norm, w_norm, h_norm)
    
    def parse_xml_annotation(self, xml_path: str) -> Optional[dict]:
        """
        Parse XML annotation file and extract image info and objects.
        
        Args:
            xml_path: Path to XML annotation file
            
        Returns:
            Dictionary containing image info and objects, or None if error
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing XML '{xml_path}': {e}")
            return None
            
        # Get image dimensions
        size = root.find('size')
        if size is None:
            print(f"Warning: No 'size' information in '{xml_path}'")
            return None
            
        try:
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
        except Exception as e:
            print(f"Warning: Invalid 'size' information in '{xml_path}': {e}")
            return None
            
        # Extract objects
        objects = []
        for obj in root.iter('object'):
            class_name = obj.find('name').text
            
            # Check if class is in our predefined classes
            if class_name not in self.classes:
                print(f"Warning: Unknown class '{class_name}' in '{xml_path}'. Skipping.")
                continue
                
            class_id = self.classes.index(class_name)
            
            # Get bounding box
            bndbox = obj.find('bndbox')
            if bndbox is None:
                print(f"Warning: No 'bndbox' in object from '{xml_path}'")
                continue
                
            try:
                bbox = (
                    float(bndbox.find('xmin').text),
                    float(bndbox.find('ymin').text),
                    float(bndbox.find('xmax').text),
                    float(bndbox.find('ymax').text)
                )
            except Exception as e:
                print(f"Warning: Invalid bndbox in '{xml_path}': {e}")
                continue
                
            objects.append({
                'class_id': class_id,
                'class_name': class_name,
                'bbox': bbox
            })
            
        return {
            'width': img_width,
            'height': img_height,
            'objects': objects
        }
    
    def convert_file(self, xml_path: str, output_dir: str) -> bool:
        """
        Convert single XML file to YOLO format.
        
        Args:
            xml_path: Path to XML annotation file
            output_dir: Output directory for YOLO labels
            
        Returns:
            True if conversion successful, False otherwise
        """
        annotation_data = self.parse_xml_annotation(xml_path)
        if annotation_data is None:
            return False
            
        # Create output directory if not exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        base_filename = Path(xml_path).stem
        txt_filepath = Path(output_dir) / f"{base_filename}.txt"
        
        # Write YOLO format labels
        try:
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                for obj in annotation_data['objects']:
                    yolo_bbox = self.convert_bbox_to_yolo(
                        annotation_data['width'],
                        annotation_data['height'],
                        obj['bbox']
                    )
                    
                    # Write in YOLO format: class_id x_center y_center width height
                    f.write(f"{obj['class_id']} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                           f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            return True
            
        except Exception as e:
            print(f"Error writing YOLO file '{txt_filepath}': {e}")
            return False
    
    def convert_dataset(self, xml_dir: str, output_dir: str) -> None:
        """
        Convert all XML annotations in directory to YOLO format.
        
        Args:
            xml_dir: Directory containing XML annotation files
            output_dir: Output directory for YOLO labels
        """
        # Find all XML files recursively
        xml_files = glob.glob(os.path.join(xml_dir, "**", '*.xml'), recursive=True)
        
        if not xml_files:
            print(f"Warning: No XML files found in '{xml_dir}'")
            return
            
        print(f"Processing {len(xml_files)} XML files from '{xml_dir}'")
        
        successful_conversions = 0
        for xml_file in xml_files:
            if self.convert_file(xml_file, output_dir):
                successful_conversions += 1
                
        print(f"Conversion completed: {successful_conversions}/{len(xml_files)} files successfully converted")
        print(f"YOLO labels saved to '{output_dir}'")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC XML annotations to YOLO format for steel defect detection'
    )
    parser.add_argument(
        '--root', 
        type=str, 
        default='data/NEU-DET', 
        help='Root dataset folder containing train/ and validation/ (default: data/NEU-DET)'
    )
    parser.add_argument(
        '--out', 
        type=str, 
        default='data/labels', 
        help='Output labels base folder (default: data/labels)'
    )
    parser.add_argument(
        '--subsets', 
        type=str, 
        default='train,validation', 
        help='Comma-separated subsets to process (default: train,validation)'
    )
    
    args = parser.parse_args()
    
    converter = XMLToYOLOConverter()
    subsets = [s.strip() for s in args.subsets.split(',') if s.strip()]
    
    any_processed = False
    for subset in subsets:
        xml_annotations_dir = os.path.join(args.root, subset, 'annotations')
        output_dir = os.path.join(args.out, subset)
        
        if not os.path.isdir(xml_annotations_dir):
            print(f"Info: Directory '{xml_annotations_dir}' not found. Skipping.")
            continue
            
        print(f"Processing: {xml_annotations_dir} -> {output_dir}")
        converter.convert_dataset(xml_annotations_dir, output_dir)
        any_processed = True
        
    if not any_processed:
        print("No directories processed. Please check --root and --subsets parameters.")


if __name__ == "__main__":
    main()