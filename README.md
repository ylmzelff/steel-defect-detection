# Steel Defect Detection MLOps Pipeline

A complete MLOps pipeline for steel surface defect detection using YOLOv8 and the NEU Steel Surface Defect Database. This project demonstrates end-to-end machine learning operations from data preprocessing to model deployment.

## 🎯 Project Overview

This project implements a comprehensive computer vision solution for detecting surface defects in steel materials. Using the NEU Steel Surface Defect Database, we train YOLOv8 models to identify 6 types of common steel defects with high accuracy.

### Defect Classes

- **Crazing**: Surface cracking patterns
- **Inclusion**: Foreign material inclusions
- **Patches**: Irregular surface patches
- **Pitted Surface**: Small surface pits and holes
- **Rolled-in Scale**: Scale defects from rolling process
- **Scratches**: Linear surface scratches

## 🏆 Performance Results

Our YOLOv8n model achieved excellent performance on the NEU dataset:

- **mAP50**: 76.47% - Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: 43.28% - Mean Average Precision averaged across IoU thresholds 0.5-0.95
- **Training Time**: ~19 minutes on Tesla T4 GPU (50 epochs)
- **Model Size**: 6.2MB (YOLOv8n optimized for speed)

## 📁 Project Structure

```
steel-defect-detection-mlops/
│
├── src/                          # Source code modules
│   ├── data_preprocessing/       # Data preprocessing scripts
│   │   ├── xml_to_yolo.py       # XML to YOLO format converter
│   │   └── split_data.py        # Dataset splitting utility
│   ├── training/                 # Training modules
│   │   └── train.py             # YOLOv8 training pipeline
│   └── utils/                    # Utility functions
│
├── data/                         # Dataset and processed data
│   ├── NEU-DET/                 # Original NEU dataset
│   │   ├── train/
│   │   └── validation/
│   ├── labels/                   # YOLO format labels
│   │   ├── train/
│   │   └── validation/
│   └── dataset/                  # Train/val/test splits
│       ├── images/
│       └── labels/
│
├── configs/                      # Configuration files
│   ├── neu_defect.yaml          # Dataset configuration
│   └── train_config.yaml       # Training configuration
│
├── notebooks/                    # Jupyter notebooks
│   └── steel_defect_colab.ipynb # Google Colab training notebook
│
├── models/                       # Trained models
│   └── (best.pt, last.pt)      # Model weights
│
├── runs/                         # Training results and experiments
│   └── detect/                  # YOLOv8 training outputs
│
├── deployment/                   # Deployment configurations
│   ├── Dockerfile              # Docker container
│   └── docker-compose.yml      # Docker Compose setup
│
├── scripts/                      # Utility scripts
│   └── (deployment scripts)
│
├── docs/                         # Documentation
│
├── requirements.txt              # Python dependencies
├── setup.py                     # Package setup
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/ylmzelff/steel-defect-detection-mlops.git
cd steel-defect-detection-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

```bash
# Convert XML annotations to YOLO format
python src/data_preprocessing/xml_to_yolo.py --root data/NEU-DET --out data/labels

# Split dataset into train/validation/test sets
python src/data_preprocessing/split_data.py --input-root data/NEU-DET --labels-root data/labels --output data/dataset
```

### 3. Model Training

```bash
# Train YOLOv8 model locally
python src/training/train.py --data configs/neu_defect.yaml --epochs 50 --batch 16

# Or use Google Colab notebook
# Open notebooks/steel_defect_colab.ipynb in Google Colab
```

### 4. Model Evaluation

```bash
# Evaluate trained model
python src/training/train.py --evaluate runs/detect/steel_defect_experiment/weights/best.pt
```

## 📊 Dataset Information

- **Source**: NEU Steel Surface Defect Database
- **Total Images**: 1,800 (300 per class)
- **Image Size**: 200×200 pixels
- **Format**: JPG images with XML annotations
- **Split**: 70% train, 20% validation, 10% test

## 🛠️ Technology Stack

- **Deep Learning**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch
- **Data Processing**: OpenCV, Pillow
- **MLOps**: Weights & Biases (optional), TensorBoard
- **Deployment**: Docker, FastAPI
- **Cloud**: Google Colab, AWS/Azure (configurable)

## 📈 Training Pipeline

1. **Data Preprocessing**: Convert XML annotations to YOLO format
2. **Data Splitting**: Create reproducible train/validation/test splits
3. **Model Training**: Train YOLOv8 with optimized hyperparameters
4. **Validation**: Real-time validation during training
5. **Model Export**: Export to multiple formats (PyTorch, ONNX, TensorRT)
6. **Deployment**: Containerized deployment with API endpoints

## 🔧 Configuration

### Training Configuration (`configs/train_config.yaml`)

- Batch size optimization based on GPU memory
- Learning rate scheduling with warmup
- Data augmentation strategies
- Early stopping and model checkpointing

### Dataset Configuration (`configs/neu_defect.yaml`)

- Dataset paths and class definitions
- Training/validation/test splits
- Class distribution and statistics

## 📋 Requirements

```
ultralytics>=8.0.0
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.7.0
Pillow>=9.0.0
PyYAML>=6.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.64.0
```

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t steel-defect-detector .

# Run container
docker run -p 8080:8080 steel-defect-detector

# Or use Docker Compose
docker-compose up -d
```

## 📊 Model Performance

### Training Metrics (50 epochs)

- **Final Training Loss**: 1.165 (box) + 1.068 (cls) + 1.423 (dfl)
- **Final Validation Loss**: 1.496 (box) + 1.180 (cls) + 1.651 (dfl)
- **Precision**: 74.79%
- **Recall**: 69.39%
- **mAP50**: 76.47%
- **mAP50-95**: 43.28%

### Inference Performance

- **Speed**: ~10ms per image (Tesla T4)
- **Model Size**: 6.2MB (YOLOv8n)
- **FPS**: ~100 FPS on GPU

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NEU Steel Surface Defect Database**: Dataset providers
- **Ultralytics**: YOLOv8 implementation
- **PyTorch Team**: Deep learning framework
- **Google Colab**: Cloud training environment

## 📧 Contact

**Steel Defect Detection Team**

- GitHub: [@ylmzelff](https://github.com/ylmzelff)
- Project: [steel-defect-detection-mlops](https://github.com/ylmzelff/steel-defect-detection-mlops)

## 🔗 Useful Links

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [NEU Dataset Paper](https://scholar.google.com/scholar?q=NEU+steel+surface+defect+database)
- [Steel Defect Detection Survey](https://scholar.google.com/scholar?q=steel+surface+defect+detection+computer+vision)

---

⭐ If this project helped you, please give it a star! ⭐
