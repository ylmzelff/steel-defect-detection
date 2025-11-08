# Steel Defect Detection MLOps - Quick Start Scripts

This directory contains utility scripts for various project tasks.

## Available Scripts

### Data Preprocessing

```bash
# Convert XML annotations to YOLO format
python -m src.data_preprocessing.xml_to_yolo --root data/NEU-DET --out data/labels

# Split dataset into train/val/test
python -m src.data_preprocessing.split_data --input-root data/NEU-DET --labels-root data/labels --output data/dataset
```

### Model Training

```bash
# Train YOLOv8 model
python -m src.training.train --data configs/neu_defect.yaml --epochs 50 --batch 16

# Train with custom configuration
python -m src.training.train --model yolov8s.pt --epochs 100 --batch 8 --name steel_defect_v2
```

### API Deployment

```bash
# Start API server
python -m src.api.main

# Or with uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Docker Deployment

```bash
# Build and deploy with Docker Compose
cd deployment && ./deploy.sh

# Or deploy single container
cd deployment && ./deploy.sh -t single -p 8080
```

## Environment Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Install in Development Mode

```bash
pip install -e .
```

### Run Tests

```bash
pytest tests/ -v
```

## Configuration

All configuration files are in the `configs/` directory:

- `neu_defect.yaml`: Dataset configuration for YOLO training
- `train_config.yaml`: Training hyperparameters and settings

## Model Management

Trained models are saved in the `models/` directory:

- `best.pt`: Best model weights based on validation mAP
- `last.pt`: Latest model weights from final epoch

## Monitoring and Logging

Training logs and results are saved in `runs/detect/`:

- Training curves and metrics
- Validation results and confusion matrices
- Model performance visualizations

## Quick Commands

```bash
# Full pipeline from scratch
python -m src.data_preprocessing.xml_to_yolo
python -m src.data_preprocessing.split_data
python -m src.training.train
python -m src.api.main

# Docker deployment
docker-compose up -d

# Health check
curl http://localhost:8080/health
```
