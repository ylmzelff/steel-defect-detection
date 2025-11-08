# Documentation for Steel Defect Detection MLOps Pipeline

This directory contains comprehensive documentation for the steel defect detection project.

## Documentation Structure

### [API Documentation](api.md)

- REST API endpoints and usage
- Request/response formats
- Authentication and rate limiting
- Example API calls

### [Model Documentation](model.md)

- YOLOv8 architecture details
- Training methodology and hyperparameters
- Performance benchmarks and comparisons
- Model export and optimization

### [Dataset Documentation](dataset.md)

- NEU Steel Surface Defect Database overview
- Class definitions and distributions
- Data preprocessing pipeline
- Augmentation strategies

### [Deployment Guide](deployment.md)

- Docker containerization
- Kubernetes deployment
- Cloud deployment (AWS/Azure/GCP)
- Monitoring and logging setup

### [Development Guide](development.md)

- Development environment setup
- Code structure and conventions
- Testing guidelines
- Contribution workflow

### [MLOps Pipeline](mlops.md)

- Continuous Integration/Deployment
- Experiment tracking with Weights & Biases
- Model versioning and registry
- Performance monitoring

## Quick References

### Model Performance

- **mAP50**: 76.47% on NEU test set
- **mAP50-95**: 43.28% on NEU test set
- **Inference Speed**: ~10ms per image (Tesla T4 GPU)
- **Model Size**: 6.2MB (YOLOv8n)

### Defect Classes

1. **Crazing**: Surface cracking patterns
2. **Inclusion**: Foreign material inclusions
3. **Patches**: Irregular surface patches
4. **Pitted Surface**: Small surface pits and holes
5. **Rolled-in Scale**: Scale defects from rolling process
6. **Scratches**: Linear surface scratches

### Key Features

- End-to-end MLOps pipeline
- Containerized deployment
- REST API with OpenAPI docs
- Real-time inference
- Batch processing support
- Comprehensive monitoring

## Getting Help

- Check the [FAQ](faq.md) for common questions
- Review [troubleshooting guide](troubleshooting.md) for common issues
- Submit issues on [GitHub](https://github.com/ylmzelff/steel-defect-detection-mlops/issues)
- Contact the development team for support

## Contributing

See the [Development Guide](development.md) for information on:

- Setting up development environment
- Code style and conventions
- Testing requirements
- Pull request process
