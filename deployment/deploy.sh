#!/bin/bash

# Steel Defect Detection Deployment Script
# This script builds and deploys the steel defect detection service

set -e

# Configuration
PROJECT_NAME="steel-defect-detection"
DOCKER_IMAGE="${PROJECT_NAME}:latest"
CONTAINER_NAME="${PROJECT_NAME}-container"
PORT="8080"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        warn "Docker Compose is not installed. Using docker compose instead"
    fi
    
    log "Dependencies check passed"
}

# Build Docker image
build_image() {
    log "Building Docker image..."
    
    # Navigate to project root
    cd "$(dirname "$0")/.."
    
    # Build image
    docker build -t ${DOCKER_IMAGE} -f deployment/Dockerfile . || error "Failed to build Docker image"
    
    log "Docker image built successfully: ${DOCKER_IMAGE}"
}

# Deploy with Docker Compose
deploy_compose() {
    log "Deploying with Docker Compose..."
    
    cd deployment
    
    # Stop existing services
    docker-compose down 2>/dev/null || true
    
    # Start services
    docker-compose up -d || error "Failed to start services with Docker Compose"
    
    log "Services deployed successfully"
}

# Deploy single container
deploy_single() {
    log "Deploying single container..."
    
    # Stop and remove existing container
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8080 \
        -v $(pwd)/models:/app/models:ro \
        -e MODEL_PATH=/app/models/best.pt \
        -e CONFIDENCE_THRESHOLD=0.25 \
        -e LOG_LEVEL=INFO \
        --restart unless-stopped \
        ${DOCKER_IMAGE} || error "Failed to start container"
    
    log "Container deployed successfully: ${CONTAINER_NAME}"
}

# Check service health
check_health() {
    log "Checking service health..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:${PORT}/health > /dev/null 2>&1; then
            log "Service is healthy"
            return 0
        fi
        
        log "Waiting for service to start (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    error "Service health check failed"
}

# Show deployment info
show_info() {
    log "Deployment completed successfully!"
    echo ""
    echo "Service Information:"
    echo "  - API URL: http://localhost:${PORT}"
    echo "  - Health Check: http://localhost:${PORT}/health"
    echo "  - API Documentation: http://localhost:${PORT}/docs"
    echo "  - Interactive API: http://localhost:${PORT}/redoc"
    echo ""
    echo "Management Commands:"
    echo "  - View logs: docker logs ${CONTAINER_NAME}"
    echo "  - Stop service: docker stop ${CONTAINER_NAME}"
    echo "  - Remove service: docker rm ${CONTAINER_NAME}"
    echo "  - View running containers: docker ps"
    echo ""
}

# Main deployment function
main() {
    local deploy_type="${1:-compose}"
    
    log "Starting Steel Defect Detection deployment..."
    log "Deployment type: $deploy_type"
    
    check_dependencies
    build_image
    
    case $deploy_type in
        "compose")
            deploy_compose
            ;;
        "single")
            deploy_single
            ;;
        *)
            error "Unknown deployment type: $deploy_type. Use 'compose' or 'single'"
            ;;
    esac
    
    check_health
    show_info
}

# Parse command line arguments
while getopts "t:p:h" opt; do
    case $opt in
        t)
            DEPLOY_TYPE="$OPTARG"
            ;;
        p)
            PORT="$OPTARG"
            ;;
        h)
            echo "Usage: $0 [-t deployment_type] [-p port] [-h]"
            echo "  -t: Deployment type (compose|single) [default: compose]"
            echo "  -p: Port number [default: 8080]"
            echo "  -h: Show this help"
            exit 0
            ;;
        \?)
            error "Invalid option: -$OPTARG"
            ;;
    esac
done

# Run main function
main "${DEPLOY_TYPE:-compose}"