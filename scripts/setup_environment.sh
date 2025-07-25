#!/bin/bash

# VisionFlow AI - Environment Setup Script
# =========================================
# 
# This script sets up the complete development environment for VisionFlow AI.
# Think of this as your "construction crew" that prepares the entire worksite
# before you can start building - it creates directories, installs dependencies,
# downloads models, and configures everything you need.
#
# Usage:
#   ./scripts/setup_environment.sh [--production] [--gpu] [--no-models]

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.11"
NODE_VERSION="18"

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Parse command line arguments
PRODUCTION=false
GPU_SUPPORT=false
SKIP_MODELS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --production)
            PRODUCTION=true
            shift
            ;;
        --gpu)
            GPU_SUPPORT=true
            shift
            ;;
        --no-models)
            SKIP_MODELS=true
            shift
            ;;
        -h|--help)
            echo "VisionFlow AI Environment Setup"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --production    Set up for production deployment"
            echo "  --gpu           Install GPU support for PyTorch"
            echo "  --no-models     Skip downloading large model files"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 VisionFlow AI Environment Setup"
echo "=================================="
echo ""

if [[ "$PRODUCTION" == true ]]; then
    log_info "Setting up PRODUCTION environment"
else
    log_info "Setting up DEVELOPMENT environment"
fi

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_success "Operating System: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_success "Operating System: macOS"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        log_success "Operating System: Windows"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_success "Python found: $PYTHON_VER"
        
        # Check if version is compatible
        if [[ $(echo "$PYTHON_VER >= 3.8" | bc -l) -eq 1 ]]; then
            log_success "Python version is compatible"
        else
            log_error "Python 3.8+ required, found $PYTHON_VER"
            exit 1
        fi
    else
        log_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check Node.js (for frontend)
    if command -v node &> /dev/null; then
        NODE_VER=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
        log_success "Node.js found: v$(node --version | cut -d'v' -f2)"
    else
        log_warning "Node.js not found. Frontend development will not be available."
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        log_success "Git found: $(git --version | cut -d' ' -f3)"
    else
        log_error "Git not found. Please install Git."
        exit 1
    fi
    
    # Check for GPU (if requested)
    if [[ "$GPU_SUPPORT" == true ]]; then
        if command -v nvidia-smi &> /dev/null; then
            log_success "NVIDIA GPU detected"
            nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | while read gpu; do
                log_info "GPU: $gpu"
            done
        else
            log_warning "NVIDIA GPU not detected or drivers not installed"
            log_info "Continuing with CPU-only setup"
            GPU_SUPPORT=false
        fi
    fi
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    # Data directories - this is where your application stores all its information
    mkdir -p "$PROJECT_ROOT/data/images"        # Uploaded images
    mkdir -p "$PROJECT_ROOT/data/segments"      # Image segments from SAM
    mkdir -p "$PROJECT_ROOT/data/results"       # Processing results
    mkdir -p "$PROJECT_ROOT/data/models"        # AI model files
    mkdir -p "$PROJECT_ROOT/data/logs"          # Application logs
    mkdir -p "$PROJECT_ROOT/data/exports"       # Data exports
    mkdir -p "$PROJECT_ROOT/data/thumbnails"    # Image thumbnails
    mkdir -p "$PROJECT_ROOT/data/backups"       # Database backups
    
    # Configuration directories
    mkdir -p "$PROJECT_ROOT/config"             # Configuration files
    mkdir -p "$PROJECT_ROOT/nginx"              # Nginx configuration
    
    # Development directories
    if [[ "$PRODUCTION" == false ]]; then
        mkdir -p "$PROJECT_ROOT/data/dev"       # Development data
        mkdir -p "$PROJECT_ROOT/data/test"      # Test data
    fi
    
    log_success "Directory structure created"
}

# Set up Python virtual environment
setup_python_environment() {
    log_info "Setting up Python virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip to latest version
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    log_info "Installing Python dependencies..."
    if [[ "$GPU_SUPPORT" == true ]]; then
        # Install PyTorch with GPU support first
        log_info "Installing PyTorch with GPU support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    
    # Install all requirements
    pip install -r requirements.txt
    
    # Install development dependencies if not production
    if [[ "$PRODUCTION" == false ]]; then
        log_info "Installing development dependencies..."
        pip install pytest pytest-asyncio pytest-cov black flake8 mypy
    fi
    
    log_success "Python environment setup complete"
}

# Download SAM model weights
download_sam_models() {
    if [[ "$SKIP_MODELS" == true ]]; then
        log_warning "Skipping model downloads (--no-models flag)"
        return
    fi
    
    log_info "Downloading SAM model weights..."
    
    # Run our Python script to download SAM weights
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    python scripts/download_sam_weights.py
    
    log_success "SAM models downloaded"
}

# Set up Node.js environment (for frontend)
setup_node_environment() {
    if ! command -v node &> /dev/null; then
        log_warning "Node.js not found, skipping frontend setup"
        return
    fi
    
    log_info "Setting up Node.js environment..."
    
    cd "$PROJECT_ROOT/frontend"
    
    # Install dependencies
    if [[ -f "package.json" ]]; then
        log_info "Installing Node.js dependencies..."
        npm install
        log_success "Node.js dependencies installed"
    else
        log_warning "package.json not found, skipping Node.js setup"
    fi
}

# Create configuration files
create_config_files() {
    log_info "Creating configuration files..."
    
    # Create .env file from example if it doesn't exist
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
            cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
            log_success "Created .env file from example"
            log_warning "Please edit .env file with your actual configuration values"
        else
            # Create basic .env file
            cat > "$PROJECT_ROOT/.env" << EOF
# VisionFlow AI Configuration
# Copy this file to .env and update with your actual values

# Core Application
SECRET_KEY=$(openssl rand -hex 32)
DEBUG=true

# Database (for development - use PostgreSQL in production)
DATABASE_URL=sqlite:///./data/visionflow.db

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# OpenAI API (REQUIRED - get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_api_key_here

# SAM Service
SAM_SERVICE_URL=http://localhost:8001
SAM_MODEL_TYPE=vit_h
SAM_DEVICE=cpu

# CORS Origins (add your frontend URL)
CORS_ORIGINS=http://localhost:3000

# File Upload
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp

# Logging
LOG_LEVEL=INFO
LOG_FILE=./data/logs/visionflow.log

# Training
ENABLE_TRAINING=true
MIN_TRAINING_SAMPLES=100

# Scheduler
ENABLE_SCHEDULER=true
DAILY_PROCESSING_TIME=09:00
EOF
            log_success "Created basic .env file"
            log_warning "Please edit .env file and add your OpenAI API key"
        fi
    else
        log_info ".env file already exists"
    fi
    
    # Set proper permissions on sensitive files
    chmod 600 "$PROJECT_ROOT/.env"
    
    # Create logging configuration
    mkdir -p "$PROJECT_ROOT/config"
    cat > "$PROJECT_ROOT/config/logging.json" << EOF
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "detailed"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "./data/logs/visionflow.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "level": "DEBUG",
            "formatter": "detailed"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}
EOF
    
    log_success "Configuration files created"
}

# Set up database
setup_database() {
    log_info "Setting up database..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Initialize database
    python scripts/database_init.py
    
    log_success "Database setup complete"
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        # Ubuntu/Debian
        if command -v apt-get &> /dev/null; then
            log_info "Installing dependencies via apt..."
            sudo apt-get update
            sudo apt-get install -y \
                python3-dev \
                python3-pip \
                python3-venv \
                build-essential \
                libmagic1 \
                libmagic-dev \
                libpq-dev \
                redis-server \
                curl \
                wget \
                git
        # RedHat/CentOS/Fedora
        elif command -v yum &> /dev/null; then
            log_info "Installing dependencies via yum..."
            sudo yum install -y \
                python3-devel \
                python3-pip \
                gcc \
                gcc-c++ \
                file-devel \
                postgresql-devel \
                redis \
                curl \
                wget \
                git
        fi
        
    elif [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            log_info "Installing dependencies via Homebrew..."
            brew install \
                python@3.11 \
                libmagic \
                postgresql \
                redis \
                git
        else
            log_warning "Homebrew not found. Please install manually or install Homebrew first."
        fi
    fi
    
    log_success "System dependencies installed"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Test Python imports
    python -c "
import fastapi
import sqlalchemy
import openai
import PIL
import cv2
import numpy
print('✓ All Python packages import successfully')
"
    
    # Test configuration loading
    python -c "
import sys
sys.path.insert(0, 'backend')
from backend.config import get_settings
settings = get_settings()
print('✓ Configuration loads successfully')
"
    
    # Check if model files exist (if downloaded)
    if [[ -f "$PROJECT_ROOT/data/models/sam_vit_h_4b8939.pth" ]]; then
        log_success "SAM model weights found"
    elif [[ "$SKIP_MODELS" == false ]]; then
        log_warning "SAM model weights not found"
    fi
    
    log_success "Installation verification complete"
}

# Main setup function
main() {
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run setup steps
    check_system_requirements
    install_system_dependencies
    create_directories
    setup_python_environment
    download_sam_models
    setup_node_environment
    create_config_files
    setup_database
    verify_installation
    
    echo ""
    echo "🎉 VisionFlow AI environment setup complete!"
    echo ""
    log_success "Next steps:"
    echo "  1. Edit .env file with your OpenAI API key"
    echo "  2. Activate virtual environment: source venv/bin/activate"
    echo "  3. Start the backend: uvicorn backend.main:app --reload"
    echo "  4. In another terminal, start frontend: cd frontend && npm start"
    echo "  5. Visit http://localhost:3000 to access the application"
    echo ""
    echo "For production deployment:"
    echo "  docker-compose up -d"
    echo ""
    log_info "Happy coding! 🚀"
}

# Run main function
main "$@"