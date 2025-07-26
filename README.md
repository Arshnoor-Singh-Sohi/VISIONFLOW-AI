# VisionFlow AI

A powerful computer vision pipeline that combines Meta's Segment Anything Model (SAM) with OpenAI's vision capabilities for automated image segmentation and classification.

## 🚀 Features

- **Advanced Image Segmentation**: Leverage Meta's SAM model for precise object segmentation
- **AI-Powered Classification**: Use OpenAI's vision models for intelligent image classification
- **Real-time Processing**: WebSocket support for live progress updates
- **RESTful API**: Comprehensive FastAPI backend with interactive documentation
- **Modern Frontend**: React-based dashboard for intuitive interaction
- **Scalable Architecture**: Microservices design with separate SAM service
- **Training Pipeline**: Automated model training on accumulated data
- **Comprehensive Monitoring**: Built-in health checks and performance metrics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │   SAM Service   │
│    (Port 3000)   │◄──►│   (Port 8000)    │◄──►│   (Port 8001)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  SQLite Database │
                       │   File Storage   │
                       └─────────────────┘
```

## 📁 Project Structure

```
visionflow-ai/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── setup.py
│
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI main application
│   ├── config.py                  # Configuration management
│   ├── database.py                # Database models and connections
│   ├── schemas.py                 # Pydantic models
│   ├── dependencies.py            # FastAPI dependencies
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── sam_service.py         # SAM segmentation service
│   │   ├── openai_service.py      # OpenAI classification service
│   │   ├── training_service.py    # Model training pipeline
│   │   └── storage_service.py     # File and data storage
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── images.py          # Image upload/processing endpoints
│   │   │   ├── results.py         # Results viewing endpoints
│   │   │   ├── training.py        # Training management endpoints
│   │   │   └── health.py          # Health check endpoints
│   │   └── router.py              # API router configuration
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database_models.py     # SQLAlchemy models
│   │   └── ml_models.py           # ML model definitions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image_processing.py    # Image utilities
│       ├── logging.py             # Logging configuration
│       └── helpers.py             # General helper functions
│
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   └── src/
│       ├── App.js                 # Main React application
│       ├── index.js               # React entry point
│       ├── index.css              # Global styles
│       │
│       ├── components/
│       │   ├── ImageUpload.js     # Image upload component
│       │   ├── ResultsViewer.js   # Results display component
│       │   ├── TrainingDashboard.js # Training progress dashboard
│       │   ├── Navigation.js      # Navigation component
│       │   ├── LoadingSpinner.js  # Loading component
│       │   └── SystemMonitoring.js # System monitoring component
│       │
│       ├── services/
│       │   ├── api.js             # API service layer
│       │   └── websocket.js       # Real-time updates
│       │
│       └── styles/
│           ├── components.css     # Component-specific styles
│           └── dashboard.css      # Dashboard styles
│
├── sam-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                     # SAM service main application
│   ├── sam_processor.py           # SAM model wrapper
│   └── utils.py                   # SAM utilities
│
├── scheduler/
│   ├── __init__.py
│   ├── daily_processor.py         # Daily automation script
│   ├── scheduler.py               # Task scheduler
│   └── config.py                  # Scheduler configuration
│
├── data/
│   ├── images/                    # Uploaded images
│   ├── segments/                  # Segmented image parts
│   ├── results/                   # Processing results
│   │   ├── thumbnails/            # Generated thumbnails
│   │   ├── annotated/             # Annotated images
│   │   └── exports/               # Export files
│   ├── models/                    # Trained models & SAM weights
│   └── logs/                      # Application logs
│
├── tests/
│   ├── __init__.py
│   ├── test_backend/
│   │   ├── __init__.py
│   │   ├── test_services.py       # Service layer tests
│   │   └── test_endpoints.py      # API endpoint tests
│   ├── test_sam_service/
│   │   ├── __init__.py
│   │   └── test_sam.py            # SAM service tests
│   └── test_integration/
│       ├── __init__.py
│       └── test_pipeline.py       # End-to-end pipeline tests
│
├── scripts/
│   ├── setup_environment.sh      # Environment setup script
│   ├── download_sam_weights.py   # SAM model download
│   ├── database_init.py          # Database initialization
│   └── deploy.sh                 # Deployment script
│
└── docs/
    ├── API.md                     # API documentation
    ├── DEPLOYMENT.md              # Deployment guide
    ├── DEVELOPMENT.md             # Development guide
    └── ARCHITECTURE.md            # System architecture
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- 4GB+ RAM (for SAM model)
- OpenAI API key

## 🛠️ Installation

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd visionflow-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-benchmark httpx
```

### 2. Download SAM Model

```bash
# Create model directory
mkdir -p data/models

# Download SAM model (replace with actual download command)
# Place sam_vit_h_4b8939.pth in data/models/
```

### 3. Configuration

Create `.env` file in project root:

```env
SECRET_KEY=your-super-secret-key-at-least-32-characters-long
DATABASE_URL=sqlite:///./data/visionflow.db
OPENAI_API_KEY=sk-your-openai-api-key-here
SAM_SERVICE_URL=http://localhost:8001
DEBUG=true

# Optional settings
MAX_FILE_SIZE=10485760
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
LOG_LEVEL=INFO
```

### 4. Initialize Database

```bash
python scripts/database_init.py
```

### 5. Create Required Directories

```bash
mkdir -p data/{images,segments,results,models,logs}
mkdir -p data/results/{thumbnails,annotated,exports}
```

## 🚀 Running the Application

### Start SAM Service

```bash
# Terminal 1
cd sam-service
pip install segment-anything
python app.py
```

### Start Backend API

```bash
# Terminal 2
python -m backend.main
```

### Start Frontend (Optional)

```bash
# Terminal 3
cd frontend
npm install
npm start
```

## 📚 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Upload Image

```bash
curl -X POST "http://localhost:8000/api/v1/images/upload" \
  -F "image=@test_image.jpg" \
  -F "config={\"min_area\": 1000}"
```

### Check Processing Status

```bash
curl http://localhost:8000/api/v1/images/status/{image_id}
```

### Get Results

```bash
curl http://localhost:8000/api/v1/results/detailed/{image_id}
```

## 🔧 API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/images/upload` | Upload image for processing |
| GET | `/api/v1/images/status/{id}` | Get processing status |
| GET | `/api/v1/images/list` | List all images |
| GET | `/api/v1/results/detailed/{id}` | Get detailed results |
| POST | `/api/v1/training/start` | Start model training |
| GET | `/health` | Service health check |

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests
pytest tests/test_backend/test_services.py -v

# SAM service tests
pytest tests/test_sam_service/test_sam.py -v

# Integration tests
pytest tests/test_integration/test_pipeline.py -v
```

### Performance Benchmarks

```bash
pip install pytest-benchmark
pytest tests/ -m benchmark --benchmark-only
```

## 🐳 Docker Deployment

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔍 Monitoring

### Database Monitoring

```python
python scripts/db_monitor.py
```

### Application Logs

```bash
tail -f data/logs/visionflow.log
```

### WebSocket Testing

```python
python scripts/websocket_test.py
```

## ⚠️ Troubleshooting

### Common Issues

**SAM Model Not Loading**
- Verify model file location: `data/models/sam_vit_h_4b8939.pth`
- Check file permissions: `chmod 644 data/models/sam_vit_h_4b8939.pth`
- Ensure sufficient RAM (2-3GB minimum)

**OpenAI API Errors**
- Verify API key in `.env` file
- Check account credits and permissions
- Test with: `curl https://api.openai.com/v1/models -H "Authorization: Bearer YOUR_API_KEY"`

**Port Conflicts**
- Check port usage: `lsof -i :8000` and `lsof -i :8001`
- Modify ports in configuration if needed

**Database Issues**
- Ensure `data/` directory is writable
- Reset database: `rm data/visionflow.db && python scripts/database_init.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 backend/
black backend/

# Run tests with coverage
pytest tests/ --cov=backend --cov-report=html
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Meta's Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [OpenAI API](https://openai.com/api/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the comprehensive testing guide in `docs/`

---

**Happy coding!** 🚀
