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
│       │   └── LoadingSpinner.js  # Loading component
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
│   ├── models/                    # Trained models
│   └── logs/                      # Application logs
│
├── tests/
│   ├── __init__.py
│   ├── test_backend/
│   │   ├── __init__.py
│   │   ├── test_services.py
│   │   └── test_endpoints.py
│   ├── test_sam_service/
│   │   ├── __init__.py
│   │   └── test_sam.py
│   └── test_integration/
│       ├── __init__.py
│       └── test_pipeline.py
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