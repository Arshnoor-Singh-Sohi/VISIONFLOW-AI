"""
VisionFlow AI - Minimal Main Application for Testing
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Simplified app without complex imports
app = FastAPI(
    title="VisionFlow AI",
    description="Computer vision pipeline",
    version="1.0.0"
)

# Basic CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "VisionFlow AI is running",
        "status": "ok",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "VisionFlow AI"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )