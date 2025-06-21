from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.youtube_router import router as youtube_router, shutdown
import uvicorn
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube Video Processing API",
    description="API for processing YouTube videos, extracting captions, and generating embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(youtube_router)

@app.on_event("startup")
async def startup_event():
    """Initialize services when application starts"""
    logger.info("Starting YouTube Video Processing API")

@app.on_event("shutdown")
async def app_shutdown():
    """Clean up resources when application shuts down"""
    logger.info("Shutting down YouTube Video Processing API")
    shutdown()
    logger.info("Cleanup completed")

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": app.version}

if __name__ == "__main__":
    # Get configuration from environment variables with defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", 1))
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Reload: {reload}, Workers: {workers}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_config=None,
        access_log=False
    )