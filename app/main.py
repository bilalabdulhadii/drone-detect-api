"""FastAPI entrypoint for the cnn-course YOLO detection API.

- Loads the YOLO model once at startup and stores it on the app state.
- Mounts routes from `app.routes.detect`.
- Adds CORS middleware so a React frontend can call the API directly.

Run locally:

pip install fastapi uvicorn python-multipart
# also install dependencies in repo:
pip install -r requirements.txt
uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import torch
from ultralytics import YOLO

from app.routes import detect as detect_router

logger = logging.getLogger("uvicorn")

app = FastAPI(title="cnn-course YOLO Detection API")

# Simple CORS config to allow a React frontend running on a different host to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production set this to your frontend origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(detect_router.router, prefix="/detect", tags=["detect"])


@app.on_event("startup")
async def load_model_on_startup():
    """Load the YOLO model once during application startup and store on app.state.

    This ensures the model is not reloaded on every request, improving performance.
    """
    # Choose device similarly to existing project training script (main.py)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        model_path = "51ep-16-GPU.pt"
        logger.info(f"Loading model {model_path} on device={device}")
        # Load model; ultralytics internally manages device allocation but we log the selection
        app.state.model = YOLO(model_path)
        logger.info("Model loaded and ready")
    except Exception as exc:
        logger.exception("Failed to load model on startup")
        # Re-raise so the process fails fast and you get an obvious error on startup
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup: remove model from memory when shutting down (best-effort)."""
    try:
        if hasattr(app.state, "model"):
            del app.state.model
            logger.info("Model has been removed from app state")
    except Exception:
        logger.exception("Error during shutdown cleanup")
