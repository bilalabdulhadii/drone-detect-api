"""Entrypoint for the FastAPI app.

This module exposes `app` at module-level so the process can be started with:

    uvicorn main:app --host 0.0.0.0 --port $PORT

If you want to train the model, use `python train.py` instead.
"""
from app.main import app

if __name__ == "__main__":
    # When run directly, start a local uvicorn server for convenience
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")