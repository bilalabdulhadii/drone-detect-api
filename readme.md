# Drone Detector API ✅

A YOLOv8-based drone detection backend with a clean FastAPI interface.

This repository provides a production-ready backend API that runs inference with a pre-trained YOLO model and exposes a single image detection endpoint for easy integration with frontends (e.g. React).

---

## 🔧 Features

- FastAPI server exposing `POST /detect` for image uploads
- Model loaded once at startup for efficient inference
- Optional annotated image return (`return_image=true`)
- Healthy separation between routing and inference logic

---

## ⚙️ Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
# If you need a specific PyTorch build (CUDA/MPS/CPU), install it explicitly following https://pytorch.org/
```

> Note: `opencv-python-headless` is used for server environments (no GUI).

---

## 🚀 Quickstart (local)

1. Ensure the model weights `51ep-16-GPU.pt` are present in the repository root (or set `MODEL_PATH` to a different path).
2. Start the API server:

```bash
# recommended (works locally and in Railway):
uvicorn main:app --reload

# production:
uvicorn main:app --host 0.0.0.0 --port $PORT
```

- Health-check: `GET /` → returns `{ "status": "ok" }`
- Detection endpoint: `POST /detect/` (multipart form with `file` field)

---

## 🧪 API: POST /detect

- Content type: `multipart/form-data` with `file` (image upload)
- Optional query param: `return_image` (boolean). If `true` the response includes `annotated_image` (base64-encoded JPEG)

Response example:

```json
{
  "detections": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.97, "class_id": 0, "class_name": "drone"},
    ...
  ],
  "annotated_image": "<base64 JPEG>" // only when return_image=true
}
```

cURL example:

```bash
curl -X POST "http://127.0.0.1:8000/detect/?return_image=true" \
  -F "file=@/path/to/image.jpg"
```

---

## 📦 Deployment on Railway

Railway will detect Python projects and can run the app with the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

You can also add a `Procfile` with:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set environment variables in Railway:

- `MODEL_PATH` (optional) — path to the model weights in the project or a remote location
- `ALLOWED_ORIGINS` (optional) — comma-separated origins for CORS

Railway will set `$PORT` automatically; the server reads it when starting via `uvicorn`.

If you run into build issues on Railway, this repo includes a `Dockerfile` with the
system libraries needed by OpenCV. Railway will auto-detect and use it.

---

## ▶️ Local utilities

- `detect-img.py` — run the model on images from `test-img/` (interactive/show annotated frames)
- `detect-cam.py` — run the model on a camera or video file
- `train.py` — training entrypoint (migrated from `main.py` to keep `main.py` as the API entrypoint)

---

## ✅ Notes

- The model is loaded at startup and stored on `app.state.model` to avoid reloading per request.
- The API enforces basic validation and file size limits to improve safety and reliability.

---

## License

Add your project license here (e.g., MIT).

---

Happy detecting! 🛩️
