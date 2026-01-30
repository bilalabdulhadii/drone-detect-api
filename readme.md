# Drone Detector API ✅

A lightweight YOLOv8-based drone detection project that provides:

- A FastAPI server that exposes an image detection endpoint (`POST /detect`).
- Simple scripts to run detection on images and videos (`detect-img.py`, `detect-cam.py`).
- Training utilities and a training entrypoint (`main.py`) using Ultralytics YOLO.

---

## 🔧 Features

- Single-shot YOLOv8 model (weights: `51ep-16-GPU.pt`) loaded on startup for fast inference
- API: upload an image and receive bounding boxes, confidence, class id/name
- Optional base64-encoded annotated image return (`return_image=true`)
- Example scripts for image and video processing

---

## ⚙️ Requirements

- Python 3.10+
- macOS / Linux / Windows
- GPU recommended (MPS for Apple Silicon or CUDA) but CPU works

Install minimal Python deps:

```bash
pip install -r requirements.txt
# For the API server you also need:
pip install fastapi uvicorn python-multipart
```

---

## 🚀 Quickstart

1. Make sure the model weights are present in the repo root: `51ep-16-GPU.pt`.
2. Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

- The detection endpoint will be available at: `POST http://127.0.0.1:8000/detect/`.

---

## 🧪 API: POST /detect

- Content type: `multipart/form-data` with field `file` (image upload)
- Optional query param: `return_image` (boolean). If `true`, the response JSON will include `annotated_image` (base64-encoded JPEG).

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

## ▶️ Run detection scripts

- Detect on images in `test-img/` (shows annotated frames):

```bash
python detect-img.py
```

- Detect on videos in `test-video/` and write annotated video files:

```bash
python detect-cam.py
```

---

## 🧠 Training

Training entrypoint: `main.py`.

```bash
python main.py
```

It will attempt to resume from the most recent checkpoint in `runs/detect/*/weights/last.pt`. If none found, it will try to use `51ep-16-GPU.pt` or fall back to a yolov8 base model.

Training options (set in `main.py`): epochs, batch size, img size, mixed precision, caching.

---

## 📁 Project layout

- `app/` - FastAPI app and routes (`/detect`)
- `detect-img.py` - run model on images
- `detect-cam.py` - run model on videos
- `main.py` - training entrypoint
- `51ep-16-GPU.pt` - model weights (checkpoint)
- `data.yaml` - dataset config used for training
- `drone_dataset_yolo/` - dataset files (labels / txts)

---

## 💡 Notes & Tips

- The server loads the model at startup to avoid reloading on each request. If model fails to load, check logs and weight path.
- For best performance on Apple Silicon use MPS (`torch.backends.mps`), for NVIDIA GPUs use CUDA.

---

## Contributing

Contributions are welcome. Open an issue or submit a pull request with a clear description of changes.

---

## License

Add your project license here (e.g., MIT).

---

Happy detecting! 🛩️
