"""Detection route for uploading an image and returning YOLO detections.

POST /detect
- Accepts multipart/form-data with field `file` (image upload)
- Optional query parameter `return_image` (bool). If true the response will include a
  base64-encoded annotated image in the JSON under `annotated_image`.

Response JSON example:
{
  "detections": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.97, "class_id": 0, "class_name": "drone"},
    ...
  ],
  "annotated_image": "<base64 JPEG>" (only present if return_image=true)
}
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Query
from typing import Optional, List, Dict, Any
import numpy as np
import cv2 as cv
import base64
import io

router = APIRouter()


def _read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes to an OpenCV BGR image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return img


@router.post("/", summary="Detect objects in an uploaded image")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    return_image: bool = Query(False, description="Return annotated image as base64 in JSON"),
) -> Dict[str, Any]:
    """Handle image upload, run the YOLO model, and return detections as JSON.

    The YOLO model is expected to be available from `request.app.state.model` (set at startup).
    """
    # Get model from app state - ensure startup loaded it
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # Read bytes from upload and decode into numpy image (BGR)
    file_bytes = await file.read()
    img = _read_image_from_upload(file_bytes)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    # Run inference
    # NOTE: ultralytics YOLO returns a Results object; running model(img) returns a list-like
    # object where results[0] contains boxes, masks, keypoints, etc.
    results = model(img)
    if len(results) == 0:
        return {"detections": []}

    r = results[0]

    detections: List[Dict[str, Any]] = []

    # Extract boxes if present
    boxes = getattr(r, "boxes", None)
    if boxes is not None:
        # boxes.xyxy, boxes.conf, boxes.cls are typical fields (torch tensors or numpy)
        try:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
            classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
        except Exception:
            # Fallback: try to treat attributes as lists
            xyxy = np.array(boxes.xyxy)
            confs = np.array(boxes.conf)
            classes = np.array(boxes.cls)

        names = getattr(r, "names", {}) or {}

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = [float(x) for x in box]
            cls_id = int(classes[i]) if len(classes) > i else None
            cls_name = names.get(cls_id, str(cls_id)) if names is not None else str(cls_id)
            conf = float(confs[i]) if len(confs) > i else None

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                }
            )

    # Prepare annotated image if requested
    annotated_b64: Optional[str] = None
    if return_image:
        try:
            annotated = r.plot()  # returns annotated image (numpy array)
            # Encode as JPEG and base64
            success, encoded = cv.imencode(".jpg", annotated)
            if not success:
                raise RuntimeError("Failed to encode annotated image")
            annotated_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        except Exception as exc:
            # Do not fail the whole request if annotation fails; include a warning instead
            annotated_b64 = None

    response: Dict[str, Any] = {"detections": detections}
    if return_image:
        response["annotated_image"] = annotated_b64

    return response
