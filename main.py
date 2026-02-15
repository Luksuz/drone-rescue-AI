import os
import uuid
import shutil
import base64
from pathlib import Path

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

app = FastAPI(title="Aerial Rescue AI")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/public", StaticFiles(directory=str(BASE_DIR / "public")), name="public")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "drone-icerc/2"
API_URL = f"https://detect.roboflow.com/{MODEL_ID}"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Annotation colors
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 255, 0)


MAX_DIMENSION = 1280


def resize_image_bytes(image_bytes: bytes, orig_w: int, orig_h: int) -> tuple[bytes, float]:
    """Resize image if it exceeds MAX_DIMENSION. Returns (jpeg_bytes, scale_factor)."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes(), scale


def call_roboflow_api(image_bytes: bytes, orig_w: int, orig_h: int) -> tuple[dict, float]:
    """Send image bytes to Roboflow hosted API and return predictions + scale."""
    resized_bytes, scale = resize_image_bytes(image_bytes, orig_w, orig_h)
    b64 = base64.b64encode(resized_bytes).decode("utf-8")
    resp = httpx.post(
        API_URL,
        params={"api_key": ROBOFLOW_API_KEY, "confidence": 15},
        data=b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json(), scale


def parse_predictions(api_result: dict, scale: float = 1.0):
    """Parse Roboflow API response into a list of detections, scaling bboxes back."""
    inv_scale = 1.0 / scale if scale != 0 else 1.0
    detections = []
    for pred in api_result.get("predictions", []):
        x_center = pred["x"] * inv_scale
        y_center = pred["y"] * inv_scale
        w = pred["width"] * inv_scale
        h = pred["height"] * inv_scale
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        detections.append({
            "class": pred["class"],
            "confidence": round(pred["confidence"], 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    return detections


def annotate_image(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on an image."""
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)

    return annotated


def run_image_inference(image_path: str, result_path: str, clean_path: str) -> list:
    """Run inference on a single image via Roboflow API."""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    api_result, scale = call_roboflow_api(image_bytes, w, h)
    detections = parse_predictions(api_result, scale)

    annotated = annotate_image(image, detections)
    cv2.imwrite(result_path, annotated)
    cv2.imwrite(clean_path, image)

    return detections


def run_video_inference(video_path: str, result_path: str, clean_path: str) -> list:
    """Run inference on each frame of a video via Roboflow API."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    out_clean = cv2.VideoWriter(clean_path, fourcc, fps, (width, height))

    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buf = cv2.imencode(".jpg", frame)
        api_result, scale = call_roboflow_api(buf.tobytes(), width, height)
        detections = parse_predictions(api_result, scale)
        all_detections.extend(detections)

        annotated = annotate_image(frame, detections)
        out.write(annotated)
        out_clean.write(frame)

    cap.release()
    out.release()
    out_clean.release()
    return all_detections


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    public_dir = BASE_DIR / "public"
    samples = []
    if public_dir.exists():
        for f in sorted(public_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                samples.append(f.name)
    return templates.TemplateResponse("index.html", {"request": request, "samples": samples})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {suffix}"},
        )

    unique_id = uuid.uuid4().hex[:10]
    upload_path = UPLOAD_DIR / f"{unique_id}{suffix}"

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    is_video = suffix in VIDEO_EXTENSIONS
    result_ext = ".mp4" if is_video else suffix
    result_filename = f"{unique_id}_result{result_ext}"
    clean_filename = f"{unique_id}_clean{result_ext}"
    result_path = RESULTS_DIR / result_filename
    clean_path = RESULTS_DIR / clean_filename

    try:
        if is_video:
            detections = run_video_inference(str(upload_path), str(result_path), str(clean_path))
        else:
            detections = run_image_inference(str(upload_path), str(result_path), str(clean_path))

        result_url = f"/static/results/{result_filename}"
        clean_url = f"/static/results/{clean_filename}"

        return JSONResponse(content={
            "result_url": result_url,
            "clean_url": clean_url,
            "detections": detections,
            "is_video": is_video,
            "total_detections": len(detections),
        })
    finally:
        if upload_path.exists():
            upload_path.unlink()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
