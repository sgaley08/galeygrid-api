from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os
import time

app = FastAPI(title="GaleyGrid API", version="1.0")

# CORS — allow requests from any origin (Vercel, localhost, etc)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
model = YOLO(MODEL_PATH)

# Landmark key order — must match training
LM_KEYS = [
    'right_teardrop', 'left_teardrop',
    'right_lesser_trochanter', 'left_lesser_trochanter',
    'right_ischial_tuberosity', 'left_ischial_tuberosity',
    'symphysis_superior', 'symphysis_inferior',
    'coccyx_superior', 'coccyx_inferior',
    'right_obturator_center', 'left_obturator_center',
]

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH, "landmarks": len(LM_KEYS)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    
    # Read image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img_w, img_h = img.size
    
    # Run inference
    results = model(img, conf=0.3, verbose=False)
    
    # Extract keypoints
    landmarks = {}
    confidence = 0.0
    bbox = None
    
    if results and len(results) > 0:
        r = results[0]
        
        # Bounding box
        if r.boxes is not None and len(r.boxes) > 0:
            box = r.boxes[0]
            bbox = {
                "x1": round(box.xyxy[0][0].item()),
                "y1": round(box.xyxy[0][1].item()),
                "x2": round(box.xyxy[0][2].item()),
                "y2": round(box.xyxy[0][3].item()),
                "confidence": round(box.conf[0].item(), 3)
            }
            confidence = bbox["confidence"]
        
        # Keypoints
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kp = r.keypoints.data[0]  # first detection
            for i, key in enumerate(LM_KEYS):
                if i < len(kp):
                    x = round(kp[i][0].item())
                    y = round(kp[i][1].item())
                    conf = round(kp[i][2].item(), 3)
                    # Only include if confidence > threshold
                    if conf > 0.3:
                        landmarks[key] = {"x": x, "y": y, "confidence": conf}
                    else:
                        landmarks[key] = {"x": None, "y": None, "confidence": conf}
                else:
                    landmarks[key] = {"x": None, "y": None, "confidence": 0}
    
    elapsed = round((time.time() - start) * 1000)
    
    return {
        "landmarks": landmarks,
        "bbox": bbox,
        "image_size": {"w": img_w, "h": img_h},
        "inference_ms": elapsed,
        "model_confidence": confidence,
    }
