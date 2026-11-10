from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os
import time
import json
import urllib.request
import urllib.error

app = FastAPI(title="GaleyGrid API", version="1.0")

# CORS — allow requests from any origin (Vercel, localhost, etc)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sign-in check — /predict only answers GaleyGrid users signed in through Supabase.
# The token is confirmed with Supabase Auth (no secret needed here) and cached briefly.
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qzopthbeiavsiyvcpfnx.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF6b3B0aGJlaWF2c2l5dmNwZm54Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM4NzcyNzEsImV4cCI6MjA4OTQ1MzI3MX0.tWTWDGpfJpa6yLd2avc2OLEHldyZwDr7hrUWqaCSnrM")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "1") != "0"
TOKEN_CACHE_SECONDS = 300
_token_cache = {}  # token -> (user_id, expires_at)


def _supabase_user_id(token):
    req = urllib.request.Request(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp).get("id")
    except urllib.error.HTTPError:
        return None


async def require_user(authorization):
    if not REQUIRE_AUTH:
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in to GaleyGrid to analyze images")
    token = authorization[7:].strip()
    now = time.time()
    cached = _token_cache.get(token)
    if cached and cached[1] > now:
        return cached[0]
    user_id = await run_in_threadpool(_supabase_user_id, token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Sign in to GaleyGrid to analyze images")
    if len(_token_cache) > 1000:
        _token_cache.clear()
    _token_cache[token] = (user_id, now + TOKEN_CACHE_SECONDS)
    return user_id

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
async def predict(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    await require_user(authorization)
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
