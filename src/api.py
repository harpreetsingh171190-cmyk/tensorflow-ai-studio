"""Optional FastAPI inference service. Set MODEL_PATH before starting."""
import os
from contextlib import asynccontextmanager
from io import BytesIO
import numpy as np
import tensorflow as tf
import keras
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/cifar10_transfer.keras")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
    yield


app = FastAPI(title="Harpreet Singh ML Inference API", version="1.1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded; train/export a model and set MODEL_PATH")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 8 MiB limit")
    try:
        with Image.open(BytesIO(raw)) as image:
            image = image.convert("RGB").resize((96, 96))
            batch = np.asarray(image, dtype=np.float32)[None, ...]
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Upload must be a valid image")
    scores = np.asarray(model.predict(batch, verbose=0))[0]
    order = np.argsort(scores)[-3:][::-1]
    return {"predictions": [{"class_index": int(i), "score": float(scores[i])} for i in order]}
