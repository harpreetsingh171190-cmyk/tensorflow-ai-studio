"""FastAPI demo endpoints. No model is loaded automatically; configure MODEL_PATH to enable predictions."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chatbot import answer

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.keras"))
_model: Any = None
_model_error: str | None = None


def _load_model() -> None:
    """Load MODEL_PATH if it exists; remember any failure so /health can report it."""
    global _model, _model_error
    _model, _model_error = None, None
    if not MODEL_PATH.exists():
        return
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as exc:  # the API stays usable for chat/health
        _model_error = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Harpreet Singh ML Learning API", version="2026.2",
              description="Educational chatbot and optional TensorFlow model inference.", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class PredictRequest(BaseModel):
    values: list[float] = Field(min_length=1, max_length=10000)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None,
            "model_path": str(MODEL_PATH), "model_load_error": _model_error}


@app.post("/chat")
def chat(body: ChatRequest):
    return {"reply": answer(body.message), "mode": "offline-faq-demo"}


@app.post("/predict")
def predict(body: PredictRequest):
    if _model is None:
        detail = "No compatible model loaded. Set MODEL_PATH to a saved .keras model and restart the API."
        if _model_error:
            detail += f" Load error: {_model_error}"
        raise HTTPException(status_code=503, detail=detail)
    try:
        import numpy as np
        result = _model.predict(np.asarray([body.values], dtype="float32"), verbose=0)
        return {"predictions": np.asarray(result).tolist()}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Input shape does not match the loaded model: {exc}") from exc
