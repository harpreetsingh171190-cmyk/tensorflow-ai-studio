"""Small CLI for a saved Keras image classifier (Harpreet Singh, 2026)."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from PIL import Image


def load_image(path: str, image_size: int = 96) -> np.ndarray:
    """Load an image as an RGB float32 batch in [0, 255]."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((image_size, image_size))
        return np.asarray(im, dtype=np.float32)[None, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to an image")
    parser.add_argument("--model", default="artifacts/cifar10_transfer.keras")
    parser.add_argument("--labels", default="", help="optional JSON list of class labels")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    model = keras.models.load_model(args.model)
    probs = np.asarray(model.predict(load_image(args.image), verbose=0))[0]
    k = min(args.top_k, len(probs))
    labels = json.loads(Path(args.labels).read_text()) if args.labels else [str(i) for i in range(len(probs))]
    if len(labels) != len(probs):
        raise ValueError(f"Got {len(labels)} labels for {len(probs)} model outputs")
    for idx in np.argsort(probs)[-k:][::-1]:
        print(f"{labels[int(idx)]}\t{float(probs[idx]):.4f}")

if __name__ == "__main__":
    main()
