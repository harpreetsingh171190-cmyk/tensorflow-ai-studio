"""Optional computer-vision utilities for the Harpreet Singh learning dashboard.

Prediction and training require TensorFlow. GPU telemetry is best-effort and may
be unavailable on CPU-only systems or inside restricted containers.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gpu_status() -> dict[str, Any]:
    """Return best-effort GPU information without requiring NVIDIA tooling."""
    result: dict[str, Any] = {"available": False, "devices": [], "nvidia_smi": None, "note": "GPU telemetry is best-effort."}
    try:
        import tensorflow as tf
        devices = tf.config.list_physical_devices("GPU")
        result["devices"] = [d.name for d in devices]
        result["available"] = bool(devices)
        result["tensorflow_version"] = tf.__version__
    except Exception as exc:  # optional dependency / driver failure
        result["tensorflow_error"] = f"{type(exc).__name__}: {exc}"
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if proc.returncode == 0:
            rows = []
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 5:
                    rows.append({"name": parts[0], "utilization_percent": parts[1],
                                 "memory_used_mb": parts[2], "memory_total_mb": parts[3],
                                 "temperature_c": parts[4]})
            result["nvidia_smi"] = rows
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def predict_image(image_bytes: bytes, model_path: str | None = None, labels_path: str | None = None,
                  top_k: int = 5) -> dict[str, Any]:
    """Classify image with a saved Keras model or pretrained ImageNet MobileNetV2."""
    if not image_bytes:
        raise ValueError("Image upload is empty.")
    if not 1 <= int(top_k) <= 20:
        raise ValueError("top_k must be between 1 and 20.")
    try:
        import numpy as np
        import tensorflow as tf
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies from requirements-vision.txt first.") from exc

    model = None
    target_size = (224, 224)
    path = None
    if model_path:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Saved model not found: {path}")
        model = tf.keras.models.load_model(path)
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) >= 3 and shape[1] and shape[2]:
            target_size = (int(shape[1]), int(shape[2]))

    try:
        import io
        with Image.open(io.BytesIO(image_bytes)) as im:
            image = im.convert("RGB").resize((target_size[1], target_size[0]))
            arr = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a supported, readable image.") from exc

    x = np.expand_dims(arr, axis=0)
    if model_path:
        # Custom models trained by this dashboard use MobileNetV2 preprocessing.
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        raw = np.asarray(model.predict(x, verbose=0))[0].reshape(-1)
        labels = None
        if labels_path:
            lp = Path(labels_path).expanduser()
            if lp.is_file():
                labels = json.loads(lp.read_text(encoding="utf-8"))
        if not labels:
            labels = [f"class_{i}" for i in range(len(raw))]
        scores = raw
        model_name = path.name
    else:
        model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        raw = model.predict(x, verbose=0)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(raw, top=min(int(top_k), 1000))[0]
        return {"model": "MobileNetV2 / ImageNet pretrained", "predictions": [
            {"label": label, "class_id": class_id, "confidence": float(score)}
            for class_id, label, score in decoded[:int(top_k)]
        ], "warning": "General ImageNet labels only; predictions can be wrong and are not a safety-critical decision."}

    if len(labels) != len(scores):
        labels = [f"class_{i}" for i in range(len(scores))]
    indices = np.argsort(scores)[::-1][:min(int(top_k), len(scores))]
    return {"model": model_name, "predictions": [
        {"label": str(labels[int(i)]), "class_id": int(i), "confidence": float(scores[int(i)])}
        for i in indices
    ], "warning": "Check the model's documented output semantics; scores may not be calibrated probabilities."}


def retrain_image_classifier(dataset_dir: str, output_dir: str = "artifacts/vision",
                             epochs: int = 5, batch_size: int = 16, image_size: int = 160) -> dict[str, Any]:
    """Train a transfer-learning image classifier from class-named subfolders.

    Layout: dataset_dir/class_a/*.jpg and dataset_dir/class_b/*.png. Writes a
    timestamped .keras model, class_names.json and TensorBoard logs.
    """
    import tensorflow as tf
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Dataset directory does not exist.")
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(classes) < 2:
        raise ValueError("Dataset must contain at least two class subfolders with labeled images.")
    image_count = sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if image_count < 4:
        raise ValueError("Add at least four supported images across the class folders before training.")
    if not 1 <= int(epochs) <= 100 or not 1 <= int(batch_size) <= 256:
        raise ValueError("epochs must be 1–100 and batch_size 1–256.")

    seed = 42
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root, validation_split=0.2, subset="training", seed=seed,
        image_size=(int(image_size), int(image_size)), batch_size=int(batch_size), label_mode="int")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root, validation_split=0.2, subset="validation", seed=seed,
        image_size=(int(image_size), int(image_size)), batch_size=int(batch_size), label_mode="int")
    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE
    # Keep preprocessing outside the saved model so predict_image() and the
    # benchmark helper can apply the same MobileNetV2 preprocessing exactly once.
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    train_ds = train_ds.map(lambda images, labels: (preprocess(images), labels), num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(lambda images, labels: (preprocess(images), labels), num_parallel_calls=autotune).prefetch(autotune)

    base = tf.keras.applications.MobileNetV2(input_shape=(int(image_size), int(image_size), 3),
        include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(int(image_size), int(image_size), 3))
    # Random augmentation is active only during training, helping reduce overfitting.
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ], name="training_augmentation")
    x = augmentation(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "logs")),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=int(epochs), callbacks=callbacks, verbose=1)
    model_path = run_dir / "classifier.keras"
    model.save(model_path)
    (run_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    metrics = {key: [float(v) for v in values] for key, values in history.history.items()}
    (run_dir / "history.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "labels_path": str(run_dir / "class_names.json"),
            "run_dir": str(run_dir), "classes": class_names, "image_count": image_count,
            "final_metrics": {key: values[-1] for key, values in metrics.items() if values},
            "history": metrics, "gpu": gpu_status()}
