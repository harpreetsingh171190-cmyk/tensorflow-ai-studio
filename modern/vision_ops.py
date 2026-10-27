"""Additional reproducibility, inference benchmarking and deployment checks."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from modern.computer_vision import IMAGE_EXTENSIONS


def build_split_manifest(dataset_dir: str, output_path: str = "artifacts/vision/split_manifest.json",
                         train_ratio: float = 0.70, val_ratio: float = 0.15,
                         test_ratio: float = 0.15, seed: int = 42) -> dict[str, Any]:
    """Write a reproducible stratified file manifest without copying images."""
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(not math.isfinite(r) or r <= 0 or r >= 1 for r in ratios):
        raise ValueError("Each split ratio must be between 0 and 1.")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("Train, validation and test ratios must sum to 1.")
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Dataset directory does not exist.")
    classes = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(classes) < 2:
        raise ValueError("Dataset must contain at least two class subfolders.")
    import random
    rng = random.Random(int(seed))
    rows: list[dict[str, str]] = []
    counts: dict[str, dict[str, int]] = {}
    for folder in classes:
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(files) < 3:
            raise ValueError(f"Class '{folder.name}' needs at least 3 images for train/validation/test splits.")
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        buckets = [("train", files[:n_train]), ("validation", files[n_train:n_train+n_val]),
                   ("test", files[n_train+n_val:])]
        counts[folder.name] = {}
        for split, paths in buckets:
            counts[folder.name][split] = len(paths)
            rows.extend({"path": str(p), "relative_path": p.relative_to(root).as_posix(),
                         "label": folder.name, "split": split} for p in paths)
    payload = {"dataset_root": str(root), "seed": int(seed), "ratios": {"train": ratios[0], "validation": ratios[1], "test": ratios[2]},
               "class_counts": counts, "sample_count": len(rows), "manifest": rows,
               "note": "Manifest only; no files copied. Keep related images from the same subject/video in one split to prevent leakage."}
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def deployment_readiness(model_path: str, labels_path: str | None = None) -> dict[str, Any]:
    """Run static artifact/config checks; this is a checklist, not a security certification."""
    model = Path(model_path).expanduser() if model_path else None
    checks: list[dict[str, Any]] = []
    model_exists = bool(model and model.is_file())
    checks.append({"check": "Model artifact exists", "passed": model_exists,
                   "detail": str(model.resolve()) if model_exists else "Choose an existing saved model file."})
    if model_exists:
        size = model.stat().st_size
        checks.append({"check": "Model artifact non-empty", "passed": size > 0,
                       "detail": f"{size / (1024**2):.2f} MiB"})
    else:
        size = None
    label_file = Path(labels_path).expanduser() if labels_path else None
    if labels_path:
        label_exists = bool(label_file and label_file.is_file())
        valid_labels = False
        label_detail = "Label file not found."
        if label_exists:
            try:
                labels = json.loads(label_file.read_text(encoding="utf-8"))
                valid_labels = isinstance(labels, list) and bool(labels) and all(isinstance(x, str) and x.strip() for x in labels)
                label_detail = f"{len(labels)} labels" if valid_labels else "Expected a non-empty JSON array of class-name strings."
            except (OSError, json.JSONDecodeError):
                label_detail = "Could not parse labels as JSON."
        checks.append({"check": "Class labels valid", "passed": valid_labels, "detail": label_detail})
    else:
        checks.append({"check": "Class labels supplied", "passed": False, "detail": "Labels are optional for some models but recommended for custom classifiers."})
    checks.append({"check": "Serving security configured", "passed": False,
                   "detail": "Manual review required: authentication, HTTPS, request limits, secrets and access controls."})
    checks.append({"check": "Target-environment smoke test", "passed": False,
                   "detail": "Run predictions and load tests on the actual target platform before release."})
    passed = sum(1 for c in checks if c["passed"])
    return {"model_path": str(model.resolve()) if model_exists else (str(model) if model else ""),
            "model_size_bytes": size, "checks": checks, "passed_checks": passed,
            "total_checks": len(checks), "readiness_percent": round(100 * passed / len(checks)),
            "status": "review required" if not all(c["passed"] for c in checks) else "checks passed",
            "disclaimer": "Static helper only; not a security audit or guarantee of production readiness."}


def benchmark_keras_model(model_path: str, image_bytes: bytes, repeats: int = 10,
                          warmup: int = 2, image_size: int = 160) -> dict[str, Any]:
    """Benchmark repeated single-image model inference after one model load."""
    if not image_bytes:
        raise ValueError("Upload a readable image to benchmark.")
    if not 1 <= int(repeats) <= 200 or not 0 <= int(warmup) <= 20:
        raise ValueError("Repeats must be 1–200 and warmup must be 0–20.")
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        import io
        import numpy as np
        import tensorflow as tf
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install requirements-vision.txt to benchmark TensorFlow models.") from exc
    model = tf.keras.models.load_model(path)
    shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    h = int(shape[1]) if len(shape) > 2 and shape[1] else int(image_size)
    w = int(shape[2]) if len(shape) > 2 and shape[2] else int(image_size)
    with Image.open(io.BytesIO(image_bytes)) as im:
        arr = np.asarray(im.convert("RGB").resize((w, h)), dtype=np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(arr.copy())
    x = np.expand_dims(x, axis=0)
    for _ in range(int(warmup)):
        model.predict(x, verbose=0)
    durations = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        model.predict(x, verbose=0)
        durations.append((time.perf_counter() - start) * 1000)
    ordered = sorted(durations)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    mean_ms = statistics.mean(durations)
    return {"model_path": str(path.resolve()), "repeats": int(repeats), "warmup_runs": int(warmup),
            "input_size": [h, w], "mean_latency_ms": round(mean_ms, 3),
            "median_latency_ms": round(statistics.median(durations), 3),
            "p95_latency_ms": round(ordered[p95_index], 3),
            "min_latency_ms": round(min(durations), 3), "max_latency_ms": round(max(durations), 3),
            "throughput_images_per_second": round(1000 / mean_ms, 3) if mean_ms > 0 else None,
            "note": "Includes model.predict call overhead but excludes model loading and image decoding. Results vary by hardware, batch size and runtime."}
