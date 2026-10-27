"""Extra safe utilities for batch image inference and dataset quality checks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable
import io

from modern.computer_vision import IMAGE_EXTENSIONS, predict_image


def batch_predict_images(files: Iterable[tuple[str, bytes]], model_path: str | None = None,
                         labels_path: str | None = None, top_k: int = 5) -> dict[str, Any]:
    """Predict multiple uploaded images, retaining per-file errors rather than aborting."""
    results: list[dict[str, Any]] = []
    for name, content in files:
        try:
            pred = predict_image(content, model_path, labels_path, top_k)
            best = pred.get("predictions", [{}])[0]
            results.append({"file": str(name), "status": "ok", "model": pred.get("model", ""),
                            "top_label": best.get("label", ""),
                            "top_confidence": best.get("confidence", None),
                            "predictions_json": __import__("json").dumps(pred.get("predictions", []), ensure_ascii=False)})
        except Exception as exc:
            results.append({"file": str(name), "status": "error", "model": "", "top_label": "",
                            "top_confidence": None, "predictions_json": "",
                            "error": f"{type(exc).__name__}: {exc}"})
    return {"processed": len(results), "succeeded": sum(r["status"] == "ok" for r in results),
            "failed": sum(r["status"] != "ok" for r in results), "results": results}


def audit_image_dataset(dataset_dir: str) -> dict[str, Any]:
    """Summarize class folders, file sizes, extensions and unreadable images."""
    from PIL import Image, UnidentifiedImageError

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    classes: list[dict[str, Any]] = []
    unreadable: list[str] = []
    extension_counts: Counter[str] = Counter()
    total_bytes = 0
    for folder in class_dirs:
        images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        for path in images:
            extension_counts[path.suffix.lower()] += 1
            try:
                total_bytes += path.stat().st_size
                with Image.open(path) as im:
                    im.verify()
            except (OSError, UnidentifiedImageError, ValueError):
                unreadable.append(str(path.relative_to(root)))
        classes.append({"class_name": folder.name, "image_count": len(images)})
    counts = [x["image_count"] for x in classes]
    nonzero = [n for n in counts if n > 0]
    total = sum(counts)
    return {"dataset_path": str(root), "class_count": len(classes), "image_count": total,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "classes": classes, "extension_counts": dict(extension_counts),
            "unreadable_images": unreadable,
            "imbalance_ratio": round(max(nonzero) / min(nonzero), 2) if nonzero else None,
            "recommendations": _recommendations(classes, unreadable)}


def _recommendations(classes: list[dict[str, Any]], unreadable: list[str]) -> list[str]:
    counts = [c["image_count"] for c in classes]
    tips: list[str] = []
    if len(classes) < 2:
        tips.append("Add at least two class subfolders before training a classifier.")
    if any(n < 10 for n in counts):
        tips.append("Some classes have fewer than 10 images; gather more examples and validate results carefully.")
    nonzero = [n for n in counts if n]
    if nonzero and max(nonzero) > 3 * min(nonzero):
        tips.append("Class counts are imbalanced; consider collecting more examples for underrepresented classes.")
    if unreadable:
        tips.append("Remove or replace unreadable/corrupt image files before training.")
    if not tips:
        tips.append("Basic checks look reasonable; still inspect labels and use a separate test set.")
    return tips
