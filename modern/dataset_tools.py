"""Exact-duplicate and train/test-leakage checks for image datasets (standard library only).

Only byte-identical files are detected. Near-duplicates (resized, re-compressed or cropped
copies) need perceptual hashing, which is not implemented here.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from modern.computer_vision import IMAGE_EXTENSIONS


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_images(dataset_dir: str) -> dict[str, Any]:
    """Group byte-identical images under `dataset_dir/<class>/...`.

    `cross_class_groups` counts identical files filed under different classes, which means
    contradictory labels for the same picture.
    """
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            by_hash[_sha256(path)].append(path)
    groups = []
    for digest, paths in by_hash.items():
        if len(paths) < 2:
            continue
        rel = [p.relative_to(root).as_posix() for p in paths]
        labels = sorted({r.split("/")[0] for r in rel if "/" in r})
        groups.append({"sha256": digest, "count": len(paths), "files": rel,
                       "labels": labels, "cross_class": len(labels) > 1})
    groups.sort(key=lambda g: (-g["count"], g["files"][0]))
    return {"dataset_path": str(root), "image_count": sum(len(v) for v in by_hash.values()),
            "duplicate_groups": groups,
            "redundant_file_count": sum(g["count"] - 1 for g in groups),
            "cross_class_groups": sum(g["cross_class"] for g in groups),
            "note": "Exact (byte-identical) duplicates only; near-duplicates are not detected."}


def find_split_leakage(manifest_path: str) -> dict[str, Any]:
    """Find identical images that a split manifest placed in different splits.

    `manifest_path` is the JSON written by `modern.vision_ops.build_split_manifest`.
    """
    manifest_file = Path(manifest_path).expanduser()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    rows = data.get("manifest") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a JSON object with a non-empty 'manifest' list.")
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing = 0
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            missing += 1
            continue
        by_hash[_sha256(path)].append({"path": row["path"], "split": row["split"], "label": row.get("label", "")})
    leaks = []
    for digest, items in by_hash.items():
        splits = sorted({i["split"] for i in items})
        if len(splits) > 1:
            leaks.append({"sha256": digest, "splits": splits, "files": items})
    return {"checked_files": sum(len(v) for v in by_hash.values()), "missing_files": missing,
            "leaking_groups": leaks, "leaked_file_count": sum(len(g["files"]) for g in leaks),
            "note": "Exact duplicates only. Also group frames from the same video/subject manually."}
