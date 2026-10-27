"""Human-in-the-loop correction queue for image predictions; stores metadata only."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def add_feedback(*, filename: str, model: str, predicted_label: str, confidence: float | None,
                 corrected_label: str, notes: str = "", queue_path: str = "artifacts/feedback_queue.jsonl") -> str:
    corrected_label = str(corrected_label).strip()
    predicted_label = str(predicted_label).strip()
    if not corrected_label:
        raise ValueError("Corrected label must not be empty")
    if not predicted_label:
        raise ValueError("Predicted label must not be empty")
    if confidence is not None and not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be between 0 and 1")
    path = Path(queue_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "filename": Path(str(filename)).name,
        "model": str(model),
        "predicted_label": predicted_label,
        "confidence": float(confidence) if confidence is not None else None,
        "corrected_label": corrected_label,
        "is_correct": predicted_label.casefold() == corrected_label.casefold(),
        "notes": str(notes)[:1000],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def read_feedback(queue_path: str = "artifacts/feedback_queue.jsonl", limit: int = 1000) -> list[dict[str, Any]]:
    if not 1 <= int(limit) <= 10000:
        raise ValueError("limit must be between 1 and 10000")
    path = Path(queue_path).expanduser().resolve()
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        except json.JSONDecodeError:
            continue
    return rows[-int(limit):]
