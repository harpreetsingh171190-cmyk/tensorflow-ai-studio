"""Privacy-conscious local prediction audit log (metadata only; no image bytes)."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_prediction_log(result: dict[str, Any], filename: str, log_path: str = "artifacts/prediction_log.jsonl") -> str:
    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    best = (result.get("predictions") or [{}])[0]
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "filename": Path(filename).name,
           "model": str(result.get("model", "")), "top_label": str(best.get("label", "")),
           "top_confidence": best.get("confidence"), "prediction_count": len(result.get("predictions") or [])}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def read_prediction_log(log_path: str = "artifacts/prediction_log.jsonl", limit: int = 500) -> list[dict[str, Any]]:
    if not 1 <= int(limit) <= 10000:
        raise ValueError("limit must be between 1 and 10000")
    path = Path(log_path).expanduser().resolve()
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except json.JSONDecodeError:
            continue
    return rows[-int(limit):]
