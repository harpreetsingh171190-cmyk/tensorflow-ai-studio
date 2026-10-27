"""Lightweight heuristics for prediction-log monitoring; not a formal drift detector."""
from __future__ import annotations
from collections import Counter
from math import isfinite
from typing import Any, Iterable


def summarize_predictions(rows: Iterable[dict[str, Any]], confidence_threshold: float = 0.60) -> dict[str, Any]:
    if not 0 <= float(confidence_threshold) <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    rows = list(rows)
    labels = Counter(str(r.get("top_label", "unknown") or "unknown") for r in rows)
    scores: list[float] = []
    for row in rows:
        try:
            score = float(row.get("top_confidence"))
            if isfinite(score) and 0 <= score <= 1:
                scores.append(score)
        except (TypeError, ValueError):
            continue
    return {
        "sample_count": len(rows),
        "scored_count": len(scores),
        "mean_confidence": round(sum(scores) / len(scores), 4) if scores else None,
        "low_confidence_count": sum(s < confidence_threshold for s in scores),
        "low_confidence_rate": round(sum(s < confidence_threshold for s in scores) / len(scores), 4) if scores else None,
        "label_counts": dict(labels),
        "label_distribution": {k: round(v / len(rows), 4) for k, v in labels.items()} if rows else {},
    }


def compare_prediction_windows(baseline_rows: Iterable[dict[str, Any]], current_rows: Iterable[dict[str, Any]],
                              confidence_threshold: float = 0.60) -> dict[str, Any]:
    baseline = summarize_predictions(baseline_rows, confidence_threshold)
    current = summarize_predictions(current_rows, confidence_threshold)
    bdist, cdist = baseline["label_distribution"], current["label_distribution"]
    labels = sorted(set(bdist) | set(cdist))
    # Total-variation distance (0 = same label mix, 1 = disjoint mix).
    label_mix_distance = round(0.5 * sum(abs(bdist.get(label, 0.0) - cdist.get(label, 0.0)) for label in labels), 4)
    bconf, cconf = baseline["mean_confidence"], current["mean_confidence"]
    confidence_delta = round(cconf - bconf, 4) if bconf is not None and cconf is not None else None
    return {
        "baseline": baseline, "current": current,
        "label_mix_distance": label_mix_distance,
        "confidence_delta": confidence_delta,
        "warning": "Heuristic monitoring only. Label mix changes and confidence shifts do not prove model drift or failure; inspect data and validate with labeled ground truth.",
    }
