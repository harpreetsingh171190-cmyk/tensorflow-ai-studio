"""NumPy-only classification metrics and calibration diagnostics.

These helpers work on plain arrays (for example a CSV of true/predicted labels or a
prediction log), so they run without TensorFlow or scikit-learn.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _label_arrays(y_true: Sequence, y_pred: Sequence) -> tuple[np.ndarray, np.ndarray]:
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if t.ndim != 1 or p.ndim != 1 or t.shape != p.shape or t.size == 0:
        raise ValueError("y_true and y_pred must be non-empty 1-D sequences of the same length.")
    return t, p


def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> dict[str, Any]:
    """Rows = true label, columns = predicted label. Values not in `labels` are ignored."""
    t, p = _label_arrays(y_true, y_pred)
    if labels is None:
        labels = sorted(set(t.tolist()) | set(p.tolist()))
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(t.tolist(), p.tolist()):
        if actual in index and predicted in index:
            matrix[index[actual], index[predicted]] += 1
    return {"labels": labels, "matrix": matrix.tolist()}


def classification_metrics(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> dict[str, Any]:
    """Accuracy plus per-class and averaged precision / recall / F1 (zero-division gives 0.0)."""
    cm = confusion_matrix(y_true, y_pred, labels)
    labels, m = cm["labels"], np.asarray(cm["matrix"], dtype=float)
    true_pos = np.diag(m)
    support = m.sum(axis=1)
    predicted = m.sum(axis=0)
    precision = np.divide(true_pos, predicted, out=np.zeros_like(true_pos), where=predicted > 0)
    recall = np.divide(true_pos, support, out=np.zeros_like(true_pos), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    total = support.sum()
    per_class = {str(label): {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                 for i, label in enumerate(labels)}
    weights = support / total if total else support
    return {
        "accuracy": float(true_pos.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * weights).sum()),
        "per_class": per_class,
        "confusion_matrix": cm,
        "sample_count": int(total),
    }


def top_k_accuracy(y_true: Sequence[int], scores: Sequence[Sequence[float]], k: int = 3) -> float:
    """Fraction of samples whose true class index is among the k highest scores."""
    s = np.asarray(scores, dtype=float)
    t = np.asarray(y_true)
    if s.ndim != 2 or t.ndim != 1 or len(t) != len(s) or len(t) == 0:
        raise ValueError("scores must be 2-D (samples x classes) and y_true 1-D with matching length.")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite.")
    if not 1 <= int(k) <= s.shape[1]:
        raise ValueError("k must be between 1 and the number of classes.")
    if ((t < 0) | (t >= s.shape[1])).any():
        raise ValueError("y_true contains class indices outside the score columns.")
    top = np.argsort(-s, axis=1)[:, : int(k)]
    return float((top == t[:, None]).any(axis=1).mean())


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 10) -> dict[str, Any]:
    """Expected calibration error with equal-width confidence bins.

    ECE is the sample-weighted mean gap between a bin's accuracy and its mean confidence.
    0 means perfectly calibrated; the value depends on the number of bins and sample size.
    """
    c = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=float)
    if c.ndim != 1 or c.shape != ok.shape or c.size == 0:
        raise ValueError("confidences and correct must be non-empty 1-D sequences of equal length.")
    if not np.isfinite(c).all() or ((c < 0) | (c > 1)).any():
        raise ValueError("confidences must be finite values between 0 and 1.")
    if int(n_bins) < 1:
        raise ValueError("n_bins must be at least 1.")
    n_bins = int(n_bins)
    idx = np.minimum((c * n_bins).astype(int), n_bins - 1)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        row = {"lower": b / n_bins, "upper": (b + 1) / n_bins, "count": count,
               "accuracy": None, "mean_confidence": None}
        if count:
            acc, conf = float(ok[mask].mean()), float(c[mask].mean())
            gap = abs(acc - conf)
            ece += gap * count / c.size
            mce = max(mce, gap)
            row.update(accuracy=acc, mean_confidence=conf)
        bins.append(row)
    return {"ece": float(ece), "max_calibration_error": float(mce), "bins": bins, "sample_count": int(c.size)}


def brier_score(probabilities: Sequence[Sequence[float]], y_true: Sequence[int]) -> float:
    """Multi-class Brier score (lower is better): mean squared distance to the one-hot truth."""
    p = np.asarray(probabilities, dtype=float)
    t = np.asarray(y_true)
    if p.ndim != 2 or t.ndim != 1 or len(t) != len(p) or len(t) == 0:
        raise ValueError("probabilities must be 2-D and y_true 1-D with matching length.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1.")
    if ((t < 0) | (t >= p.shape[1])).any():
        raise ValueError("y_true contains class indices outside the probability columns.")
    onehot = np.eye(p.shape[1])[t.astype(int)]
    return float(((p - onehot) ** 2).sum(axis=1).mean())
