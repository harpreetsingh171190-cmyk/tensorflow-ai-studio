import pytest
from modern.model_monitoring import summarize_predictions, compare_prediction_windows


def test_monitoring_summary_and_low_confidence_rate():
    result = summarize_predictions([
        {"top_label": "cat", "top_confidence": 0.9},
        {"top_label": "dog", "top_confidence": 0.2},
    ], confidence_threshold=0.5)
    assert result["sample_count"] == 2
    assert result["low_confidence_count"] == 1
    assert result["label_counts"] == {"cat": 1, "dog": 1}


def test_compare_windows_flags_distribution_change_heuristically():
    result = compare_prediction_windows(
        [{"top_label": "cat", "top_confidence": 0.9}] * 4,
        [{"top_label": "dog", "top_confidence": 0.4}] * 4,
    )
    assert result["label_mix_distance"] == 1.0
    assert result["confidence_delta"] == -0.5
    assert "do not prove" in result["warning"]


def test_threshold_validation():
    with pytest.raises(ValueError):
        summarize_predictions([], confidence_threshold=2)
