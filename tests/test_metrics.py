import pytest
from modern.metrics import (confusion_matrix, classification_metrics, top_k_accuracy,
                            expected_calibration_error, brier_score)


def test_confusion_matrix_layout():
    cm = confusion_matrix(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert cm["labels"] == ["a", "b"]
    assert cm["matrix"] == [[1, 1], [0, 2]]


def test_classification_metrics_hand_computed():
    m = classification_metrics([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["per_class"]["0"]["precision"] == pytest.approx(1 / 2)
    assert m["per_class"]["0"]["recall"] == pytest.approx(1 / 2)
    assert m["per_class"]["1"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["1"]["recall"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert m["weighted_f1"] == pytest.approx(0.5 * 2 / 5 + (2 / 3) * 3 / 5)


def test_unpredicted_class_gives_zero_not_nan():
    m = classification_metrics([0, 1, 2], [0, 1, 1])
    assert m["per_class"]["2"]["precision"] == 0.0
    assert m["per_class"]["2"]["recall"] == 0.0
    assert m["per_class"]["2"]["f1"] == 0.0


def test_top_k_accuracy():
    scores = [[0.1, 0.7, 0.2], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]
    assert top_k_accuracy([1, 1, 0], scores, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy([1, 1, 0], scores, k=2) == pytest.approx(2 / 3)
    assert top_k_accuracy([1, 1, 0], scores, k=3) == pytest.approx(1.0)


def test_perfectly_calibrated_has_zero_ece():
    conf = [0.8] * 10
    correct = [True] * 8 + [False] * 2
    r = expected_calibration_error(conf, correct, n_bins=10)
    assert r["ece"] == pytest.approx(0.0)


def test_overconfident_model_ece():
    r = expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5)
    assert r["ece"] == pytest.approx(0.4)
    assert r["max_calibration_error"] == pytest.approx(0.4)
    assert sum(b["count"] for b in r["bins"]) == 10


def test_confidence_of_exactly_one_lands_in_last_bin():
    r = expected_calibration_error([1.0, 1.0], [True, True], n_bins=5)
    assert r["bins"][-1]["count"] == 2


def test_brier_score():
    assert brier_score([[1.0, 0.0], [0.0, 1.0]], [0, 1]) == pytest.approx(0.0)
    assert brier_score([[0.5, 0.5]], [0]) == pytest.approx(0.5)


@pytest.mark.parametrize("call", [
    lambda: confusion_matrix([], []),
    lambda: confusion_matrix([1, 2], [1]),
    lambda: top_k_accuracy([0], [[0.5, 0.5]], k=3),
    lambda: top_k_accuracy([5], [[0.5, 0.5]], k=1),
    lambda: expected_calibration_error([1.5], [True]),
    lambda: expected_calibration_error([0.5], [True], n_bins=0),
    lambda: brier_score([[0.5, 0.5]], [2]),
])
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
