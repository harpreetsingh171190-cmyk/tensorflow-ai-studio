import json
import pytest
from modern.feedback_queue import add_feedback, read_feedback


def test_feedback_queue_stores_correction_without_image_bytes(tmp_path):
    path = tmp_path / "feedback.jsonl"
    add_feedback(filename="folder/cat.png", model="demo.keras", predicted_label="dog",
                 confidence=0.72, corrected_label="cat", notes="manual review", queue_path=str(path))
    rows = read_feedback(str(path))
    assert len(rows) == 1
    assert rows[0]["filename"] == "cat.png"
    assert rows[0]["is_correct"] is False
    assert "image_bytes" not in rows[0]


def test_feedback_rejects_empty_correction(tmp_path):
    with pytest.raises(ValueError):
        add_feedback(filename="x.png", model="m", predicted_label="cat", confidence=0.4,
                     corrected_label=" ", queue_path=str(tmp_path / "f.jsonl"))
