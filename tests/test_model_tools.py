from modern.prediction_log import append_prediction_log, read_prediction_log


def test_prediction_log_does_not_store_image_bytes(tmp_path):
    path = tmp_path / "audit.jsonl"
    result = {"model": "demo.keras", "predictions": [{"label": "cat", "confidence": 0.91}]}
    append_prediction_log(result, "private-photo.png", str(path))
    rows = read_prediction_log(str(path))
    assert len(rows) == 1
    assert rows[0]["filename"] == "private-photo.png"
    assert rows[0]["top_label"] == "cat"
    assert "image_bytes" not in rows[0]


def test_prediction_log_skips_invalid_lines_and_limits(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"top_label":"dog"}\nnot-json\n{"top_label":"cat"}\n', encoding="utf-8")
    assert [r["top_label"] for r in read_prediction_log(str(path), limit=2)] == ["dog", "cat"]
