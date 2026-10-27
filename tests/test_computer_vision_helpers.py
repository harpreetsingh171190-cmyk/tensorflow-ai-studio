from pathlib import Path

from modern.computer_vision import IMAGE_EXTENSIONS, gpu_status


def test_supported_image_extensions():
    assert {".jpg", ".jpeg", ".png", ".webp"}.issubset(IMAGE_EXTENSIONS)
    assert ".txt" not in IMAGE_EXTENSIONS


def test_gpu_status_has_stable_shape():
    result = gpu_status()
    assert isinstance(result["available"], bool)
    assert isinstance(result["devices"], list)
    assert "nvidia_smi" in result


def test_dataset_audit_counts_classes_and_extensions(tmp_path):
    from PIL import Image
    from modern.vision_extras import audit_image_dataset

    for label, count in (("cats", 2), ("dogs", 1)):
        folder = tmp_path / label
        folder.mkdir()
        for i in range(count):
            Image.new("RGB", (8, 8), (i * 20, 30, 40)).save(folder / f"{i}.png")
    report = audit_image_dataset(str(tmp_path))
    assert report["class_count"] == 2
    assert report["image_count"] == 3
    assert report["extension_counts"] == {".png": 3}
    assert report["imbalance_ratio"] == 2.0


def test_batch_prediction_keeps_per_file_failures(monkeypatch):
    from modern import vision_extras

    def fake_predict(content, model_path, labels_path, top_k):
        if content == b"bad":
            raise ValueError("invalid image")
        return {"model": "demo", "predictions": [{"label": "cat", "confidence": 0.9}]}

    monkeypatch.setattr(vision_extras, "predict_image", fake_predict)
    result = vision_extras.batch_predict_images([("a.png", b"ok"), ("b.png", b"bad")])
    assert result["processed"] == 2
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["results"][1]["status"] == "error"
