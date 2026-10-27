import json

import pytest
from PIL import Image

from modern.vision_ops import build_split_manifest, deployment_readiness


def make_dataset(root, count=10):
    for label in ("cats", "dogs"):
        folder = root / label
        folder.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (8, 8), (i * 10, 20, 30)).save(folder / f"{i}.png")


def test_split_manifest_is_reproducible_and_stratified(tmp_path):
    data = tmp_path / "dataset"
    make_dataset(data)
    a = build_split_manifest(str(data), str(tmp_path / "a.json"), seed=17)
    b = build_split_manifest(str(data), str(tmp_path / "b.json"), seed=17)
    assert a["manifest"] == b["manifest"]
    assert a["sample_count"] == 20
    assert set(r["split"] for r in a["manifest"]) == {"train", "validation", "test"}
    assert all(set(v) == {"train", "validation", "test"} for v in a["class_counts"].values())
    assert json.loads((tmp_path / "a.json").read_text())["seed"] == 17


def test_split_manifest_rejects_bad_ratios(tmp_path):
    data = tmp_path / "dataset"
    make_dataset(data)
    with pytest.raises(ValueError, match="sum to 1"):
        build_split_manifest(str(data), str(tmp_path / "m.json"), .6, .2, .3)


def test_deployment_readiness_checks_model_and_labels(tmp_path):
    model = tmp_path / "model.keras"
    model.write_bytes(b"demo")
    labels = tmp_path / "class_names.json"
    labels.write_text('["cat", "dog"]')
    result = deployment_readiness(str(model), str(labels))
    assert result["passed_checks"] == 3
    assert result["total_checks"] == 5
    assert result["status"] == "review required"
    assert any(c["check"] == "Serving security configured" and not c["passed"] for c in result["checks"])


def test_deployment_readiness_missing_artifact(tmp_path):
    result = deployment_readiness(str(tmp_path / "missing.keras"))
    assert result["passed_checks"] == 0
    assert result["status"] == "review required"
