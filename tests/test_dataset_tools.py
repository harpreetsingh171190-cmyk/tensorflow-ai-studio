import json
from PIL import Image
from modern.dataset_tools import find_duplicate_images, find_split_leakage
from modern.vision_ops import build_split_manifest


def _img(path, colour):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)


def test_finds_duplicates_and_cross_class_conflicts(tmp_path):
    _img(tmp_path / "cats" / "a.png", (10, 10, 10))
    _img(tmp_path / "cats" / "a_copy.png", (10, 10, 10))
    _img(tmp_path / "dogs" / "same_as_cat.png", (10, 10, 10))
    _img(tmp_path / "dogs" / "unique.png", (200, 0, 0))
    r = find_duplicate_images(str(tmp_path))
    assert r["image_count"] == 4
    assert len(r["duplicate_groups"]) == 1
    group = r["duplicate_groups"][0]
    assert group["count"] == 3 and group["cross_class"] is True
    assert r["redundant_file_count"] == 2 and r["cross_class_groups"] == 1


def test_no_duplicates(tmp_path):
    _img(tmp_path / "a" / "1.png", (1, 2, 3))
    _img(tmp_path / "b" / "2.png", (4, 5, 6))
    r = find_duplicate_images(str(tmp_path))
    assert r["duplicate_groups"] == [] and r["redundant_file_count"] == 0


def test_missing_directory_rejected(tmp_path):
    try:
        find_duplicate_images(str(tmp_path / "nope"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_split_leakage_detected_when_duplicates_cross_splits(tmp_path):
    data = tmp_path / "data"
    for label in ("cats", "dogs"):
        for i in range(6):
            _img(data / label / f"{i}.png", (i * 20, 40 if label == "cats" else 90, 5))
    # make every cat identical so any split must contain copies of the same image
    for i in range(6):
        _img(data / "cats" / f"{i}.png", (1, 1, 1))
    manifest = tmp_path / "m.json"
    build_split_manifest(str(data), str(manifest), seed=3)
    r = find_split_leakage(str(manifest))
    assert r["leaked_file_count"] > 0 and r["missing_files"] == 0
    assert all(len(g["splits"]) > 1 for g in r["leaking_groups"])


def test_split_without_duplicates_has_no_leakage(tmp_path):
    data = tmp_path / "data"
    for label, base in (("cats", 0), ("dogs", 100)):
        for i in range(6):
            _img(data / label / f"{i}.png", (base + i * 10, i, 7))
    manifest = tmp_path / "m.json"
    build_split_manifest(str(data), str(manifest), seed=3)
    r = find_split_leakage(str(manifest))
    assert r["leaking_groups"] == [] and r["checked_files"] == 12


def test_leakage_counts_missing_files(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"manifest": [{"path": str(tmp_path / "gone.png"), "split": "train", "label": "x"}]}))
    r = find_split_leakage(str(manifest))
    assert r["missing_files"] == 1 and r["checked_files"] == 0
