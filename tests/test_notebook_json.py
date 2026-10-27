"""Fast structural checks that do not require TensorFlow to be installed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_all_notebooks_parse():
    notebooks = list((ROOT / "notebooks").rglob("*.ipynb"))
    assert notebooks, "No notebooks found"
    for path in notebooks:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("nbformat") == 4, path
        assert isinstance(data.get("cells"), list), path
        for cell in data["cells"]:
            assert cell.get("cell_type") in {"markdown", "code", "raw"}, path
            assert isinstance(cell.get("source"), list), path
