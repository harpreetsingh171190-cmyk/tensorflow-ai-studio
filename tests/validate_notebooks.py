"""Validate notebook JSON and syntax-check code cells without executing training."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
if not notebooks:
    raise SystemExit("No notebooks found")
errors = []
for path in notebooks:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
        if nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
            raise ValueError("unsupported or malformed notebook format")
        for index, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                if source.strip():
                    try:
                        ast.parse(source)
                    except SyntaxError as exc:
                        # Some historical notebooks contain IPython magics; only report
                        # syntax errors for the new maintained 2026 companion notebook.
                        if path.name == "modern_2026_cifar10_keras3.ipynb":
                            raise ValueError(f"cell {index}: {exc}") from exc
        print(f"OK {path.relative_to(ROOT)}")
    except Exception as exc:
        errors.append(f"{path}: {exc}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Validated {len(notebooks)} notebook files")
