"""Export a Keras History object to CSV for the Streamlit dashboard."""
import argparse
import csv
from pathlib import Path

def export(history_path: str, output_path: str):
    """Convert a JSON file containing history.history (metric -> list) to CSV."""
    import json
    data = json.loads(Path(history_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data or not all(isinstance(v, list) for v in data.values()):
        raise ValueError("Input must be a JSON object mapping metric names to lists of values")
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1:
        raise ValueError("All metric arrays must have the same length")
    keys = list(data)
    with Path(output_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for i in range(next(iter(lengths))):
            writer.writerow([i + 1, *(data[k][i] for k in keys)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("history_json")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    export(args.history_json, args.output_csv)
