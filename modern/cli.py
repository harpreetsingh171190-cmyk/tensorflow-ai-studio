"""Unified command line for the dependency-light MLOps and data tools.

    python -m modern profile data.csv
    python -m modern drift reference.csv current.csv
    python -m modern split dataset/ --seed 7
    python -m modern duplicates dataset/
    python -m modern leakage artifacts/vision/split_manifest.json
    python -m modern deploy-check artifacts/model.keras --labels class_names.json
    python -m modern monitor --log artifacts/prediction_log.jsonl
    python -m modern experiments --best accuracy --mode max
    python -m modern metrics predictions.csv --true-col label --pred-col prediction

Every command prints JSON. None of them needs TensorFlow.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Sequence


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m modern", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="profile a CSV file")
    p.add_argument("csv")

    p = sub.add_parser("drift", help="compare numeric columns of two CSV files")
    p.add_argument("reference_csv")
    p.add_argument("current_csv")
    p.add_argument("--bins", type=int, default=10)

    p = sub.add_parser("split", help="write a reproducible train/validation/test manifest")
    p.add_argument("dataset_dir")
    p.add_argument("--output", default="artifacts/vision/split_manifest.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)

    p = sub.add_parser("duplicates", help="find byte-identical images in a class-folder dataset")
    p.add_argument("dataset_dir")

    p = sub.add_parser("leakage", help="find identical images placed in different splits")
    p.add_argument("manifest")

    p = sub.add_parser("deploy-check", help="static deployment-readiness checklist")
    p.add_argument("model")
    p.add_argument("--labels")

    p = sub.add_parser("monitor", help="summarise a prediction log (confidence and label mix)")
    p.add_argument("--log", default="artifacts/prediction_log.jsonl")
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--limit", type=int, default=1000)

    p = sub.add_parser("experiments", help="list logged experiments or show the best run")
    p.add_argument("--path", default="experiments/runs.jsonl")
    p.add_argument("--best", metavar="METRIC")
    p.add_argument("--mode", choices=["max", "min"], default="max")

    p = sub.add_parser("metrics", help="classification metrics from a CSV of true and predicted labels")
    p.add_argument("csv")
    p.add_argument("--true-col", default="y_true")
    p.add_argument("--pred-col", default="y_pred")
    return parser


def _read_label_columns(path: str, true_col: str, pred_col: str) -> tuple[list[str], list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for col in (true_col, pred_col):
            if col not in fields:
                raise ValueError(f"Column '{col}' not found. Available columns: {', '.join(fields) or 'none'}")
        rows = list(reader)
    return [r[true_col] for r in rows], [r[pred_col] for r in rows]


def run(args: argparse.Namespace):
    # Imports are local so `--help` stays fast.
    if args.command == "profile":
        from modern.data_quality import profile_csv
        return profile_csv(args.csv)
    if args.command == "drift":
        from modern.data_quality import compare_numeric_drift
        return compare_numeric_drift(args.reference_csv, args.current_csv, args.bins)
    if args.command == "split":
        from modern.vision_ops import build_split_manifest
        result = build_split_manifest(args.dataset_dir, args.output, args.train, args.val, args.test, args.seed)
        result["manifest_file"] = args.output
        del result["manifest"]  # keep terminal output short; the full list is in the file
        return result
    if args.command == "duplicates":
        from modern.dataset_tools import find_duplicate_images
        return find_duplicate_images(args.dataset_dir)
    if args.command == "leakage":
        from modern.dataset_tools import find_split_leakage
        return find_split_leakage(args.manifest)
    if args.command == "deploy-check":
        from modern.vision_ops import deployment_readiness
        return deployment_readiness(args.model, args.labels)
    if args.command == "monitor":
        from modern.model_monitoring import summarize_predictions
        from modern.prediction_log import read_prediction_log
        return summarize_predictions(read_prediction_log(args.log, args.limit), args.threshold)
    if args.command == "experiments":
        from modern.experiment_tracker import best_experiment, read_experiments
        if args.best:
            return best_experiment(args.best, args.mode, args.path)
        return read_experiments(args.path)
    if args.command == "metrics":
        from modern.metrics import classification_metrics
        truth, predicted = _read_label_columns(args.csv, args.true_col, args.pred_col)
        return classification_metrics(truth, predicted)
    raise ValueError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _print(run(args))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
