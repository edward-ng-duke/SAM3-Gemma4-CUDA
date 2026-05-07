"""CLI entry point for the blade defect evaluation pipeline.

Usage:
    python -m scripts.blade --dataset PATH --predictor NAME --out PATH

This module is intentionally light at import time: only stdlib (``argparse``,
``pathlib``, ``sys``) is pulled in eagerly. Heavy imports (PyYAML, PIL,
predictor implementations, the runner and report renderer) are deferred to
``main()`` so that ``-h``/``--help`` stays snappy and import failures show
up only when actually running an evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the blade eval CLI."""

    parser = argparse.ArgumentParser(
        prog="python -m scripts.blade",
        description=(
            "Run the blade defect evaluation pipeline: load a dataset, "
            "run a predictor over every record, compute metrics, and render "
            "an HTML report."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the dataset root (Roboflow YOLO format expected).",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="dummy",
        help="Predictor name registered in the CLI registry (default: dummy).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for predictions.jsonl, metrics.yaml, report.html.",
    )
    parser.add_argument(
        "--source-dataset-name",
        type=str,
        default=None,
        help="Override the source dataset name recorded in metrics.yaml.",
    )
    return parser


def _predictor_registry() -> dict[str, Callable[[], object]]:
    """Return the mapping of predictor name -> zero-arg factory.

    Factories are used (rather than instances) so that constructing a
    predictor only happens for the one actually selected on the CLI.
    """

    from .predictors import BaselineSAM3VLMPredictor, DummyPredictor

    return {
        "dummy": lambda: DummyPredictor(),
        "baseline": lambda: BaselineSAM3VLMPredictor(),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    registry = _predictor_registry()
    if args.predictor not in registry:
        known = ", ".join(sorted(registry))
        parser.error(f"unknown predictor '{args.predictor}'. Known: {known}")

    # Lazy imports — keep CLI startup snappy.
    from .datasets import RoboflowYoloLoader
    from .eval_runner import run_eval
    from .report import render_report

    dataset_root = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = RoboflowYoloLoader(dataset_root)
    predictor = registry[args.predictor]()

    metrics = run_eval(
        loader,
        predictor,
        out_dir,
        source_dataset_name=args.source_dataset_name,
    )

    predictions_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.yaml"
    report_path = out_dir / "report.html"
    image_root = dataset_root / "images"

    render_report(
        metrics,
        predictions_path,
        image_root,
        report_path,
    )

    dataset_info = metrics.get("dataset") or {}
    dc = metrics.get("defect_classification") or {}
    print("blade eval complete:")
    print(f"  dataset:       {dataset_info.get('name')}")
    print(f"  records:       {dataset_info.get('n_records')}")
    print(f"  predictor:     {args.predictor}")
    print(f"  accuracy:      {dc.get('accuracy')}")
    print(f"  metrics:       {metrics_path}")
    print(f"  predictions:   {predictions_path}")
    print(f"  report:        {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
