"""Tests for scripts.blade.eval_runner.run_eval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.blade.datasets import RoboflowYoloLoader
from scripts.blade.eval_runner import run_eval
from scripts.blade.predictors import DummyPredictor


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_dataset"


def test_run_eval_writes_predictions_and_metrics(tmp_path: Path) -> None:
    loader = RoboflowYoloLoader(FIXTURE_ROOT)
    predictor = DummyPredictor()

    metrics = run_eval(loader, predictor, tmp_path)

    pred_file = tmp_path / "predictions.jsonl"
    metrics_file = tmp_path / "metrics.yaml"
    assert pred_file.is_file()
    assert metrics_file.is_file()

    lines = pred_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    # Each line is a JSON object with the expected keys.
    image_ids = set()
    for line in lines:
        entry = json.loads(line)
        assert set(entry.keys()) >= {"image_id", "ground_truth", "prediction"}
        assert isinstance(entry["ground_truth"], dict)
        assert isinstance(entry["prediction"], dict)
        image_ids.add(entry["image_id"])
    assert image_ids == {"a", "b", "c"}

    # Returned dict matches the on-disk YAML.
    assert isinstance(metrics, dict)
    on_disk = yaml.safe_load(metrics_file.read_text(encoding="utf-8"))
    assert on_disk == metrics

    # Required top-level metric blocks.
    assert "defect_classification" in metrics
    assert "defect_type_classification" in metrics
    assert "localization" in metrics

    # mini_dataset: 1 negative ('a') + 2 positives ('b', 'c'); dummy predicts
    # all-negative -> accuracy = 1/3.
    dc = metrics["defect_classification"]
    assert dc["accuracy"] == pytest.approx(1.0 / 3.0)
    assert "precision" in dc
    assert "recall" in dc
    assert "f1" in dc
    # No true positives in dummy predictions -> precision/recall/F1 all zero.
    assert dc["precision"] == 0.0
    assert dc["recall"] == 0.0
    assert dc["f1"] == 0.0

    # Defect-type confusion matrix should be a square matrix with classes.
    dtc = metrics["defect_type_classification"]
    assert "classes" in dtc
    assert "confusion_matrix" in dtc
    n = len(dtc["classes"])
    assert len(dtc["confusion_matrix"]) == n
    for row in dtc["confusion_matrix"]:
        assert len(row) == n

    # Localization mAP: dummy emits no boxes -> 0.0.
    loc = metrics["localization"]
    assert "map@0.5" in loc
    assert loc["map@0.5"] == 0.0

    # Severity: ground-truth has none -> block omitted or null.
    if "severity" in metrics:
        assert metrics["severity"] is None or metrics["severity"] == {}


def test_run_eval_creates_output_dir(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "out"
    loader = RoboflowYoloLoader(FIXTURE_ROOT)
    metrics = run_eval(loader, DummyPredictor(), nested)
    assert (nested / "predictions.jsonl").is_file()
    assert (nested / "metrics.yaml").is_file()
    assert isinstance(metrics, dict)


def test_render_report_creates_html(tmp_path: Path) -> None:
    """render_report builds a single-page HTML with metrics + failure gallery."""
    from scripts.blade.report import render_report

    loader = RoboflowYoloLoader(FIXTURE_ROOT)
    metrics = run_eval(loader, DummyPredictor(), tmp_path)

    predictions_jsonl = tmp_path / "predictions.jsonl"
    image_root = FIXTURE_ROOT / "images"
    output_html = tmp_path / "report.html"

    returned = render_report(metrics, predictions_jsonl, image_root, output_html)

    assert Path(returned) == output_html
    assert output_html.is_file()
    assert output_html.stat().st_size > 1024
    content = output_html.read_text(encoding="utf-8")
    assert "<table" in content
    assert "Metrics Summary" in content
    # The dataset name (mini_dataset, derived from fixture root) should appear.
    assert "mini_dataset" in content
