"""Evaluation runner orchestrating predictors over the blade dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from .datasets import BladeDatasetLoader, BladeRecord
from .metrics import (
    accuracy,
    confusion_matrix,
    map_at_iou,
    ordinal_mae,
    precision_recall_f1,
)
from .predictors import Prediction, Predictor


_SEVERITY_ORDERING = ("none", "mild", "moderate", "severe")


def run_eval(
    loader: BladeDatasetLoader,
    predictor: Predictor,
    output_dir: Path,
    *,
    source_dataset_name: Optional[str] = None,
) -> dict:
    """Run ``predictor`` over every record in ``loader`` and write artefacts.

    Iterates the loader once, calls ``predictor.predict`` on each record,
    accumulates ground-truth/prediction pairs, computes the standard suite
    of blade-defect metrics, and writes:

        - ``output_dir/predictions.jsonl``: one JSON object per record with
          keys ``{image_id, ground_truth, prediction}``.
        - ``output_dir/metrics.yaml``: dict with sub-blocks for
          ``defect_classification``, ``defect_type_classification``,
          ``localization``, and (when severity GT is available) ``severity``.

    Returns the metrics dict.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[BladeRecord] = []
    predictions: list[Prediction] = []

    pred_path = output_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for record in loader:
            prediction = predictor.predict(record["image_path"], record)
            records.append(record)
            predictions.append(prediction)
            entry = {
                "image_id": record["image_id"],
                "ground_truth": dict(record),
                "prediction": dict(prediction),
            }
            fh.write(json.dumps(entry, default=str, ensure_ascii=False))
            fh.write("\n")

    metrics: dict = {}

    # ------------------------------------------------------------------
    # defect_classification (binary)
    # ------------------------------------------------------------------
    y_true_bin = [bool(r["defect_label"]) for r in records]
    y_pred_bin = [bool(p["has_defect"]) for p in predictions]
    acc = accuracy(y_true_bin, y_pred_bin)
    precision, recall, f1 = precision_recall_f1(y_true_bin, y_pred_bin, positive_label=True)
    metrics["defect_classification"] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n": len(records),
    }

    # ------------------------------------------------------------------
    # defect_type_classification (multi-class confusion matrix)
    # ------------------------------------------------------------------
    y_true_type = [(r["defect_type"] or "none") for r in records]
    y_pred_type = [(p["defect_type"] or "none") for p in predictions]
    classes = sorted(set(y_true_type) | set(y_pred_type))
    cm = confusion_matrix(y_true_type, y_pred_type, classes)
    metrics["defect_type_classification"] = {
        "classes": classes,
        "confusion_matrix": cm,
    }

    # ------------------------------------------------------------------
    # severity (ordinal MAE) -- only when at least one record has GT severity
    # ------------------------------------------------------------------
    if any(r["severity"] is not None for r in records):
        y_true_sev = [(r["severity"] or "none") for r in records]
        y_pred_sev = [(p["severity"] or "none") for p in predictions]
        mae = ordinal_mae(y_true_sev, y_pred_sev, ordering=_SEVERITY_ORDERING)
        metrics["severity"] = {
            "mae": mae,
            "ordering": list(_SEVERITY_ORDERING),
        }
    else:
        metrics["severity"] = None

    # ------------------------------------------------------------------
    # localization (mAP @ IoU 0.5)
    # ------------------------------------------------------------------
    loc_preds: list[dict] = []
    for record, prediction in zip(records, predictions):
        bbox = prediction.get("bbox")
        if bbox is None:
            continue
        loc_preds.append({
            "image_id": record["image_id"],
            "bbox": tuple(bbox),
            "score": float(prediction.get("confidence", 0.0)),
        })
    loc_gts: list[dict] = []
    for record in records:
        for bbox in record["bboxes"]:
            loc_gts.append({
                "image_id": record["image_id"],
                "bbox": tuple(bbox),
            })
    map50 = map_at_iou(loc_preds, loc_gts, iou_threshold=0.5)
    metrics["localization"] = {
        "map@0.5": map50,
        "n_pred_boxes": len(loc_preds),
        "n_gt_boxes": len(loc_gts),
    }

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    if source_dataset_name is None and records:
        source_dataset_name = records[0]["source_dataset"]
    metrics["dataset"] = {
        "name": source_dataset_name,
        "n_records": len(records),
    }

    metrics_path = output_dir / "metrics.yaml"
    with metrics_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(metrics, fh, sort_keys=True, allow_unicode=True)

    return metrics
