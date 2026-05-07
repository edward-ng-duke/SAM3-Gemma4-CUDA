"""Metrics computation (precision, recall, IoU, mAP) for blade defect evaluation.

Pure-Python / numpy implementations. No model or dataset dependencies.

Functions
---------
- accuracy(y_true, y_pred)
- precision_recall_f1(y_true, y_pred, positive_label=True) -> (P, R, F1)
- confusion_matrix(y_true, y_pred, classes) -> rows=true, cols=pred
- iou(box_a, box_b) -> float in [0, 1]; boxes are xyxy tuples
- map_at_iou(preds, gts, iou_threshold=0.5) -> float (single-class AP, all-points)
- ordinal_mae(y_true, y_pred, ordering=['none','mild','moderate','severe'])
"""

from __future__ import annotations

from typing import Sequence


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------
def accuracy(y_true: list, y_pred: list) -> float:
    """Fraction of equal pairs.

    Empty inputs -> 0.0. Mismatched lengths -> ValueError.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}"
        )
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


# ---------------------------------------------------------------------------
# precision / recall / f1 (binary)
# ---------------------------------------------------------------------------
def precision_recall_f1(
    y_true: list,
    y_pred: list,
    positive_label=True,
) -> tuple[float, float, float]:
    """Binary precision, recall, F1 for the given positive label.

    Empty inputs or no TP -> (0.0, 0.0, 0.0). Mismatched lengths -> ValueError.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}"
        )
    if len(y_true) == 0:
        return (0.0, 0.0, 0.0)

    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = t == positive_label
        p_pos = p == positive_label
        if p_pos and t_pos:
            tp += 1
        elif p_pos and not t_pos:
            fp += 1
        elif (not p_pos) and t_pos:
            fn += 1

    if tp == 0:
        return (0.0, 0.0, 0.0)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


# ---------------------------------------------------------------------------
# confusion matrix
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: list, y_pred: list, classes: list) -> list[list[int]]:
    """Confusion matrix as list-of-lists; rows = true, cols = pred, ordered by ``classes``.

    Mismatched lengths -> ValueError. Unknown class label -> ValueError.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}"
        )
    index = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t not in index:
            raise ValueError(f"y_true contains unknown class: {t!r}")
        if p not in index:
            raise ValueError(f"y_pred contains unknown class: {p!r}")
        matrix[index[t]][index[p]] += 1
    return matrix


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------
def iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """IoU of two xyxy boxes. Degenerate (zero-area) boxes -> 0.0."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0 or area_b <= 0:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# mAP @ IoU (single class, all-points AP)
# ---------------------------------------------------------------------------
def map_at_iou(
    preds: list[dict],
    gts: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Single-class average precision at the given IoU threshold.

    Schema:
      preds: list of {"image_id": str, "bbox": (x1,y1,x2,y2), "score": float}
      gts:   list of {"image_id": str, "bbox": (x1,y1,x2,y2)}

    AP is computed via the **all-points** PR curve (Pascal VOC 2010+) where
    precision at each recall level is replaced by the maximum precision at
    any recall >= that level, then integrated as a step sum.

    Empty preds or empty gts -> 0.0.
    """
    if len(preds) == 0 or len(gts) == 0:
        return 0.0

    # Group GT boxes by image_id and track which have been matched.
    gts_by_image: dict[str, list[tuple[tuple[float, float, float, float], bool]]] = {}
    total_gts = 0
    for gt in gts:
        img = gt["image_id"]
        gts_by_image.setdefault(img, []).append([gt["bbox"], False])  # type: ignore[arg-type]
        total_gts += 1

    if total_gts == 0:
        return 0.0

    # Sort predictions by score descending.
    preds_sorted = sorted(preds, key=lambda p: p["score"], reverse=True)

    tps = [0] * len(preds_sorted)
    fps = [0] * len(preds_sorted)

    for i, pred in enumerate(preds_sorted):
        img = pred["image_id"]
        candidates = gts_by_image.get(img, [])
        best_iou = 0.0
        best_j = -1
        for j, entry in enumerate(candidates):
            gt_bbox, matched = entry
            if matched:
                continue
            cur = iou(pred["bbox"], gt_bbox)
            if cur > best_iou:
                best_iou = cur
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            tps[i] = 1
            candidates[best_j][1] = True  # mark matched
        else:
            fps[i] = 1

    # Cumulative TP/FP -> precision/recall arrays.
    cum_tp = 0
    cum_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for i in range(len(preds_sorted)):
        cum_tp += tps[i]
        cum_fp += fps[i]
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / total_gts)

    # All-points interpolation: at each unique recall step, use max precision
    # for any recall >= that step. Compute as area under interpolated curve.
    # First, post-process precisions so they are monotonically non-increasing
    # when traversed from the end (Pascal VOC style):
    interp_p = list(precisions)
    for i in range(len(interp_p) - 2, -1, -1):
        if interp_p[i + 1] > interp_p[i]:
            interp_p[i] = interp_p[i + 1]

    # Integrate: sum over recall increments * precision at that step.
    ap = 0.0
    prev_recall = 0.0
    for r, p in zip(recalls, interp_p):
        delta = r - prev_recall
        if delta > 0:
            ap += delta * p
            prev_recall = r
    return ap


# ---------------------------------------------------------------------------
# ordinal MAE
# ---------------------------------------------------------------------------
def ordinal_mae(
    y_true: list,
    y_pred: list,
    ordering: Sequence[str] = ("none", "mild", "moderate", "severe"),
) -> float:
    """Mean absolute error of label indices in an ordered scale.

    Empty inputs -> 0.0. Mismatched lengths -> ValueError.
    Unknown labels -> ValueError.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}"
        )
    if len(y_true) == 0:
        return 0.0
    index = {label: i for i, label in enumerate(ordering)}
    total = 0.0
    for t, p in zip(y_true, y_pred):
        if t not in index:
            raise ValueError(f"y_true contains unknown label: {t!r}")
        if p not in index:
            raise ValueError(f"y_pred contains unknown label: {p!r}")
        total += abs(index[t] - index[p])
    return total / len(y_true)
