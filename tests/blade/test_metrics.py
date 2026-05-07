"""TDD tests for scripts.blade.metrics — written before implementation.

Covers: accuracy, precision_recall_f1, confusion_matrix, iou, map_at_iou, ordinal_mae.
"""

from __future__ import annotations

import math

import pytest

from scripts.blade.metrics import (
    accuracy,
    confusion_matrix,
    iou,
    map_at_iou,
    ordinal_mae,
    precision_recall_f1,
)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------
class TestAccuracy:
    def test_all_correct(self):
        assert accuracy([1, 0, 1, 1], [1, 0, 1, 1]) == 1.0

    def test_all_wrong(self):
        assert accuracy([1, 0, 1], [0, 1, 0]) == 0.0

    def test_partial(self):
        # 2/4 correct
        assert accuracy(["a", "b", "c", "d"], ["a", "b", "x", "y"]) == 0.5

    def test_empty_inputs_return_zero(self):
        assert accuracy([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            accuracy([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# precision_recall_f1
# ---------------------------------------------------------------------------
class TestPrecisionRecallF1:
    def test_perfect_classification(self):
        p, r, f = precision_recall_f1([True, False, True], [True, False, True])
        assert p == 1.0 and r == 1.0 and f == 1.0

    def test_zero_tp(self):
        p, r, f = precision_recall_f1([True, True], [False, False])
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_empty_inputs(self):
        assert precision_recall_f1([], []) == (0.0, 0.0, 0.0)

    def test_known_values(self):
        # y_true: T T T F F
        # y_pred: T F T T F
        # TP=2, FP=1, FN=1
        # precision = 2/3, recall = 2/3, f1 = 2/3
        p, r, f = precision_recall_f1(
            [True, True, True, False, False],
            [True, False, True, True, False],
        )
        assert math.isclose(p, 2 / 3)
        assert math.isclose(r, 2 / 3)
        assert math.isclose(f, 2 / 3)

    def test_negative_label_positive(self):
        # treat False as positive label
        p, r, f = precision_recall_f1(
            [True, False, False],
            [True, False, True],
            positive_label=False,
        )
        # TP (False/False) = 1, FP (False predicted but True actual) = 0,
        # FN (True predicted but False actual) = 1 (index 2)
        assert math.isclose(p, 1.0)
        assert math.isclose(r, 0.5)
        assert math.isclose(f, 2 * 1.0 * 0.5 / (1.0 + 0.5))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            precision_recall_f1([True], [True, False])


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------
class TestConfusionMatrix:
    def test_basic_3x3(self):
        classes = ["a", "b", "c"]
        y_true = ["a", "a", "b", "c", "c", "c"]
        y_pred = ["a", "b", "b", "c", "a", "c"]
        # rows = true, cols = pred ordered by classes
        # row a: [1, 1, 0]
        # row b: [0, 1, 0]
        # row c: [1, 0, 2]
        assert confusion_matrix(y_true, y_pred, classes) == [
            [1, 1, 0],
            [0, 1, 0],
            [1, 0, 2],
        ]

    def test_diagonal_all_correct(self):
        classes = ["x", "y"]
        m = confusion_matrix(["x", "y", "x"], ["x", "y", "x"], classes)
        assert m == [[2, 0], [0, 1]]

    def test_empty_inputs(self):
        m = confusion_matrix([], [], ["a", "b"])
        assert m == [[0, 0], [0, 0]]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            confusion_matrix(["a"], ["a", "b"], ["a", "b"])

    def test_unknown_class_raises(self):
        with pytest.raises(ValueError):
            confusion_matrix(["a", "z"], ["a", "a"], ["a", "b"])


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------
class TestIoU:
    def test_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert iou((0, 0, 5, 5), (10, 10, 20, 20)) == 0.0

    def test_partial_overlap_known(self):
        # box_a area = 100, box_b area = 100
        # intersection: x in [5,10], y in [5,10] -> 25
        # union: 100 + 100 - 25 = 175
        # iou = 25/175
        result = iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert math.isclose(result, 25 / 175)

    def test_box_inside_another(self):
        # smaller box fully inside larger one
        # intersection = 4, union = 100, iou = 0.04
        result = iou((0, 0, 10, 10), (1, 1, 3, 3))
        assert math.isclose(result, 4 / 100)

    def test_touching_boxes_zero(self):
        # boxes share an edge but no area
        assert iou((0, 0, 5, 5), (5, 0, 10, 5)) == 0.0

    def test_zero_area_box(self):
        # degenerate box should return 0
        assert iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# map_at_iou
# ---------------------------------------------------------------------------
class TestMapAtIoU:
    def test_empty_predictions(self):
        gts = [{"image_id": "a", "bbox": (0.0, 0.0, 10.0, 10.0)}]
        assert map_at_iou([], gts, iou_threshold=0.5) == 0.0

    def test_empty_gts(self):
        preds = [{"image_id": "a", "bbox": (0.0, 0.0, 10.0, 10.0), "score": 0.9}]
        assert map_at_iou(preds, [], iou_threshold=0.5) == 0.0

    def test_perfect_match(self):
        preds = [
            {"image_id": "a", "bbox": (0.0, 0.0, 10.0, 10.0), "score": 0.9},
            {"image_id": "b", "bbox": (5.0, 5.0, 15.0, 15.0), "score": 0.8},
        ]
        gts = [
            {"image_id": "a", "bbox": (0.0, 0.0, 10.0, 10.0)},
            {"image_id": "b", "bbox": (5.0, 5.0, 15.0, 15.0)},
        ]
        assert map_at_iou(preds, gts, iou_threshold=0.5) == 1.0

    def test_one_tp_one_fp(self):
        # 1 TP, 1 FP, 1 GT -> precision/recall curve gives ~1.0 at recall 1.0
        preds = [
            {"image_id": "a", "bbox": (0.0, 0.0, 10.0, 10.0), "score": 0.9},  # TP
            {"image_id": "a", "bbox": (50.0, 50.0, 60.0, 60.0), "score": 0.8},  # FP
        ]
        gts = [{"image_id": "a", "bbox": (0.0, 0.0, 10.0, 10.0)}]
        ap = map_at_iou(preds, gts, iou_threshold=0.5)
        # Sorted by score desc: TP first (precision 1, recall 1), then FP (precision 0.5, recall 1)
        # AP via all-points = 1.0 (max precision at recall>=1.0 is 1.0)
        # AP via 11-point = 1.0 (same reason)
        assert math.isclose(ap, 1.0)

    def test_no_matches_below_threshold(self):
        preds = [{"image_id": "a", "bbox": (0.0, 0.0, 10.0, 10.0), "score": 0.9}]
        gts = [{"image_id": "a", "bbox": (100.0, 100.0, 110.0, 110.0)}]
        assert map_at_iou(preds, gts, iou_threshold=0.5) == 0.0


# ---------------------------------------------------------------------------
# ordinal_mae
# ---------------------------------------------------------------------------
class TestOrdinalMae:
    def test_identical_returns_zero(self):
        assert ordinal_mae(["mild", "severe"], ["mild", "severe"]) == 0.0

    def test_max_distance(self):
        # 'severe' vs 'none' -> distance 3
        assert ordinal_mae(["severe"], ["none"]) == 3.0

    def test_average_distances(self):
        # |0-1| + |1-2| + |2-3| = 3 / 3 = 1.0
        y_true = ["none", "mild", "moderate"]
        y_pred = ["mild", "moderate", "severe"]
        assert ordinal_mae(y_true, y_pred) == 1.0

    def test_custom_ordering(self):
        ordering = ["low", "med", "high"]
        # |0-2| = 2, |1-1| = 0 -> avg 1.0
        assert ordinal_mae(["low", "med"], ["high", "med"], ordering=ordering) == 1.0

    def test_empty_returns_zero(self):
        assert ordinal_mae([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            ordinal_mae(["mild"], ["mild", "severe"])

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError):
            ordinal_mae(["mild", "extreme"], ["mild", "severe"])
