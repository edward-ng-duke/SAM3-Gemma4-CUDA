"""Tests for blade predictor adapters."""

from __future__ import annotations

from pathlib import Path

from scripts.blade.datasets import RoboflowYoloLoader
from scripts.blade.predictors import DummyPredictor, Predictor


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_dataset"


_REQUIRED_KEYS = {
    "has_defect",
    "defect_type",
    "severity",
    "evidence",
    "confidence",
    "bbox",
    "raw",
}


def test_dummy_predictor_returns_valid_schema() -> None:
    records = list(RoboflowYoloLoader(FIXTURE_ROOT))
    assert records, "fixture must contain at least one record"
    record = records[0]

    predictor = DummyPredictor()
    pred = predictor.predict(record["image_path"], record)

    # All required keys present, no extras.
    assert set(pred.keys()) == _REQUIRED_KEYS

    # Type checks per Prediction TypedDict.
    assert isinstance(pred["has_defect"], bool)
    assert isinstance(pred["defect_type"], str)
    assert isinstance(pred["severity"], str)
    assert isinstance(pred["evidence"], str)
    assert isinstance(pred["confidence"], float)
    assert pred["bbox"] is None or isinstance(pred["bbox"], (list, tuple))
    assert isinstance(pred["raw"], dict)

    # Confidence within valid [0.0, 1.0] range.
    assert 0.0 <= pred["confidence"] <= 1.0


def test_dummy_predictor_protocol_compliance() -> None:
    # Predictor is a runtime_checkable Protocol; structural compliance suffices.
    assert isinstance(DummyPredictor(), Predictor)


def test_dummy_predictor_always_no_defect() -> None:
    records = list(RoboflowYoloLoader(FIXTURE_ROOT))
    assert len(records) >= 2, "need multiple records to exercise this test"

    predictor = DummyPredictor()
    for record in records:
        pred = predictor.predict(record["image_path"], record)
        assert pred["has_defect"] is False
        assert pred["confidence"] == 0.0
        assert pred["defect_type"] == "none"
        assert pred["severity"] == "none"
        assert pred["bbox"] is None
