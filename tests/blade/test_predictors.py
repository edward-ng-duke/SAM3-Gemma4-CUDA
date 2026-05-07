"""Tests for blade predictor adapters."""

from __future__ import annotations

from pathlib import Path

from scripts.blade.datasets import RoboflowYoloLoader
from scripts.blade.predictors import BaselineSAM3VLMPredictor, DummyPredictor, Predictor


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


# ---------------------------------------------------------------------------
# BaselineSAM3VLMPredictor — integration smoke tests with monkeypatched HTTP.
# ---------------------------------------------------------------------------


class _FakeRegion:
    """Minimal stand-in for ``servers_client.DetectedRegion``."""

    def __init__(self, bbox: list, score: float, region_index: int = 0) -> None:
        self.region_index = region_index
        self.bbox = bbox
        self.score = score
        self.mask = None


def _first_record() -> dict:
    records = list(RoboflowYoloLoader(FIXTURE_ROOT))
    assert records, "fixture must contain at least one record"
    return records[0]


def test_baseline_predictor_happy_path(monkeypatch) -> None:
    from scripts.blade import predictors

    monkeypatch.setattr(
        predictors.servers_client,
        "sam3_detect",
        lambda image, prompt, **kw: [_FakeRegion([10, 10, 50, 50], 0.85)],
    )
    monkeypatch.setattr(
        predictors.servers_client,
        "vlm_generate",
        lambda image, prompt, **kw: (
            '{"has_defect": true, "defect_type": "crack", '
            '"severity": "moderate", "evidence": "visible crack near root", '
            '"confidence": 0.82}'
        ),
    )

    predictor = BaselineSAM3VLMPredictor()
    record = _first_record()
    pred = predictor.predict(record["image_path"], record)

    assert set(pred.keys()) == _REQUIRED_KEYS
    assert pred["has_defect"] is True
    assert pred["defect_type"] == "crack"
    assert pred["severity"] == "moderate"
    assert pred["evidence"] == "visible crack near root"
    assert 0.0 <= pred["confidence"] <= 1.0
    # bbox falls back to the SAM3 top region when VLM omits one.
    assert pred["bbox"] is not None
    assert isinstance(pred["raw"], dict)
    assert "sam3_regions" in pred["raw"]
    assert "vlm_raw" in pred["raw"]
    assert "parsed" in pred["raw"]


def test_baseline_predictor_parse_error_fallback(monkeypatch) -> None:
    from scripts.blade import predictors

    monkeypatch.setattr(
        predictors.servers_client, "sam3_detect", lambda *a, **kw: []
    )
    monkeypatch.setattr(
        predictors.servers_client,
        "vlm_generate",
        lambda *a, **kw: "gibberish not json",
    )

    predictor = BaselineSAM3VLMPredictor()
    record = _first_record()
    pred = predictor.predict(record["image_path"], record)

    assert pred["has_defect"] is False
    assert pred["confidence"] == 0.0
    assert "parse_error" in pred["evidence"]
    assert pred["defect_type"] == "none"
    assert pred["severity"] == "none"
    assert pred["bbox"] is None


def test_baseline_predictor_server_error_fallback(monkeypatch) -> None:
    from scripts.blade import predictors
    from servers_client import ServerError

    def boom(*args, **kw):
        raise ServerError("connection refused")

    monkeypatch.setattr(predictors.servers_client, "sam3_detect", boom)

    predictor = BaselineSAM3VLMPredictor()
    record = _first_record()
    pred = predictor.predict(record["image_path"], record)

    assert pred["has_defect"] is False
    assert pred["confidence"] == 0.0
    evidence_lower = pred["evidence"].lower()
    assert "server_error" in evidence_lower or "connection refused" in evidence_lower


def test_baseline_predictor_protocol_compliance() -> None:
    assert isinstance(BaselineSAM3VLMPredictor(), Predictor)
