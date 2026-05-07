"""Predictor adapters wrapping SAM/Gemma services for blade defect evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, TypedDict, runtime_checkable

from .datasets import BladeRecord


class Prediction(TypedDict):
    """A single defect prediction produced for one BladeRecord.

    Fields:
        has_defect: Predicted binary label (True if defect detected).
        defect_type: Predicted defect category string ("none" if no defect).
        severity: Predicted severity bucket
            (one of "none" | "mild" | "moderate" | "severe").
        evidence: Free-form textual rationale from the underlying model.
        confidence: Model confidence score in [0.0, 1.0].
        bbox: Optional predicted bounding box in xyxy pixel coordinates,
            or None when no localization is available.
        raw: Raw provider response payload, retained for debugging/audit.
    """

    has_defect: bool
    defect_type: str
    severity: str
    evidence: str
    confidence: float
    bbox: Optional[list]
    raw: dict


@runtime_checkable
class Predictor(Protocol):
    """Structural protocol for blade defect predictors.

    Any object exposing a compatible `predict` method satisfies this protocol;
    concrete implementations live elsewhere.
    """

    def predict(self, image_path: Path, record: BladeRecord) -> Prediction:
        """Run prediction on a single image and return a Prediction dict."""
        ...


class DummyPredictor:
    """Baseline predictor that always reports no defect.

    Useful as a sanity-check predictor for the evaluation pipeline: every
    image is predicted as defect-free with zero confidence, so downstream
    metrics reflect only the dataset's class prior.
    """

    def predict(self, image_path: Path, record: BladeRecord) -> Prediction:
        """Return a constant Prediction indicating no defect detected."""
        return {
            "has_defect": False,
            "defect_type": "none",
            "severity": "none",
            "evidence": "dummy predictor: always reports no defect",
            "confidence": 0.0,
            "bbox": None,
            "raw": {},
        }
