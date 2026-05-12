"""Predictor adapters wrapping SAM/Gemma services for blade defect evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, TypedDict, runtime_checkable

import servers_client

from .datasets import BladeRecord
from .prompts import (
    SAM3_PROMPT_BASELINE,
    VLM_SYSTEM_BASELINE,
    build_user_prompt,
)


_VALID_SEVERITIES = {"none", "mild", "moderate", "severe"}


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


def _safe_prediction(
    *,
    evidence: str,
    raw: Optional[dict] = None,
) -> Prediction:
    """Return a fallback Prediction (no defect, zero confidence)."""

    return {
        "has_defect": False,
        "defect_type": "none",
        "severity": "none",
        "evidence": evidence,
        "confidence": 0.0,
        "bbox": None,
        "raw": raw or {},
    }


def _coerce_bbox(value: Any) -> Optional[list]:
    """Coerce a bbox-like value to a 4-element list of numbers, else None."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) != 4:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _clamp_confidence(value: Any) -> float:
    """Coerce ``value`` to a float in ``[0.0, 1.0]``; default 0.0 on failure."""

    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if c != c:  # NaN
        return 0.0
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    return c


class BaselineSAM3VLMPredictor:
    """Single-prompt SAM3 + Qwen VLM JSON-output baseline.

    Pipeline: SAM3 detect (open-vocabulary, prompt = ``"defect"``) → assemble a
    text prompt that lists candidate regions → call the VLM with the image and
    expect a strict JSON object → parse into a :class:`Prediction`.

    All failures (server errors, JSON parse errors, type errors) are caught and
    surfaced as a safe Prediction with ``has_defect=False`` and ``confidence=0.0``;
    the predictor never raises.
    """

    def __init__(self, conf_threshold: float = 0.45, max_new_tokens: int = 768) -> None:
        self.conf_threshold = float(conf_threshold)
        self.max_new_tokens = int(max_new_tokens)

    def predict(self, image_path: Path, record: BladeRecord) -> Prediction:
        """Run SAM3 + VLM and return a Prediction; never raises."""

        # Lazy-import PIL so that bare unit tests (which monkeypatch the
        # downstream HTTP calls) don't pay the import cost when not needed.
        from PIL import Image

        image_id = str(record.get("image_id", Path(image_path).stem))

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:  # noqa: BLE001 — never raise from predict
            return _safe_prediction(evidence=f"image_open_error: {e}")

        # 1. SAM3 detect.
        try:
            regions = servers_client.sam3_detect(
                image,
                SAM3_PROMPT_BASELINE,
                conf_threshold=self.conf_threshold,
            )
        except servers_client.ServerError as e:
            return _safe_prediction(evidence=f"server_error: {e}")
        except Exception as e:  # noqa: BLE001
            return _safe_prediction(evidence=f"server_error: {e}")

        sam3_region_dicts: list[dict] = []
        for r in regions or []:
            sam3_region_dicts.append(
                {
                    "region_index": getattr(r, "region_index", None),
                    "bbox": list(getattr(r, "bbox", []) or []),
                    "score": float(getattr(r, "score", 0.0) or 0.0),
                    "prompt": SAM3_PROMPT_BASELINE,
                }
            )

        # 2. Build the per-image user prompt and full prompt for the VLM.
        user_prompt = build_user_prompt(
            image_id=image_id,
            sam3_regions=sam3_region_dicts,
        )
        full_prompt = VLM_SYSTEM_BASELINE + "\n\n" + user_prompt

        # 3. VLM call — temperature=0.0 for deterministic baseline output.
        try:
            raw_text = servers_client.vlm_generate(
                image,
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
            )
        except servers_client.ServerError as e:
            return _safe_prediction(
                evidence=f"server_error: {e}",
                raw={"sam3_regions": sam3_region_dicts},
            )
        except Exception as e:  # noqa: BLE001
            return _safe_prediction(
                evidence=f"server_error: {e}",
                raw={"sam3_regions": sam3_region_dicts},
            )

        raw_text = raw_text or ""

        # 4. Parse JSON with the project's tolerant parser; empty dict on failure.
        try:
            parsed = servers_client.safe_parse_json(raw_text)
        except Exception as e:  # noqa: BLE001 — defensive
            parsed = {}
            parse_err = str(e)
        else:
            parse_err = None

        if not isinstance(parsed, dict) or not parsed:
            return _safe_prediction(
                evidence=(
                    f"parse_error: {parse_err}"
                    if parse_err
                    else "parse_error: VLM output was not valid JSON"
                ),
                raw={
                    "sam3_regions": sam3_region_dicts,
                    "vlm_raw": raw_text,
                    "parsed": parsed if isinstance(parsed, dict) else {},
                },
            )

        # 5. Coerce / validate fields with safe defaults.
        try:
            has_defect = bool(parsed.get("has_defect", False))
            defect_type_val = parsed.get("defect_type", "none")
            defect_type = (
                str(defect_type_val) if defect_type_val is not None else "none"
            ) or "none"

            severity_val = parsed.get("severity", "none")
            severity = (
                str(severity_val) if severity_val is not None else "none"
            ).lower()
            if severity not in _VALID_SEVERITIES:
                severity = "none"

            evidence_val = parsed.get("evidence", "")
            evidence = str(evidence_val) if evidence_val is not None else ""

            confidence = _clamp_confidence(parsed.get("confidence", 0.0))

            bbox = _coerce_bbox(parsed.get("bbox"))
            if bbox is None and has_defect and sam3_region_dicts:
                # Fall back to the highest-scoring SAM3 region's bbox.
                top = max(sam3_region_dicts, key=lambda r: r.get("score", 0.0))
                bbox = _coerce_bbox(top.get("bbox"))
        except Exception as e:  # noqa: BLE001 — never raise from predict
            return _safe_prediction(
                evidence=f"parse_error: {e}",
                raw={
                    "sam3_regions": sam3_region_dicts,
                    "vlm_raw": raw_text,
                    "parsed": parsed,
                },
            )

        # 6. Coherence: if has_defect=False, force defect_type/severity to "none".
        if not has_defect:
            defect_type = "none"
            severity = "none"

        return {
            "has_defect": has_defect,
            "defect_type": defect_type,
            "severity": severity,
            "evidence": evidence,
            "confidence": confidence,
            "bbox": bbox,
            "raw": {
                "sam3_regions": sam3_region_dicts,
                "vlm_raw": raw_text,
                "parsed": parsed,
            },
        }
