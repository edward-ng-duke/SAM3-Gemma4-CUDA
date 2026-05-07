"""Tests for blade dataset loaders."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.blade.datasets import BladeDatasetLoader, RoboflowYoloLoader


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_dataset"


def test_loader_is_blade_dataset_loader_subclass() -> None:
    assert issubclass(RoboflowYoloLoader, BladeDatasetLoader)


def test_loader_yields_three_records_sorted_by_image_id() -> None:
    loader = RoboflowYoloLoader(FIXTURE_ROOT)
    records = list(loader)
    assert len(records) == 3
    assert [r["image_id"] for r in records] == ["a", "b", "c"]


def test_record_a_has_no_defect() -> None:
    records = list(RoboflowYoloLoader(FIXTURE_ROOT))
    a = records[0]
    assert a["image_id"] == "a"
    assert a["defect_label"] is False
    assert a["bboxes"] == []
    assert a["defect_type"] is None
    assert a["severity"] is None
    assert a["masks"] is None


def test_record_b_has_one_bbox_in_pixel_xyxy() -> None:
    records = list(RoboflowYoloLoader(FIXTURE_ROOT))
    b = records[1]
    assert b["image_id"] == "b"
    assert b["defect_label"] is True
    assert len(b["bboxes"]) == 1
    assert b["defect_type"] == "defect"
    assert b["severity"] is None
    # Image is 64x64; line is "0 0.5 0.5 0.3 0.3"
    # cx=32, cy=32, w=19.2, h=19.2 → x1=22.4, y1=22.4, x2=41.6, y2=41.6
    x1, y1, x2, y2 = b["bboxes"][0]
    assert math.isclose(x1, 22.4, abs_tol=1e-6)
    assert math.isclose(y1, 22.4, abs_tol=1e-6)
    assert math.isclose(x2, 41.6, abs_tol=1e-6)
    assert math.isclose(y2, 41.6, abs_tol=1e-6)


def test_record_c_has_two_bboxes() -> None:
    records = list(RoboflowYoloLoader(FIXTURE_ROOT))
    c = records[2]
    assert c["image_id"] == "c"
    assert c["defect_label"] is True
    assert len(c["bboxes"]) == 2
    assert c["defect_type"] == "defect"
    # First box "0 0.3 0.3 0.2 0.2": cx=19.2, cy=19.2, w=12.8, h=12.8
    # x1=12.8, y1=12.8, x2=25.6, y2=25.6
    x1, y1, x2, y2 = c["bboxes"][0]
    assert math.isclose(x1, 12.8, abs_tol=1e-6)
    assert math.isclose(y1, 12.8, abs_tol=1e-6)
    assert math.isclose(x2, 25.6, abs_tol=1e-6)
    assert math.isclose(y2, 25.6, abs_tol=1e-6)


def test_source_dataset_defaults_to_root_name() -> None:
    records = list(RoboflowYoloLoader(FIXTURE_ROOT))
    for r in records:
        assert r["source_dataset"] == "mini_dataset"


def test_source_dataset_can_be_overridden() -> None:
    records = list(RoboflowYoloLoader(FIXTURE_ROOT, source_dataset="custom"))
    for r in records:
        assert r["source_dataset"] == "custom"


def test_image_paths_are_absolute_or_resolvable() -> None:
    records = list(RoboflowYoloLoader(FIXTURE_ROOT))
    for r in records:
        assert isinstance(r["image_path"], Path)
        assert r["image_path"].exists()
