"""Dataset loaders for blade defect evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Literal, Optional, TypedDict


Severity = Literal["none", "mild", "moderate", "severe"]


class BladeRecord(TypedDict):
    """A single blade image record loaded from a source dataset.

    Fields:
        image_path: Absolute path to the image file on disk.
        image_id: Stable identifier (typically the filename stem).
        defect_label: True if the image is annotated as containing a defect.
        defect_type: Optional defect category string from the source dataset.
        severity: Optional severity bucket (one of Severity).
        bboxes: List of bounding boxes in xyxy pixel coordinates
            (x_min, y_min, x_max, y_max). Empty list when no boxes available.
        masks: Optional list of paths to per-instance mask files; None if
            the source dataset does not provide masks.
        source_dataset: Identifier of the originating dataset (e.g. "DTU-Drone").
    """

    image_path: Path
    image_id: str
    defect_label: bool
    defect_type: Optional[str]
    severity: Optional[Severity]
    bboxes: list[tuple[float, float, float, float]]
    masks: Optional[list[Path]]
    source_dataset: str


class BladeDatasetLoader(ABC):
    """Abstract base class for blade defect dataset loaders.

    Concrete subclasses iterate the source dataset and yield BladeRecord
    instances one at a time.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[BladeRecord]:
        """Yield BladeRecord entries from the underlying dataset."""
        ...
