"""Dataset loaders for blade defect evaluation."""

from __future__ import annotations

import re
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


# Image extensions recognised by the Roboflow YOLO loader. Roboflow exports
# typically use jpg/png; we accept the common variants for robustness.
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_class_names(data_yaml: Path) -> list[str]:
    """Read the ``names`` list from a Roboflow ``data.yaml``.

    Prefers PyYAML when available; falls back to a small regex parser for
    the simple ``names: ['a', 'b']`` form Roboflow emits.
    """

    text = data_yaml.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            # YOLO sometimes uses a mapping {0: 'foo', 1: 'bar'}
            names = [names[k] for k in sorted(names)]
        return [str(n) for n in names]
    except ImportError:
        match = re.search(r"^names\s*:\s*\[(.*?)\]", text, flags=re.MULTILINE | re.DOTALL)
        if not match:
            return []
        inner = match.group(1)
        items = re.findall(r"['\"]([^'\"]+)['\"]", inner)
        return list(items)


class RoboflowYoloLoader(BladeDatasetLoader):
    """Loader for datasets exported in Roboflow's YOLO format.

    The directory is expected to contain a ``data.yaml`` with a ``names``
    field, an ``images/`` directory of image files, and a ``labels/``
    directory of YOLO label ``.txt`` files. Each label line follows the
    ``class cx cy w h`` convention with normalised ``[0, 1]`` coordinates;
    boxes are converted to xyxy pixel coordinates using the actual image
    dimensions.

    Images without a paired label file (or with an empty label file) are
    treated as defect-free (``defect_label=False`` / ``bboxes=[]``).

    The blade-defect datasets used by this project are typically single
    class, so ``defect_type`` is taken from the *first* defect class found
    in the file. Multi-class disambiguation is intentionally out of scope
    for this loader.
    """

    def __init__(self, root: Path, source_dataset: Optional[str] = None) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.data_yaml = self.root / "data.yaml"
        self.source_dataset = source_dataset if source_dataset is not None else self.root.name
        self.class_names = _load_class_names(self.data_yaml) if self.data_yaml.exists() else []

    def __iter__(self) -> Iterator[BladeRecord]:
        from PIL import Image  # local import to avoid hard dep at module load

        if not self.images_dir.is_dir():
            return

        image_files = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        for image_path in image_files:
            label_path = self.labels_dir / f"{image_path.stem}.txt"

            with Image.open(image_path) as img:
                width, height = img.size

            bboxes: list[tuple[float, float, float, float]] = []
            class_ids: list[int] = []

            if label_path.is_file():
                for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    x1 = (cx - w / 2.0) * width
                    y1 = (cy - h / 2.0) * height
                    x2 = (cx + w / 2.0) * width
                    y2 = (cy + h / 2.0) * height
                    bboxes.append((x1, y1, x2, y2))
                    class_ids.append(cls)

            defect_label = len(bboxes) > 0
            defect_type: Optional[str] = None
            if defect_label and self.class_names:
                first_cls = class_ids[0]
                if 0 <= first_cls < len(self.class_names):
                    defect_type = self.class_names[first_cls]

            record: BladeRecord = {
                "image_path": image_path,
                "image_id": image_path.stem,
                "defect_label": defect_label,
                "defect_type": defect_type,
                "severity": None,
                "bboxes": bboxes,
                "masks": None,
                "source_dataset": self.source_dataset,
            }
            yield record
