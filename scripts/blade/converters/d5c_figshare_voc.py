"""Convert the D5c Figshare WTBD Multiclass dataset (PASCAL VOC) to Roboflow YOLO.

Source dataset
--------------
Article: https://springernature.figshare.com/articles/dataset/30210175
License: CC BY 4.0

Download (no auth required)::

    curl -sL -o wt_blade_defect.zip \
        https://ndownloader.figshare.com/files/61029058
    unzip wt_blade_defect.zip

The unpacked tree looks like::

    WT blade defect dataset/
        JPEGImages/<stem>.jpg     (1024x1024 typical)
        Annotations/<stem>.xml    (PASCAL VOC, single or multi <object>)

There are 1,065 paired image/annotation files across 6 defect classes:
``corrosion``, ``crack``, ``craze``, ``hide_craze``, ``surface_injure``,
``thunderstrike``.

This script writes a Roboflow YOLO directory::

    <out>/
        images/<stem>.jpg
        labels/<stem>.txt   # ``cls cx cy w h`` normalised
        data.yaml           # ``nc`` + ``names``

The label class indices map to the alphabetised list of class names found
in the source XMLs; the same ordering is recorded in ``data.yaml``.

Sampling
--------
``--limit N`` keeps only the first N images (sorted by filename) so a
local baseline run (T14) can finish in a few minutes without touching the
full 1,065-image set. The default is no limit.

Usage::

    python -m scripts.blade.converters.d5c_figshare_voc \
        --src "data/blade_eval/d5c/WT blade defect dataset" \
        --dst data/blade_eval/d5c_yolo \
        --limit 80
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


def _natural_key(p: Path) -> tuple:
    """Sort key that orders ``9.jpg`` before ``10.jpg``."""

    stem = p.stem
    try:
        return (0, int(stem), stem)
    except ValueError:
        return (1, 0, stem)


def collect_class_names(xml_paths: Iterable[Path]) -> list[str]:
    """Return the sorted list of unique ``<name>`` values across XMLs."""

    names: set[str] = set()
    for xml in xml_paths:
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        for obj in root.findall("object"):
            n = obj.findtext("name")
            if n:
                names.add(n.strip())
    return sorted(names)


def voc_to_yolo_lines(
    xml_path: Path,
    class_index: dict[str, int],
) -> list[str]:
    """Read a PASCAL VOC XML and return YOLO label lines.

    Returns an empty list when the file has no usable boxes (the image is
    then treated as a negative sample by ``RoboflowYoloLoader``).
    """

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    size = root.find("size")
    if size is None:
        return []
    try:
        w = float(size.findtext("width") or 0)
        h = float(size.findtext("height") or 0)
    except (TypeError, ValueError):
        return []
    if w <= 0 or h <= 0:
        return []

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if not name or name not in class_index:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin") or 0)
            ymin = float(bnd.findtext("ymin") or 0)
            xmax = float(bnd.findtext("xmax") or 0)
            ymax = float(bnd.findtext("ymax") or 0)
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2.0) / w
        cy = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        # Clamp to [0, 1] to defend against off-by-one boxes that touch
        # the image edge in the source data.
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        cls = class_index[name]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def write_data_yaml(dst: Path, names: list[str]) -> None:
    """Write a minimal Roboflow-style ``data.yaml`` (``nc`` + ``names``)."""

    quoted = ", ".join(f"'{n}'" for n in names)
    content = (
        f"nc: {len(names)}\n"
        f"names: [{quoted}]\n"
    )
    (dst / "data.yaml").write_text(content, encoding="utf-8")


def convert(src: Path, dst: Path, limit: int | None = None) -> dict:
    """Perform the conversion. Returns a small stats dict for logging."""

    images_src = src / "JPEGImages"
    annot_src = src / "Annotations"
    if not images_src.is_dir() or not annot_src.is_dir():
        raise SystemExit(
            f"expected JPEGImages/ and Annotations/ under {src}"
        )

    image_files = sorted(
        (p for p in images_src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=_natural_key,
    )
    if limit is not None and limit > 0:
        image_files = image_files[:limit]

    # Build the class list from the *selected* annotations so single-class
    # slices don't carry empty class entries.
    paired_xmls = [annot_src / f"{p.stem}.xml" for p in image_files]
    paired_xmls = [x for x in paired_xmls if x.is_file()]
    class_names = collect_class_names(paired_xmls)
    class_index = {n: i for i, n in enumerate(class_names)}

    images_dst = dst / "images"
    labels_dst = dst / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    n_pos = 0
    n_neg = 0
    for img in image_files:
        xml = annot_src / f"{img.stem}.xml"
        if not xml.is_file():
            continue
        shutil.copy2(img, images_dst / img.name)
        lines = voc_to_yolo_lines(xml, class_index)
        (labels_dst / f"{img.stem}.txt").write_text(
            ("\n".join(lines) + ("\n" if lines else "")),
            encoding="utf-8",
        )
        if lines:
            n_pos += 1
        else:
            n_neg += 1

    write_data_yaml(dst, class_names)

    return {
        "n_images": n_pos + n_neg,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_classes": len(class_names),
        "class_names": class_names,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.blade.converters.d5c_figshare_voc",
        description="Convert D5c Figshare WTBD (PASCAL VOC) -> Roboflow YOLO.",
    )
    parser.add_argument("--src", type=Path, required=True, help="Source root containing JPEGImages/ and Annotations/.")
    parser.add_argument("--dst", type=Path, required=True, help="Destination Roboflow YOLO root.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of images.")
    args = parser.parse_args(argv)

    stats = convert(args.src, args.dst, limit=args.limit)
    print("d5c_figshare_voc -> yolo conversion complete:")
    print(f"  src:        {args.src}")
    print(f"  dst:        {args.dst}")
    print(f"  images:     {stats['n_images']} (positive={stats['n_positive']}, negative={stats['n_negative']})")
    print(f"  classes:    {stats['n_classes']}")
    print(f"  class_names:{stats['class_names']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
