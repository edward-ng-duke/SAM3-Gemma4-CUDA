"""Reporting utilities (tables, plots, summaries) for blade defect evaluation.

This module renders a single, self-contained HTML page summarising a blade
defect evaluation run. The page is designed to be opened directly from the
filesystem (``file://``) and contains:

    1. A **Metrics Summary** table with accuracy, precision/recall/F1,
       localization mAP@0.5 and (when severity GT is present) ordinal MAE.
    2. A **Defect Type Confusion Matrix** rendered from
       ``metrics['defect_type_classification']['confusion_matrix']``.
    3. An optional **Severity Confusion Matrix**, computed on the fly from
       ``predictions.jsonl`` when ``metrics['severity']`` is not None.
    4. **Failure Gallery** sections showing the Top-N false-positive (GT no
       defect, predicted defect) and false-negative (GT defect, predicted no
       defect) cases as cards with thumbnail, ground truth, prediction and
       evidence.

The styling is intentionally lightweight (inline CSS only, no JavaScript and
no external assets). The colour palette and table aesthetic loosely follow
``scripts/generate_report.py`` so that all blade reports share a consistent
visual identity, but no code is shared with that module.

Public entry point: :func:`render_report`.
"""

from __future__ import annotations

import json
from html import escape as html_escape
from pathlib import Path
from typing import Any, Iterable, Optional


# Default number of failure cases shown per gallery (FP and FN each).
_DEFAULT_TOP_N = 10

# Severity ordering used for the on-the-fly severity confusion matrix.
_SEVERITY_ORDERING = ("none", "mild", "moderate", "severe")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_report(
    metrics: dict,
    predictions_jsonl: Path,
    image_root: Path,
    output_html: Path,
    *,
    top_n: int = _DEFAULT_TOP_N,
) -> Path:
    """Render a single-page HTML evaluation report.

    Args:
        metrics: Metrics dict as produced by ``eval_runner.run_eval``.
        predictions_jsonl: Path to ``predictions.jsonl`` (one JSON record per
            line, each with keys ``image_id``, ``ground_truth``, ``prediction``).
        image_root: Directory used to resolve thumbnail paths when a record's
            ``ground_truth.image_path`` is missing or relative.
        output_html: Destination file path; parent directories are created.
        top_n: Number of failure cases to show per gallery (FP and FN each).

    Returns:
        The ``output_html`` path (absolute, as resolved by ``Path``).

    Notes:
        Thumbnails are emitted as ``<img src="file://{absolute path}">`` so
        that the page works as a standalone local-file viewer. No JavaScript
        is used; the report is pure static HTML + inline CSS.
    """

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(Path(predictions_jsonl))
    dataset_name = _dataset_name(metrics, predictions)

    fp_cases, fn_cases = _split_failures(predictions, top_n=top_n)

    sections: list[str] = []
    sections.append(_render_metrics_summary(metrics))
    sections.append(_render_defect_type_confusion(metrics))
    severity_block = _render_severity_confusion(metrics, predictions)
    if severity_block:
        sections.append(severity_block)
    sections.append(
        _render_failure_gallery(
            "False Positive Gallery (GT no defect, predicted defect)",
            "fp",
            fp_cases,
            Path(image_root),
        )
    )
    sections.append(
        _render_failure_gallery(
            "False Negative Gallery (GT defect, predicted no defect)",
            "fn",
            fn_cases,
            Path(image_root),
        )
    )

    html = _render_page(dataset_name, sections)

    tmp_path = output_html.with_suffix(output_html.suffix + ".tmp")
    tmp_path.write_text(html, encoding="utf-8")
    tmp_path.replace(output_html)
    return output_html


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_predictions(path: Path) -> list[dict]:
    """Load a ``predictions.jsonl`` file into a list of dicts.

    Robust to blank lines and trailing whitespace; lines that fail to parse
    as JSON are skipped silently (the report is best-effort).
    """

    records: list[dict] = []
    if not path.is_file():
        return records
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _dataset_name(metrics: dict, predictions: Iterable[dict]) -> str:
    """Best-effort dataset name for the page title."""

    info = metrics.get("dataset") if isinstance(metrics, dict) else None
    if isinstance(info, dict) and info.get("name"):
        return str(info["name"])
    for entry in predictions:
        gt = entry.get("ground_truth") or {}
        name = gt.get("source_dataset")
        if name:
            return str(name)
    return "unknown"


# ---------------------------------------------------------------------------
# Failure splitting
# ---------------------------------------------------------------------------


def _split_failures(
    predictions: list[dict],
    *,
    top_n: int,
) -> tuple[list[dict], list[dict]]:
    """Partition predictions into FP / FN top lists.

    FP: GT ``defect_label`` is False but prediction ``has_defect`` is True;
        sorted by prediction confidence **descending** (most-confident wrong
        positives first).
    FN: GT ``defect_label`` is True but prediction ``has_defect`` is False;
        sorted by prediction confidence **ascending** (least-confident misses
        first; these are the "barely missed" cases worth eyeballing).
    """

    fp: list[dict] = []
    fn: list[dict] = []
    for entry in predictions:
        gt = entry.get("ground_truth") or {}
        pred = entry.get("prediction") or {}
        gt_label = bool(gt.get("defect_label"))
        pred_label = bool(pred.get("has_defect"))
        if (not gt_label) and pred_label:
            fp.append(entry)
        elif gt_label and (not pred_label):
            fn.append(entry)

    def _conf(entry: dict) -> float:
        try:
            return float((entry.get("prediction") or {}).get("confidence") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    fp.sort(key=_conf, reverse=True)
    fn.sort(key=_conf)
    return fp[:top_n], fn[:top_n]


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_metrics_summary(metrics: dict) -> str:
    """Render the top-level metrics table."""

    rows: list[tuple[str, str]] = []
    dc = metrics.get("defect_classification") or {}
    if dc:
        rows.append(("Accuracy", _fmt_num(dc.get("accuracy"))))
        rows.append(("Precision", _fmt_num(dc.get("precision"))))
        rows.append(("Recall", _fmt_num(dc.get("recall"))))
        rows.append(("F1", _fmt_num(dc.get("f1"))))
        rows.append(("N (records)", _fmt_int(dc.get("n"))))

    loc = metrics.get("localization") or {}
    if loc:
        rows.append(("mAP @ IoU 0.5", _fmt_num(loc.get("map@0.5"))))
        rows.append(("# pred boxes", _fmt_int(loc.get("n_pred_boxes"))))
        rows.append(("# GT boxes", _fmt_int(loc.get("n_gt_boxes"))))

    severity = metrics.get("severity")
    if isinstance(severity, dict) and severity.get("mae") is not None:
        rows.append(("Ordinal MAE (severity)", _fmt_num(severity.get("mae"))))

    body_rows = "".join(
        f"<tr><th>{html_escape(label)}</th><td>{html_escape(value)}</td></tr>"
        for label, value in rows
    )
    return (
        '<section class="card">'
        "<h2>Metrics Summary</h2>"
        '<table class="metrics">'
        "<tbody>"
        f"{body_rows}"
        "</tbody>"
        "</table>"
        "</section>"
    )


def _render_defect_type_confusion(metrics: dict) -> str:
    """Render the defect-type confusion matrix from metrics."""

    block = metrics.get("defect_type_classification") or {}
    classes = list(block.get("classes") or [])
    matrix = block.get("confusion_matrix") or []
    if not classes or not matrix:
        return (
            '<section class="card">'
            "<h2>Defect Type Confusion Matrix</h2>"
            "<p class=\"muted\">No defect-type confusion data available.</p>"
            "</section>"
        )
    return (
        '<section class="card">'
        "<h2>Defect Type Confusion Matrix</h2>"
        f"{_render_confusion_table(classes, matrix)}"
        "</section>"
    )


def _render_severity_confusion(metrics: dict, predictions: list[dict]) -> Optional[str]:
    """Render the severity confusion matrix when severity GT is available.

    Returns ``None`` when ``metrics['severity']`` is None / missing — the
    section is omitted entirely in that case.
    """

    severity = metrics.get("severity")
    if not isinstance(severity, dict):
        return None

    ordering = list(severity.get("ordering") or _SEVERITY_ORDERING)
    classes = list(ordering)
    seen: set[str] = set()
    for entry in predictions:
        gt = (entry.get("ground_truth") or {}).get("severity")
        pred = (entry.get("prediction") or {}).get("severity")
        if gt is not None:
            seen.add(str(gt))
        if pred is not None:
            seen.add(str(pred))
    for cls in sorted(seen):
        if cls not in classes:
            classes.append(cls)

    index = {cls: i for i, cls in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    counted = 0
    for entry in predictions:
        gt = (entry.get("ground_truth") or {}).get("severity")
        if gt is None:
            continue
        pred = (entry.get("prediction") or {}).get("severity") or "none"
        i = index.get(str(gt))
        j = index.get(str(pred))
        if i is None or j is None:
            continue
        matrix[i][j] += 1
        counted += 1

    if counted == 0:
        return None

    return (
        '<section class="card">'
        "<h2>Severity Confusion Matrix</h2>"
        f"{_render_confusion_table(classes, matrix)}"
        "</section>"
    )


def _render_failure_gallery(
    title: str,
    kind: str,
    cases: list[dict],
    image_root: Path,
) -> str:
    """Render a Top-N failure gallery as a grid of cards."""

    if not cases:
        return (
            '<section class="card">'
            f"<h2>{html_escape(title)}</h2>"
            f'<p class="muted">No {html_escape(kind.upper())} cases.</p>'
            "</section>"
        )

    cards = "\n".join(_render_failure_card(c, image_root) for c in cases)
    return (
        '<section class="card">'
        f"<h2>{html_escape(title)}</h2>"
        f'<div class="gallery">{cards}</div>'
        "</section>"
    )


def _render_failure_card(entry: dict, image_root: Path) -> str:
    """Render a single failure case as a small card with thumbnail + details."""

    image_id = str(entry.get("image_id") or "")
    gt = entry.get("ground_truth") or {}
    pred = entry.get("prediction") or {}

    img_src = _thumbnail_src(image_id, gt, image_root)
    img_html = (
        f'<img src="{html_escape(img_src, quote=True)}" alt="{html_escape(image_id)}" '
        'style="max-width:160px;max-height:120px;object-fit:contain;'
        'border:1px solid #ddd;background:#fafafa;">'
    ) if img_src else (
        '<div class="no-thumb">no image</div>'
    )

    gt_rows = [
        ("defect_label", _fmt_bool(gt.get("defect_label"))),
        ("defect_type", _fmt_str(gt.get("defect_type"))),
        ("severity", _fmt_str(gt.get("severity"))),
        ("# bboxes", str(len(gt.get("bboxes") or []))),
    ]
    pred_rows = [
        ("has_defect", _fmt_bool(pred.get("has_defect"))),
        ("defect_type", _fmt_str(pred.get("defect_type"))),
        ("severity", _fmt_str(pred.get("severity"))),
        ("confidence", _fmt_num(pred.get("confidence"))),
    ]

    gt_html = "".join(
        f"<tr><th>{html_escape(k)}</th><td>{html_escape(v)}</td></tr>"
        for k, v in gt_rows
    )
    pred_html = "".join(
        f"<tr><th>{html_escape(k)}</th><td>{html_escape(v)}</td></tr>"
        for k, v in pred_rows
    )
    evidence = pred.get("evidence") or ""
    evidence_html = (
        f'<div class="evidence"><span class="lbl">evidence:</span> '
        f"{html_escape(str(evidence))}</div>"
    ) if evidence else ""

    return (
        '<div class="case">'
        f'<div class="case-head"><span class="case-id">{html_escape(image_id)}</span></div>'
        f'<div class="thumb">{img_html}</div>'
        '<div class="case-body">'
        '<table class="kv"><caption>Ground Truth</caption><tbody>'
        f"{gt_html}"
        "</tbody></table>"
        '<table class="kv"><caption>Prediction</caption><tbody>'
        f"{pred_html}"
        "</tbody></table>"
        f"{evidence_html}"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _render_confusion_table(classes: list[str], matrix: list[list[int]]) -> str:
    """Render a square confusion matrix as a styled HTML table.

    Rows are GT classes, columns are predicted classes (Y axis = truth, X
    axis = prediction). The top-left header cell labels the orientation.
    """

    header_cells = "".join(f"<th>{html_escape(str(c))}</th>" for c in classes)
    rows: list[str] = []
    for i, cls in enumerate(classes):
        row = matrix[i] if i < len(matrix) else []
        cells: list[str] = []
        for j, _ in enumerate(classes):
            v = row[j] if j < len(row) else 0
            css_class = "diag" if i == j else ""
            cells.append(
                f'<td class="{css_class}">{html_escape(str(v))}</td>'
            )
        rows.append(
            f'<tr><th class="row-label">{html_escape(str(cls))}</th>'
            f"{''.join(cells)}</tr>"
        )
    return (
        '<table class="confusion">'
        '<thead><tr><th class="corner">truth \\\\ pred</th>'
        f"{header_cells}</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table>"
    )


def _thumbnail_src(image_id: str, gt: dict, image_root: Path) -> str:
    """Resolve an absolute ``file://`` URL for a record's thumbnail.

    Tries (in order): the GT ``image_path`` field; ``image_root / image_id``
    with common extensions. Returns ``""`` when no candidate exists on disk.
    """

    candidates: list[Path] = []
    raw_path = gt.get("image_path")
    if raw_path:
        candidates.append(Path(str(raw_path)))
    if image_id:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
            candidates.append(image_root / f"{image_id}{ext}")
        candidates.append(image_root / image_id)
    for cand in candidates:
        try:
            if cand.is_file():
                return f"file://{cand.resolve()}"
        except OSError:
            continue
    return ""


def _fmt_num(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _fmt_bool(value: Any) -> str:
    if value is None:
        return "—"
    return "True" if bool(value) else "False"


def _fmt_str(value: Any) -> str:
    if value is None:
        return "—"
    s = str(value)
    return s if s else "—"


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------


_CSS = """
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, "PingFang SC", "Helvetica Neue", sans-serif;
  margin: 0;
  padding: 0 24px 48px;
  background: #f7f7f7;
  color: #222;
}
header { padding: 24px 0 8px; }
header h1 { margin: 0; font-size: 1.6rem; color: #2f3e4d; }
header .meta { color: #666; margin: 4px 0 0; font-size: 0.9rem; }
.card {
  background: white;
  border: 1px solid #e5e5e5;
  border-left: 4px solid #4682B4;
  border-radius: 8px;
  padding: 16px 20px;
  margin: 16px 0;
}
.card h2 { margin: 0 0 12px; font-size: 1.15rem; color: #2f3e4d; }
.muted { color: #888; font-style: italic; }
table { border-collapse: collapse; }
table.metrics { width: auto; min-width: 320px; font-size: 0.95rem; }
table.metrics th, table.metrics td {
  border: 1px solid #e5e5e5;
  padding: 6px 12px;
  text-align: left;
}
table.metrics th {
  background: #f0f4f8;
  font-weight: 600;
  width: 200px;
  color: #2f3e4d;
}
table.confusion { font-size: 0.9rem; margin-top: 4px; }
table.confusion th, table.confusion td {
  border: 1px solid #ddd;
  padding: 5px 10px;
  text-align: center;
  min-width: 48px;
}
table.confusion thead th { background: #4682B4; color: white; }
table.confusion th.corner { background: #2f3e4d; font-size: 0.8rem; }
table.confusion th.row-label { background: #f0f4f8; color: #2f3e4d; }
table.confusion td.diag { background: #e8f1f8; font-weight: 600; }
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 12px;
  margin-top: 6px;
}
.case {
  border: 1px solid #e5e5e5;
  border-radius: 6px;
  padding: 10px;
  background: #fcfcfc;
}
.case-head {
  font-family: ui-monospace, "IBM Plex Mono", monospace;
  font-size: 0.85rem;
  color: #444;
  margin-bottom: 6px;
  word-break: break-all;
}
.case .thumb {
  display: flex;
  justify-content: center;
  margin-bottom: 8px;
  min-height: 60px;
}
.case .no-thumb {
  color: #aaa;
  font-style: italic;
  padding: 28px 0;
}
.case-body table.kv {
  width: 100%;
  font-size: 0.82rem;
  margin: 4px 0;
}
.case-body table.kv caption {
  caption-side: top;
  text-align: left;
  font-weight: 600;
  color: #2f3e4d;
  font-size: 0.78rem;
  margin-bottom: 2px;
}
.case-body table.kv th,
.case-body table.kv td {
  border: 1px solid #eee;
  padding: 3px 6px;
  text-align: left;
}
.case-body table.kv th {
  background: #f7f9fc;
  width: 110px;
  font-weight: 500;
  color: #555;
}
.case-body .evidence {
  font-size: 0.82rem;
  background: #fafafa;
  border-left: 3px solid #4682B4;
  padding: 6px 8px;
  margin-top: 6px;
  white-space: pre-wrap;
  word-break: break-word;
}
.case-body .evidence .lbl { color: #666; font-weight: 600; }
"""


def _render_page(dataset_name: str, sections: list[str]) -> str:
    """Stitch the final HTML document together."""

    title = f"Blade Defect Eval Report — {dataset_name}"
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html_escape(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>{html_escape(title)}</h1>",
        '<p class="meta">Generated by scripts.blade.report.render_report — '
        "open this file directly in a browser; thumbnails reference local "
        "<code>file://</code> paths.</p>",
        "</header>",
        "<main>",
        *sections,
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)
