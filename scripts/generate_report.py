import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote as url_quote


def _url_path(rel_path):
    if not rel_path:
        return ""
    parts = rel_path.replace(os.sep, "/").split("/")
    return "/".join(url_quote(p, safe="") for p in parts)


def load_all_jsons(out_dir):
    import sys
    json_dir = os.path.join(out_dir, "json")
    if not os.path.isdir(json_dir):
        return []
    items = []
    for path in Path(json_dir).glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                items.append(json.load(f))
        except Exception as e:
            print(f"warn: skipped {path}: {e}", file=sys.stderr)
            continue
    items.sort(key=lambda r: r.get("image") if isinstance(r, dict) and r.get("image") is not None else "")
    return items


def write_summary(items, out_dir):
    from datetime import datetime
    distribution = {"0": 0, "1": 0, "2": 0, "3": 0, ">=4": 0}
    processed_ok = 0
    errors = 0
    for item in items:
        status = item.get("status")
        if status == "ok":
            processed_ok += 1
            n = item.get("num_persons", 0)
            try:
                n_int = int(n)
            except (TypeError, ValueError):
                n_int = 0
            if n_int <= 0:
                distribution["0"] += 1
            elif n_int == 1:
                distribution["1"] += 1
            elif n_int == 2:
                distribution["2"] += 1
            elif n_int == 3:
                distribution["3"] += 1
            else:
                distribution[">=4"] += 1
        elif status == "error":
            errors += 1

    condensed = [
        {
            "image": item.get("image"),
            "num_persons": item.get("num_persons", -1),
            "status": item.get("status"),
        }
        for item in items
    ]

    summary = {
        "total_images": len(items),
        "processed_ok": processed_ok,
        "errors": errors,
        "person_count_distribution": distribution,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "items": condensed,
    }

    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, "summary.json")
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, final_path)
    return summary


def render_html(items, out_dir):
    from html import escape as html_escape
    from datetime import datetime

    os.makedirs(out_dir, exist_ok=True)

    css = """
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "PingFang SC", sans-serif; margin: 0; padding: 0 24px 40px; background: #f7f7f7; color: #222; }
header { padding: 24px 0 8px; }
header h1 { margin: 0; font-size: 1.6rem; }
header .meta { color: #666; margin: 4px 0 0; font-size: 0.9rem; }
#toolbar { display: flex; gap: 8px; margin: 12px 0 20px; flex-wrap: wrap; position: sticky; top: 0; background: #f7f7f7; padding: 8px 0; z-index: 10; }
#toolbar input, #toolbar select { padding: 6px 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 0.95rem; background: white; }
#toolbar input { flex: 1; min-width: 200px; }
#grid { display: grid; gap: 20px; grid-template-columns: repeat(auto-fill, minmax(720px, 1fr)); }
.card { background: white; border: 2px solid #e5e5e5; border-radius: 10px; padding: 14px 18px; transition: border-color 0.15s; }
.card[open] { border-color: #4682B4; }
.card > summary { display: flex; align-items: center; gap: 10px; cursor: pointer; list-style: none; padding: 4px 0; }
.card > summary::-webkit-details-marker { display: none; }
.filename { flex: 1; font-size: 1rem; font-family: "IBM Plex Mono", ui-monospace, monospace; word-break: break-all; line-height: 1.3; }
.badge { font-weight: 700; padding: 5px 14px; border-radius: 999px; font-size: 0.9rem; flex-shrink: 0; color: white; }
.badge-sam { background: #555; }
.badge-gemma.count-bucket-0 { background: #999; }
.badge-gemma.count-bucket-1 { background: #4682B4; }
.badge-gemma.count-bucket-2 { background: #2e8b57; }
.badge-gemma.count-bucket-3 { background: #e67e22; }
.badge-gemma.count-bucket-ge4 { background: #8e44ad; }
.badge-gemma.count-bucket-err { background: #c0392b; }
.card .body { padding-top: 12px; border-top: 1px solid #eee; margin-top: 10px; }
.tabs { display: flex; gap: 0; margin-bottom: 12px; border-bottom: 2px solid #e5e5e5; }
.tab { background: none; border: none; padding: 8px 18px; cursor: pointer; font-size: 0.95rem; border-bottom: 3px solid transparent; margin-bottom: -2px; font-weight: 600; color: #666; transition: color 0.1s; }
.tab:hover { color: #4682B4; }
.tab.active { color: #4682B4; border-bottom-color: #4682B4; }
.views .view { display: none; margin: 0; }
.views .view.active { display: block; }
.views figcaption { font-size: 0.85rem; color: #666; margin-bottom: 4px; }
.views img { width: 100%; max-height: 85vh; object-fit: contain; border-radius: 6px; border: 1px solid #ddd; background: #fafafa; }
.reasoning { background: #fafafa; border-left: 3px solid #4682B4; padding: 8px 12px; margin: 10px 0; font-size: 0.92rem; line-height: 1.5; white-space: pre-wrap; }
.per-prompt { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin: 10px 0; }
.per-prompt th, .per-prompt td { border: 1px solid #ddd; padding: 4px 6px; text-align: center; }
.per-prompt th { background: #f0f0f0; }
.persons { margin: 8px 0; padding-left: 20px; font-size: 0.9rem; }
.error-msg { background: #fdecea; border-left: 3px solid #c0392b; padding: 8px 12px; margin: 10px 0; font-family: monospace; font-size: 0.9rem; color: #611a1a; }
.json-wrap { margin-top: 10px; }
.json-wrap summary { font-size: 0.85rem; color: #666; cursor: pointer; }
.json-wrap pre { background: #2d2d2d; color: #f0f0f0; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.8rem; margin: 6px 0 0; }
.card[hidden] { display: none; }
"""

    prompt_order = ["person", "face", "head", "hands", "arm", "shoulder", "torso", "legs", "feet"]

    now = datetime.now().isoformat(timespec="seconds")
    n_images = len(items)

    parts_html = []
    parts_html.append("<!doctype html>")
    parts_html.append('<html lang="zh-CN">')
    parts_html.append("<head>")
    parts_html.append('  <meta charset="utf-8">')
    parts_html.append("  <title>SAM3 + Gemma 4 Person Detection Report</title>")
    parts_html.append(f"  <style>{css}</style>")
    parts_html.append("</head>")
    parts_html.append("<body>")
    parts_html.append("  <header>")
    parts_html.append("    <h1>SAM3 + Gemma 4 Person Detection Report</h1>")
    parts_html.append(
        f'    <p class="meta">Generated <span id="generated-at">{html_escape(now)}</span> · {n_images} images</p>'
    )
    parts_html.append("  </header>")
    parts_html.append("")
    parts_html.append('  <div id="toolbar">')
    parts_html.append('    <input id="search" type="search" placeholder="Filename search…">')
    parts_html.append('    <select id="sort">')
    parts_html.append('      <option value="name-asc">Name ↑</option>')
    parts_html.append('      <option value="count-desc">Persons ↓</option>')
    parts_html.append('      <option value="count-asc">Persons ↑</option>')
    parts_html.append("    </select>")
    parts_html.append('    <select id="filter">')
    parts_html.append('      <option value="all">All</option>')
    parts_html.append('      <option value="ok">OK only</option>')
    parts_html.append('      <option value="error">Errors only</option>')
    parts_html.append('      <option value="ge1">Persons ≥ 1</option>')
    parts_html.append('      <option value="ge2">Persons ≥ 2</option>')
    parts_html.append('      <option value="ge3">Persons ≥ 3</option>')
    parts_html.append("    </select>")
    parts_html.append("  </div>")
    parts_html.append("")
    parts_html.append('  <main id="grid">')

    for item in items:
        status = item.get("status") or ""
        try:
            num_persons = int(item.get("num_persons", 0))
        except (TypeError, ValueError):
            num_persons = 0

        image_name = item.get("image") or ""
        image_path = item.get("image_path") or ""
        name_lower = image_name.lower() if isinstance(image_name, str) else ""

        # Relative image path
        if image_path:
            try:
                rel_image = os.path.relpath(image_path, out_dir)
            except Exception:
                rel_image = ""
        else:
            rel_image = ""

        stem = Path(image_name).stem if image_name else ""
        rel_sam = f"visualizations/{stem}_sam.png" if stem else ""
        rel_gemma = f"visualizations/{stem}_gemma.png" if stem else ""

        try:
            sam_person_count = int((item.get("per_prompt_counts") or {}).get("person", 0))
        except (TypeError, ValueError):
            sam_person_count = 0

        # Badge text
        if status == "error":
            badge_text = "ERROR"
        elif status == "ok" and num_persons == 0:
            badge_text = "0"
        elif status == "ok":
            badge_text = str(num_persons)
        else:
            badge_text = str(num_persons)

        # Count bucket
        if status == "error":
            bucket = "err"
        elif num_persons == 0:
            bucket = "0"
        elif num_persons == 1:
            bucket = "1"
        elif num_persons == 2:
            bucket = "2"
        elif num_persons == 3:
            bucket = "3"
        else:
            bucket = "ge4"

        status_attr = html_escape(status, quote=True)
        name_attr = html_escape(name_lower, quote=True)
        count_attr = html_escape(str(num_persons), quote=True)
        rel_image_attr = html_escape(_url_path(rel_image), quote=True)
        rel_sam_attr = html_escape(_url_path(rel_sam), quote=True)
        rel_gemma_attr = html_escape(_url_path(rel_gemma), quote=True)

        parts_html.append(
            f'    <details open class="card status-{html_escape(status, quote=True)} count-bucket-{bucket}" '
            f'data-count="{count_attr}" data-status="{status_attr}" data-name="{name_attr}">'
        )
        parts_html.append("      <summary>")
        parts_html.append(
            f'        <span class="filename">{html_escape(image_name)}</span>'
        )
        if status == "error":
            parts_html.append(
                f'        <span class="badge badge-gemma count-bucket-err">{html_escape(badge_text)}</span>'
            )
        else:
            parts_html.append(
                f'        <span class="badge badge-sam">SAM {sam_person_count}</span>'
            )
            parts_html.append(
                f'        <span class="badge badge-gemma count-bucket-{bucket}">Gemma {html_escape(badge_text)}</span>'
            )
        parts_html.append("      </summary>")
        parts_html.append('      <div class="body">')

        if status == "ok":
            # Tabs: Original | SAM | Gemma
            parts_html.append('        <div class="tabs">')
            parts_html.append('          <button class="tab active" data-tab="orig">原图</button>')
            parts_html.append('          <button class="tab" data-tab="sam">SAM</button>')
            parts_html.append('          <button class="tab" data-tab="gemma">Gemma</button>')
            parts_html.append("        </div>")
            parts_html.append('        <div class="views">')
            parts_html.append(
                f'          <figure class="view view-orig active"><figcaption>原图</figcaption><img src="{rel_image_attr}" loading="lazy"></figure>'
            )
            parts_html.append(
                f'          <figure class="view view-sam"><figcaption>SAM raw — colored by prompt (person=red, face=orange, head=yellow, hands=green, arm=teal, shoulder=blue, torso=indigo, legs=purple, feet=pink)</figcaption><img src="{rel_sam_attr}" loading="lazy"></figure>'
            )
            parts_html.append(
                f'          <figure class="view view-gemma"><figcaption>Gemma grouped — one color per detected person</figcaption><img src="{rel_gemma_attr}" loading="lazy"></figure>'
            )
            parts_html.append("        </div>")

            # Reasoning
            reasoning = item.get("reasoning")
            if reasoning is None or reasoning == "":
                reasoning_text = "(empty)"
            else:
                reasoning_text = str(reasoning)
            parts_html.append(
                f'        <blockquote class="reasoning">{html_escape(reasoning_text)}</blockquote>'
            )

            # Per-prompt counts table
            per_prompt = item.get("per_prompt_counts") or {}
            headers = "".join(f"<th>{html_escape(p)}</th>" for p in prompt_order)
            counts_cells = []
            for p in prompt_order:
                v = per_prompt.get(p, 0) if isinstance(per_prompt, dict) else 0
                counts_cells.append(f"<td>{html_escape(str(v))}</td>")
            counts_row = "".join(counts_cells)
            parts_html.append(
                f'        <table class="per-prompt"><thead><tr>{headers}</tr></thead>'
                f"<tbody><tr>{counts_row}</tr></tbody></table>"
            )

            # Persons list
            persons = item.get("persons") or []
            parts_html.append('        <ul class="persons">')
            if not persons:
                parts_html.append("          <li>(none)</li>")
            else:
                for p in persons:
                    pid = p.get("person_id", "?") if isinstance(p, dict) else "?"
                    raw_parts = p.get("parts", []) if isinstance(p, dict) else []
                    if raw_parts:
                        parts_str = ", ".join(str(x) for x in raw_parts)
                    else:
                        parts_str = "(no parts)"
                    parts_html.append(
                        f"          <li>P{html_escape(str(pid))}: {html_escape(parts_str)}</li>"
                    )
            parts_html.append("        </ul>")
        elif status == "error":
            err_msg = item.get("error_message") or ""
            parts_html.append(
                f'        <div class="error-msg">{html_escape(str(err_msg))}</div>'
            )

        # Always: raw JSON
        try:
            json_pretty = json.dumps(item, indent=2, ensure_ascii=False, default=str)
        except Exception:
            json_pretty = str(item)
        parts_html.append(
            f'        <details class="json-wrap"><summary>Raw JSON</summary><pre>{html_escape(json_pretty)}</pre></details>'
        )

        parts_html.append("      </div>")
        parts_html.append("    </details>")

    parts_html.append("  </main>")
    parts_html.append("""  <script>
(function () {
  const grid = document.getElementById('grid');
  const search = document.getElementById('search');
  const sortSel = document.getElementById('sort');
  const filterSel = document.getElementById('filter');

  function getCards() {
    return Array.from(grid.querySelectorAll('.card'));
  }

  function passes(card, q, f) {
    const name = card.dataset.name || '';
    const count = parseInt(card.dataset.count, 10);
    const status = card.dataset.status;
    if (q && !name.includes(q)) return false;
    if (f === 'ok' && status !== 'ok') return false;
    if (f === 'error' && status !== 'error') return false;
    if (f === 'ge1' && !(count >= 1)) return false;
    if (f === 'ge2' && !(count >= 2)) return false;
    if (f === 'ge3' && !(count >= 3)) return false;
    return true;
  }

  function apply() {
    const q = (search.value || '').trim().toLowerCase();
    const f = filterSel.value;
    const s = sortSel.value;
    const cards = getCards();
    cards.forEach(c => { c.hidden = !passes(c, q, f); });
    const visible = cards.filter(c => !c.hidden);
    const cmp = {
      'name-asc': (a, b) => (a.dataset.name || '').localeCompare(b.dataset.name || ''),
      'count-desc': (a, b) => parseInt(b.dataset.count, 10) - parseInt(a.dataset.count, 10) || (a.dataset.name || '').localeCompare(b.dataset.name || ''),
      'count-asc': (a, b) => parseInt(a.dataset.count, 10) - parseInt(b.dataset.count, 10) || (a.dataset.name || '').localeCompare(b.dataset.name || ''),
    }[s];
    if (cmp) {
      visible.sort(cmp);
      visible.forEach(c => grid.appendChild(c));
    }
  }

  search.addEventListener('input', apply);
  sortSel.addEventListener('change', apply);
  filterSel.addEventListener('change', apply);
  apply();

  // Per-card tab switching: click a .tab to show the matching .view
  document.querySelectorAll('.card .tabs').forEach(function (tabs) {
    tabs.addEventListener('click', function (e) {
      if (!e.target.matches('.tab')) return;
      var which = e.target.dataset.tab;
      var card = tabs.closest('.card');
      card.querySelectorAll('.tab').forEach(function (t) {
        t.classList.toggle('active', t === e.target);
      });
      card.querySelectorAll('.view').forEach(function (v) {
        v.classList.toggle('active', v.classList.contains('view-' + which));
      });
    });
  });
})();
  </script>""")
    parts_html.append("</body>")
    parts_html.append("</html>")

    html = "\n".join(parts_html)

    final_path = os.path.join(out_dir, "report.html")
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp_path, final_path)
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from batch detection JSONs.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory containing json/ subdirectory (same as batch_detect_persons --output-dir)")
    args = parser.parse_args()
    abs_out_dir = os.path.abspath(args.output_dir)
    items = load_all_jsons(abs_out_dir)
    if items == []:
        print(f"No JSONs found in {abs_out_dir}/json/")
        return
    write_summary(items, abs_out_dir)
    render_html(items, abs_out_dir)
    print(f"Wrote {abs_out_dir}/summary.json")
    print(f"Wrote {abs_out_dir}/report.html")


if __name__ == "__main__":
    main()
