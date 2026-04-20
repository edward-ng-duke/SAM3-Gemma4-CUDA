import argparse
import json
import os
from pathlib import Path


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
    raise NotImplementedError("T11 will implement this")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from batch detection JSONs.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory containing json/ subdirectory (same as batch_detect_persons --output-dir)")
    args = parser.parse_args()
    raise NotImplementedError("T13 will wire this up")


if __name__ == "__main__":
    main()
