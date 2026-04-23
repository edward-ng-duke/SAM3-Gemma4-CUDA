"""
人数分类器（SAM3-only，单人/多人二分类）。

规则：`person ≥ 2` → 多人，否则单人。只看 SAM 的 `person` prompt 输出数量，
这是经过实测选出的最干净的信号（face/head 几乎不加新信息；hands/arm/legs/feet
会被 SAM 过分割成假阳性）。

用法：
    python scripts/classify_persons.py \
        --label 正向 --json-dir <dir> \
        --label 反向 --json-dir <dir> \
        --out <doc.md>
"""

import argparse
import json
import os
from collections import Counter


def classify(item):
    """Return ('single'|'multi', reason)."""
    if item.get("status") != "ok":
        return "single", f"status={item.get('status')}"
    ppc = item.get("per_prompt_counts") or {}
    person = ppc.get("person", 0)
    if person >= 2:
        return "multi", f"SAM person={person}"
    return "single", f"SAM person={person}"


def load_jsons(json_dir):
    items = []
    for name in sorted(os.listdir(json_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(json_dir, name), "r", encoding="utf-8") as f:
            items.append(json.load(f))
    return items


def classify_batch(items):
    rows = []
    for it in items:
        verdict, reason = classify(it)
        rows.append((it.get("image", "?"), verdict, reason, it))
    return rows


def person_distribution(items):
    c = Counter()
    for it in items:
        if it.get("status") != "ok":
            continue
        ppc = it.get("per_prompt_counts") or {}
        c[ppc.get("person", 0)] += 1
    return c


def render_doc(datasets, out_path):
    """datasets = [(label, items, rows), ...]"""
    lines = []
    lines.append("# 人数统计（SAM3）")
    lines.append("")
    lines.append("**规则：** SAM `person` prompt 返回 ≥ 2 个区域 → 多人，否则 单人。")
    lines.append("")

    lines.append("## 汇总")
    lines.append("")
    lines.append("| 数据集 | 总图数 | 单人 | 多人 | 单人占比 | 多人占比 |")
    lines.append("|--------|-------:|-----:|-----:|---------:|---------:|")
    for label, items, rows in datasets:
        total = len(items)
        single = sum(1 for r in rows if r[1] == "single")
        multi = sum(1 for r in rows if r[1] == "multi")
        s_pct = f"{single * 100 / total:.1f}%" if total else "-"
        m_pct = f"{multi * 100 / total:.1f}%" if total else "-"
        lines.append(f"| {label} | {total} | {single} | {multi} | {s_pct} | **{m_pct}** |")
    lines.append("")

    lines.append("## SAM `person` 数分布")
    lines.append("")
    for label, items, rows in datasets:
        dist = person_distribution(items)
        total = sum(dist.values())
        lines.append(f"### {label}（共 {total} 张）")
        lines.append("")
        lines.append("| person 数 | 张数 | 占比 | 分类 |")
        lines.append("|----------:|-----:|-----:|------|")
        for k in sorted(dist.keys()):
            v = dist[k]
            pct = f"{v * 100 / total:.1f}%"
            cls = "多人" if k >= 2 else "单人"
            lines.append(f"| {k} | {v} | {pct} | {cls} |")
        lines.append("")

    lines.append("## 重跑")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/classify_persons.py \\")
    lines.append('  --label 正向 --json-dir "/home/edward/research/lianzhong-project/data/outputs/正向教学教材/json" \\')
    lines.append('  --label 反向 --json-dir "/home/edward/research/lianzhong-project/data/outputs/反向教学教材/json" \\')
    lines.append('  --out "/home/edward/research/lianzhong-project/data/outputs/人数统计.md"')
    lines.append("```")
    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.replace(tmp, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--json-dir", action="append", required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    if len(args.label) != len(args.json_dir):
        parser.error("--label and --json-dir must be given same number of times")

    datasets = []
    for label, jd in zip(args.label, args.json_dir):
        items = load_jsons(jd)
        rows = classify_batch(items)
        s = sum(1 for r in rows if r[1] == "single")
        m = sum(1 for r in rows if r[1] == "multi")
        print(f"{label:>6s}: total={len(items):3d}  单人={s:3d}  多人={m:3d}")
        datasets.append((label, items, rows))

    render_doc(datasets, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
