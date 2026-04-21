"""
人数分类器：读取 batch_detect_persons 产出的 JSON，用 Gemma + SAM 混合规则分类成"单人/多人"。

规则：
- Gemma `num_persons >= 2` → 多人
- 或 SAM 独占部位 (person/face/head/torso) 任一 >= 2 → 多人
- 或 SAM 成对部位 (hands/arm/legs/feet/shoulder) 任一 >= 3 → 多人（单人有 2，3+ 说明第二人）
- 其余 → 单人

用法：
    python scripts/classify_persons.py --label 反向 --json-dir <dir> [--label 正向 --json-dir <dir>] --out <doc.md>
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path


UNIQUE_PARTS = ["person", "face", "head", "torso"]
PAIRED_PARTS = ["hands", "arm", "legs", "feet", "shoulder"]
UNIQUE_THRESHOLD = 2
PAIRED_THRESHOLD = 3


def classify(item):
    """Return ('single'|'multi', reason_str)."""
    if item.get("status") != "ok":
        return "single", f"status={item.get('status')} (non-ok → 归单人占位)"

    gemma = item.get("num_persons", -1)
    try:
        gemma = int(gemma)
    except (TypeError, ValueError):
        gemma = -1

    ppc = item.get("per_prompt_counts") or {}

    if gemma >= 2:
        return "multi", f"Gemma={gemma}"

    triggers = []
    for p in UNIQUE_PARTS:
        c = ppc.get(p, 0)
        if c >= UNIQUE_THRESHOLD:
            triggers.append(f"{p}={c}")
    for p in PAIRED_PARTS:
        c = ppc.get(p, 0)
        if c >= PAIRED_THRESHOLD:
            triggers.append(f"{p}={c}")

    if triggers:
        return "multi", f"Gemma={gemma} 但 SAM 异常: {', '.join(triggers)}"

    return "single", f"Gemma={gemma}, SAM 正常"


def load_jsons(json_dir):
    items = []
    for name in sorted(os.listdir(json_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(json_dir, name), "r", encoding="utf-8") as f:
            items.append(json.load(f))
    return items


def classify_batch(items):
    """Return (single_count, multi_count, rows) where rows = [(image, verdict, reason)]."""
    rows = []
    for it in items:
        verdict, reason = classify(it)
        rows.append((it.get("image", "?"), verdict, reason, it))
    single = sum(1 for r in rows if r[1] == "single")
    multi = sum(1 for r in rows if r[1] == "multi")
    return single, multi, rows


def gemma_distribution(items):
    """Raw Gemma counts for comparison."""
    c = Counter()
    for it in items:
        n = it.get("num_persons", -1)
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = -1
        c[n] += 1
    return c


def render_doc(datasets, out_path):
    """datasets = [(label, items, rows), ...]"""
    lines = []
    lines.append("# 人数统计（单人 / 多人）")
    lines.append("")
    lines.append("> 基于 SAM3 (9 body-part prompts) + Gemma 4 的批处理结果，用混合规则重新分类。")
    lines.append("")
    lines.append("## 为什么不能只信 Gemma？")
    lines.append("")
    lines.append("Gemma 4 用的是 2B 参数的轻量模型，视觉推理能力有限。实测反向批 107 张图里，")
    lines.append("Gemma 把 80 张判为「1 人」，但其中 62 张 SAM 的身体部位计数明显看到多人：")
    lines.append("")
    lines.append("```")
    lines.append("0248a7e8...webp.jpg    person=2 arm=3                (明显 2 人以上)")
    lines.append("20250731_143429...jpg  person=2 face=1 hands=3       (2 人)")
    lines.append("202508251027...webp    person=4 face=2 head=2        (4 人)")
    lines.append("20250829_103619...jpg  person=7 face=3 head=3 feet=5 (多人聚会)")
    lines.append("```")
    lines.append("")
    lines.append("所以需要用 SAM 的原始检测数量来反向校验 Gemma。")
    lines.append("")
    lines.append("## 分类规则")
    lines.append("")
    lines.append("一张图判定为 **多人** 当且仅当以下任一条件满足：")
    lines.append("")
    lines.append(f"1. Gemma 给出 `num_persons >= 2`")
    lines.append(f"2. SAM 在独占部位（person / face / head / torso）中任一 prompt 返回 **≥ {UNIQUE_THRESHOLD}** 个区域 —— 人只有一张脸、一颗头、一个躯干，出现 2+ 明显是多人")
    lines.append(f"3. SAM 在成对部位（hands / arm / legs / feet / shoulder）中任一 prompt 返回 **≥ {PAIRED_THRESHOLD}** 个区域 —— 单人最多 2 个，3+ 说明至少第二人出现了一部分")
    lines.append("")
    lines.append("其余归 **单人**（含 Gemma=0 的无人/近景镜头，按用户要求只要两类故归入单人占位）。")
    lines.append("")
    lines.append("注意 SAM 本身会 over-segment（同一只手被切成两块 mask），所以成对部位阈值取 3 而不是 2，")
    lines.append("给 over-segmentation 留一格容错。")
    lines.append("")
    lines.append("## 总表")
    lines.append("")
    lines.append("| 数据集 | 总图数 | 单人 | 多人 | 多人占比 |")
    lines.append("|--------|--------|------|------|----------|")
    for label, items, rows in datasets:
        total = len(items)
        single = sum(1 for r in rows if r[1] == "single")
        multi = sum(1 for r in rows if r[1] == "multi")
        pct = f"{multi * 100 / total:.1f}%" if total else "-"
        lines.append(f"| {label} | {total} | {single} | {multi} | {pct} |")
    lines.append("")

    lines.append("## Gemma 原判 vs 新分类对照")
    lines.append("")
    lines.append("显示 Gemma 的原始 `num_persons` 分布，以及新规则把多少张原本「非多人」的图升级成「多人」。")
    lines.append("")
    for label, items, rows in datasets:
        dist = gemma_distribution(items)
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Gemma `num_persons` | 张数 | 新分类后这些里有多少多人 |")
        lines.append("|---------------------|------|--------------------------|")
        # per-gemma-bucket, count multi after reclass
        by_bucket = {}
        for r in rows:
            it = r[3]
            try:
                n = int(it.get("num_persons", -1))
            except (TypeError, ValueError):
                n = -1
            by_bucket.setdefault(n, []).append(r)
        for n in sorted(dist.keys()):
            total_in_bucket = dist[n]
            upgraded = sum(1 for r in by_bucket.get(n, []) if r[1] == "multi")
            label_n = f"{n}" if n >= 0 else "-1 (parse_error)"
            lines.append(f"| {label_n} | {total_in_bucket} | {upgraded} |")
        lines.append("")

    lines.append("## 被新规则升级成「多人」的样本（前 15 条，每批）")
    lines.append("")
    lines.append("用来核查规则准不准。")
    lines.append("")
    for label, items, rows in datasets:
        lines.append(f"### {label} — Gemma 原判 ≤ 1 但 SAM 触发多人规则")
        lines.append("")
        lines.append("| 图 | Gemma | 触发理由 |")
        lines.append("|----|-------|----------|")
        upgraded = [r for r in rows if r[1] == "multi" and (
            lambda it: (lambda n: n < 2)(int(it.get("num_persons", -1)) if str(it.get("num_persons", -1)).lstrip("-").isdigit() else -1)
        )(r[3])]
        for name, verdict, reason, it in upgraded[:15]:
            g = it.get("num_persons", "?")
            # trim long filename for readability
            short = name if len(name) <= 50 else name[:47] + "…"
            lines.append(f"| `{short}` | {g} | {reason} |")
        if len(upgraded) > 15:
            lines.append(f"| … | | 还有 {len(upgraded)-15} 张 |")
        lines.append("")

    lines.append("## 如何重新生成本文档")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/classify_persons.py \\")
    lines.append('  --label 反向 --json-dir "/home/edward/research/lianzhong-project/data/outputs/反向教学教材/json" \\')
    lines.append('  --label 正向 --json-dir "/home/edward/research/lianzhong-project/data/outputs/正向教学教材/json" \\')
    lines.append('  --out "/home/edward/research/lianzhong-project/data/outputs/人数统计.md"')
    lines.append("```")
    lines.append("")
    lines.append("脚本本身在 `scripts/classify_persons.py`，分类函数是 `classify(item)`，可以直接 import。")
    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.replace(tmp, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", action="append", required=True, help="Dataset label (repeatable)")
    parser.add_argument("--json-dir", action="append", required=True, help="JSON dir for that label (repeatable, must match --label count)")
    parser.add_argument("--out", type=str, required=True, help="Output markdown path")
    args = parser.parse_args()

    if len(args.label) != len(args.json_dir):
        parser.error("--label and --json-dir must be given same number of times, paired in order")

    datasets = []
    for label, jd in zip(args.label, args.json_dir):
        items = load_jsons(jd)
        single, multi, rows = classify_batch(items)
        print(f"{label:>6s}: total={len(items):3d}  single={single:3d}  multi={multi:3d}")
        datasets.append((label, items, rows))

    render_doc(datasets, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
