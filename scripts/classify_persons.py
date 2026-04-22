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


def classify_sam_only(item):
    """只看 SAM 的 per_prompt_counts，不看 Gemma。返回 ('single'|'multi', reason)."""
    if item.get("status") != "ok":
        return "single", f"status={item.get('status')}"
    ppc = item.get("per_prompt_counts") or {}
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
        return "multi", f"SAM 异常: {', '.join(triggers)}"
    return "single", "SAM 无多人迹象"


def classify_gemma_only(item):
    """只看 Gemma 的 num_persons。parse_error (-1) → 单人（保守）。"""
    if item.get("status") != "ok":
        return "single", f"status={item.get('status')}"
    try:
        n = int(item.get("num_persons", -1))
    except (TypeError, ValueError):
        n = -1
    if n >= 2:
        return "multi", f"Gemma={n}"
    if n < 0:
        return "single", f"Gemma 解析失败 (-1) → 保守归单人"
    return "single", f"Gemma={n}"


def classify_hybrid(item):
    """SAM 或 Gemma 任一说多人就算多人。"""
    s_verdict, s_reason = classify_sam_only(item)
    g_verdict, g_reason = classify_gemma_only(item)
    if s_verdict == "multi" or g_verdict == "multi":
        parts = []
        if g_verdict == "multi": parts.append(g_reason)
        if s_verdict == "multi": parts.append(s_reason)
        return "multi", " | ".join(parts)
    return "single", f"{g_reason}; {s_reason}"


def load_jsons(json_dir):
    items = []
    for name in sorted(os.listdir(json_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(json_dir, name), "r", encoding="utf-8") as f:
            items.append(json.load(f))
    return items


def classify_batch(items, classifier=classify_hybrid):
    rows = []
    for it in items:
        verdict, reason = classifier(it)
        rows.append((it.get("image", "?"), verdict, reason, it))
    single = sum(1 for r in rows if r[1] == "single")
    multi = sum(1 for r in rows if r[1] == "multi")
    return single, multi, rows


def count_parse_errors(items):
    cnt = 0
    for it in items:
        if it.get("status") != "ok":
            continue
        try:
            n = int(it.get("num_persons", -1))
        except (TypeError, ValueError):
            n = -1
        if n < 0:
            cnt += 1
    return cnt


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
    """datasets = [(label, items, sam_rows, gemma_rows, hybrid_rows), ...]"""
    lines = []
    lines.append("# 人数统计（单人 / 多人）")
    lines.append("")
    lines.append("> 基于 SAM3 (9 body-part prompts) + Gemma 4 的批处理结果，分别按 **SAM3 only**、**Gemma4 only**、**混合规则** 三种方式分类。")
    lines.append("")

    lines.append("## 三种分类器的规则")
    lines.append("")
    lines.append("### SAM3 only（只看 SAM 的原始 9 prompt 检测数）")
    lines.append("")
    lines.append("判定 **多人** 当以下任一满足：")
    lines.append("")
    lines.append(f"- SAM 的 `person / face / head / torso`（人只有 1 个的部位）任一 ≥ **{UNIQUE_THRESHOLD}**")
    lines.append(f"- SAM 的 `hands / arm / legs / feet / shoulder`（单人有 2 个）任一 ≥ **{PAIRED_THRESHOLD}**（阈值 3 给 SAM 过分割留一格容错）")
    lines.append("")
    lines.append("其余归 **单人**。")
    lines.append("")
    lines.append("### Gemma4 only（只看 VLM 的推理结果 `num_persons`）")
    lines.append("")
    lines.append("- `num_persons >= 2` → **多人**")
    lines.append("- `num_persons == 0 或 1` → **单人**")
    lines.append("- `num_persons == -1`（JSON 解析失败）→ **单人**（保守归入，会在下表的脚注里单独列出数量）")
    lines.append("")
    lines.append("### 混合（推荐）")
    lines.append("")
    lines.append("SAM3 或 Gemma4 任一判为多人 → **多人**；否则 → 单人。")
    lines.append("")
    lines.append("## 为什么需要混合？")
    lines.append("")
    lines.append("Gemma 4 是 2B 参数的轻量模型，推理倾向保守 —— 反向教学教材里它把 80 张图判成「1 人」，但 SAM 看到的身体部位明显是多人：")
    lines.append("")
    lines.append("```")
    lines.append("0248a7e8...webp.jpg    person=2 arm=3                (2 人以上)")
    lines.append("20250731_143429...jpg  person=2 face=1 hands=3       (2 人)")
    lines.append("202508251027...webp    person=4 face=2 head=2        (4 人)")
    lines.append("20250829_103619...jpg  person=7 face=3 head=3 feet=5 (多人聚会)")
    lines.append("```")
    lines.append("")
    lines.append("SAM 的原始计数直接反过来约束 Gemma，就能把这些场景捞回来。")
    lines.append("")

    lines.append("## 总表（三种分类并排）")
    lines.append("")
    lines.append("| 数据集 | 总图数 | SAM3 单 / 多 | Gemma4 单 / 多 | 混合 单 / 多 | Gemma 解析失败 |")
    lines.append("|--------|--------|--------------|----------------|--------------|----------------|")
    for label, items, sam_rows, g_rows, h_rows in datasets:
        total = len(items)
        sam_s = sum(1 for r in sam_rows if r[1] == "single")
        sam_m = sum(1 for r in sam_rows if r[1] == "multi")
        g_s = sum(1 for r in g_rows if r[1] == "single")
        g_m = sum(1 for r in g_rows if r[1] == "multi")
        h_s = sum(1 for r in h_rows if r[1] == "single")
        h_m = sum(1 for r in h_rows if r[1] == "multi")
        pe = count_parse_errors(items)
        lines.append(
            f"| {label} | {total} | {sam_s} / **{sam_m}** ({sam_m*100//total}%) | "
            f"{g_s} / **{g_m}** ({g_m*100//total}%) | "
            f"{h_s} / **{h_m}** ({h_m*100//total}%) | {pe} |"
        )
    lines.append("")
    lines.append("「Gemma 解析失败」= Gemma 返回的 JSON 没法 parse 或给了非整数 `num_persons`，这些在 Gemma4-only 列里按保守策略归入单人。")
    lines.append("")

    lines.append("## SAM3 vs Gemma4 的分歧（混淆矩阵）")
    lines.append("")
    lines.append("对角线是一致判定，非对角是不一致 —— 越集中在对角线，两个模型越同步。")
    lines.append("")
    for label, items, sam_rows, g_rows, h_rows in datasets:
        # Build per-image index → (sam, gemma)
        pair = [(sam_rows[i][1], g_rows[i][1]) for i in range(len(items))]
        ss = sum(1 for p in pair if p == ("single", "single"))
        sm = sum(1 for p in pair if p == ("single", "multi"))
        ms = sum(1 for p in pair if p == ("multi", "single"))
        mm = sum(1 for p in pair if p == ("multi", "multi"))
        lines.append(f"### {label}")
        lines.append("")
        lines.append("|  | Gemma 单 | Gemma 多 |")
        lines.append("|--|---------|---------|")
        lines.append(f"| **SAM 单** | {ss} | {sm} |")
        lines.append(f"| **SAM 多** | {ms} | {mm} |")
        lines.append("")
        lines.append(f"SAM 判多但 Gemma 判单：**{ms}** 张（Gemma 漏了，SAM 救回来）。")
        lines.append(f"Gemma 判多但 SAM 判单：**{sm}** 张（SAM 部位少但 Gemma 从语义角度判多，需抽查）。")
        lines.append("")

    lines.append("## 被 SAM3 升级成多人的样本（前 15 条，每批）")
    lines.append("")
    lines.append("这些是 Gemma 判单人 / 无人、但 SAM 看到多人身体部位的图 —— 混合分类器的主要增量来自这里。")
    lines.append("")
    for label, items, sam_rows, g_rows, h_rows in datasets:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| 图 | Gemma | SAM 触发理由 |")
        lines.append("|----|-------|--------------|")
        upgraded = []
        for i in range(len(items)):
            if sam_rows[i][1] == "multi" and g_rows[i][1] == "single":
                upgraded.append((sam_rows[i], items[i]))
        for (sam_row, it) in upgraded[:15]:
            name = sam_row[0]
            reason = sam_row[2]
            g = it.get("num_persons", "?")
            short = name if len(name) <= 50 else name[:47] + "…"
            lines.append(f"| `{short}` | {g} | {reason} |")
        if len(upgraded) > 15:
            lines.append(f"| … | | 还有 {len(upgraded)-15} 张 |")
        lines.append("")

    lines.append("## Gemma 独家的多人判定（SAM 没触发）")
    lines.append("")
    lines.append("这些是 Gemma 说 ≥2 人、但 SAM 的部位计数没到阈值 —— 可能是 Gemma 从构图/语境推断出多人（例如远景两个小人影），也可能是 Gemma 幻觉，需要人工抽查。")
    lines.append("")
    for label, items, sam_rows, g_rows, h_rows in datasets:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| 图 | Gemma | SAM 计数 |")
        lines.append("|----|-------|----------|")
        gonly = []
        for i in range(len(items)):
            if sam_rows[i][1] == "single" and g_rows[i][1] == "multi":
                gonly.append(items[i])
        for it in gonly[:10]:
            name = it.get("image", "?")
            short = name if len(name) <= 50 else name[:47] + "…"
            ppc = it.get("per_prompt_counts") or {}
            compact = " ".join(f"{k}={ppc.get(k,0)}" for k in ["person","face","hands","arm","legs","feet"])
            lines.append(f"| `{short}` | {it.get('num_persons','?')} | {compact} |")
        if len(gonly) > 10:
            lines.append(f"| … | | 还有 {len(gonly)-10} 张 |")
        if not gonly:
            lines.append("| — | — | 无 |")
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
    lines.append("脚本在 `scripts/classify_persons.py`。可 import 的函数：`classify_sam_only(item)`、`classify_gemma_only(item)`、`classify_hybrid(item)`，都返回 `(verdict, reason)`。")
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
        _, _, sam_rows = classify_batch(items, classify_sam_only)
        _, _, g_rows = classify_batch(items, classify_gemma_only)
        _, _, h_rows = classify_batch(items, classify_hybrid)
        sam_m = sum(1 for r in sam_rows if r[1] == "multi")
        g_m = sum(1 for r in g_rows if r[1] == "multi")
        h_m = sum(1 for r in h_rows if r[1] == "multi")
        n = len(items)
        print(f"{label:>6s}: total={n:3d}  SAM3 multi={sam_m:3d}  Gemma4 multi={g_m:3d}  Hybrid multi={h_m:3d}")
        datasets.append((label, items, sam_rows, g_rows, h_rows))

    render_doc(datasets, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
