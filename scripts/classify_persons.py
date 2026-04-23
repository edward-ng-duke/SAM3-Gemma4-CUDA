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
    lines.append("# 人数统计（SAM3-only，单人 / 多人）")
    lines.append("")
    lines.append("## 算法")
    lines.append("")
    lines.append("**多人** ⇐ SAM 的 `person` prompt 返回 **≥ 2** 个区域；否则 **单人**。")
    lines.append("")
    lines.append("### 为什么是这条规则？")
    lines.append("")
    lines.append("在两批数据上实测了 5 条候选规则，结果如下（目标：正向多人数尽量低、反向多人数尽量高）：")
    lines.append("")
    lines.append("| 规则 | 正向多人 | 反向多人 |")
    lines.append("|------|----------|----------|")
    lines.append("| **A) `person ≥ 2`（当选）** | **17 / 107 (15%)** | **81 / 107 (75%)** |")
    lines.append("| B) `person ≥ 2` 或 `face ≥ 2` | 19 / 107 | 81 / 107 |")
    lines.append("| C) `person ≥ 2` 或 `face ≥ 2` 或 `head ≥ 2` | 19 / 107 | 82 / 107 |")
    lines.append("| D) 任一独占部位 ≥ 2（person/face/head/torso） | 19 / 107 | 82 / 107 |")
    lines.append("| E) 独占 ≥ 2 或 成对部位 ≥ 3（hands/arm/legs/feet/shoulder） | 21 / 107 | 84 / 107 |")
    lines.append("")
    lines.append("- 加入 `face / head ≥ 2` 只在反向多捞 0–1 张，却在正向多误报 2 张 —— 收益 < 成本")
    lines.append("- 加入成对部位阈值（规则 E，旧版用的）给反向再多 2 张，代价是正向再误报 2 张 —— SAM 经常把单人的一只手过分割成两块 mask，`hands = 3` 常见但不意味多人")
    lines.append("- **规则 A 最干净**：物理含义清晰（SAM 看到 ≥2 个 person 实例），没有任何启发式阈值，可解释、不过拟合")
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
        lines.append(f"| {label} | {total} | {single} | **{multi}** | {pct} |")
    lines.append("")

    lines.append("## SAM `person` prompt 的完整分布")
    lines.append("")
    lines.append("直观看每批图里 SAM 找到多少个 person 实例，便于核对阈值合理性。")
    lines.append("")
    for label, items, rows in datasets:
        dist = person_distribution(items)
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| SAM `person` 数 | 张数 | 分类 |")
        lines.append("|-----------------|------|------|")
        for k in sorted(dist.keys()):
            v = dist[k]
            cls = "多人" if k >= 2 else "单人"
            lines.append(f"| {k} | {v} | {cls} |")
        lines.append("")

    lines.append("## 抽查：被分到「多人」的样本（最多 15 条）")
    lines.append("")
    lines.append("如果你手工核对发现某张图实际是单人，说明 SAM 在那张图上把一个人的不同部位错误切成了多个 person blob。目前实测这种情况很少，但欢迎反馈具体样本。")
    lines.append("")
    for label, items, rows in datasets:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| 图 | SAM `person` |")
        lines.append("|----|--------------|")
        multis = [(r, it) for (r, it) in zip(rows, items) if r[1] == "multi"]
        for (row, it) in multis[:15]:
            name = row[0]
            short = name if len(name) <= 50 else name[:47] + "…"
            ppc = it.get("per_prompt_counts") or {}
            lines.append(f"| `{short}` | {ppc.get('person', 0)} |")
        if len(multis) > 15:
            lines.append(f"| … | 还有 {len(multis)-15} 张 |")
        if not multis:
            lines.append("| — | 无 |")
        lines.append("")

    lines.append("## 重跑命令")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/classify_persons.py \\")
    lines.append('  --label 正向 --json-dir "/home/edward/research/lianzhong-project/data/outputs/正向教学教材/json" \\')
    lines.append('  --label 反向 --json-dir "/home/edward/research/lianzhong-project/data/outputs/反向教学教材/json" \\')
    lines.append('  --out "/home/edward/research/lianzhong-project/data/outputs/人数统计.md"')
    lines.append("```")
    lines.append("")
    lines.append("脚本在 `scripts/classify_persons.py`。分类函数 `classify(item) -> (verdict, reason)` 可 import。")
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
