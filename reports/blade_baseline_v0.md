# Blade Baseline v0 — SAM3 + Gemma3 Zero-Shot Pipeline

**生成日期 / Date**: 2026-05-07
**Plan**: `/home/edward/.claude/plans/8-jaunty-glade.md` Layer 0 (T13–T15)
**Repo SHA**: `47e4462498a2b1f6f32a1ae07f578d12e9bf6bbb`
**作者 / Author**: blade pipeline maintainer (auto-generated under T15)

> 本文档汇总 Task T14 产出的零样本（Layer 0）评测指标，分析 Top-10 失败模式，并给出 Layer 1+ 优先级建议。
> Out-of-scope: 不包含任何 Layer 1+ 实现。

---

## Reproduction

```bash
# 数据集制备 (D5c VOC -> Roboflow YOLO)
.venv/bin/python scripts/blade/converters/d5c_figshare_voc.py \
    --src data/blade_eval/d5c_voc \
    --dst data/blade_eval/d5c_yolo

# 端到端评测 (T13/T14 入口)
make eval-blade DATASET=d5c_yolo OUT=reports/blade/20260507/d5c/baseline
```

输出目录：`reports/blade/20260507/d5c/baseline/`
（`metrics.yaml`、`predictions.jsonl`、`report.html`，已被 `.gitignore` 屏蔽以避免大文件入库。）

---

## Baseline Metrics

### Dataset

| 字段 / Field | 值 / Value |
|---|---|
| 数据集 / Dataset | D5c Figshare WTBD Multiclass UAV (CC BY 4.0) |
| 总图像 / Total images | 1065 |
| 本次评测切片 / Eval slice | 80 (全部正样本，数据集无 negative) |
| 类别空间 / GT taxonomy (6) | crack, corrosion, craze, hide_craze, surface_injure, thunderstrike (+ erosion/contamination 在 union 中由 VLM 引入) |
| GT 框总数 / Total GT bboxes | 121 |
| Severity 标签 / Severity labels | 无 / none |

### Defect Classification (binary: has_defect vs none)

| Metric | Value |
|---|---|
| Accuracy | 0.675 |
| Precision | 1.000 |
| Recall | 0.675 |
| F1 | 0.806 |
| n | 80 |

> precision=1.0 是因为 D5c 切片中没有 negative 样本，模型说 "有缺陷" 永远不会撞到 false positive。**Recall 0.675 是真正的瓶颈** —— 32% 的实际缺陷被漏检。

### Defect Type Classification (11-class)

GT × pred 类别 union = `{coating_peel, contamination, corrosion, crack, craze, erosion, hide_craze, lightning_hole, none, surface_injure, thunderstrike}`

关键单元格（confusion matrix 摘要，行=GT、列=Pred）：

| GT class | n | 主要预测分布 / Main pred distribution |
|---|---|---|
| crack | 12 | **none = 10**, erosion = 2 (Recall hole) |
| craze | 21 | erosion = 11, coating_peel = 5, crack = 3, none = 2 |
| hide_craze | 17 | crack = 8, none = 6, coating_peel = 1, lightning_hole = 1, erosion = 1 |
| corrosion | 9 | erosion = 6, none = 2, coating_peel = 1 |
| surface_injure | 10 | coating_peel = 4, none = 3, lightning_hole = 2, contamination = 2, crack = 1 |
| thunderstrike | 9 | none = 3, coating_peel = 2, erosion = 2, contamination = 1, lightning_hole = 1 |

**核心观察**: VLM 输出标签集严重偏离 GT taxonomy。erosion / coating_peel / none / lightning_hole / contamination 这五个 VLM 自创 / 偏好 label 占据了大部分预测，**没有任何样本被预测为 GT 中存在的 craze / hide_craze / surface_injure / thunderstrike** 这四个类。

### Localization

| Metric | Value |
|---|---|
| mAP@0.5 | **0.000** |
| n_gt_boxes | 121 |
| n_pred_boxes | **0** |

> SAM3 用 prompt `"defect"` 在 80/80 张图上全部返回 0 region。开放词汇 head 在该数据集上完全失效，整条流水线只剩下 VLM 在 "看图"。

### Severity

`severity: null` — D5c 数据集没有 GT severity 标签，**本次评测无法验证严重度模块**。

详见 `reports/blade/20260507/d5c/baseline/report.html`（错误画廊）。

---

## Failure Mode Analysis

### Top-10 representative failures

| # | image_id | GT defect_type | VLM pred | conf | failure archetype | evidence (truncated) |
|---|---|---|---|---|---|---|
| 1 | 12 | crack | none | n/a | **A. FN-crack** (recall hole) | (empty — VLM 直接说 has_defect=false) |
| 2 | 23 | crack | none | n/a | **A. FN-crack** | (empty) |
| 3 | 26 | crack | none | n/a | **A. FN-crack** | (empty) |
| 4 | 33 | crack | none | n/a | **A. FN-crack** | (empty) |
| 5 | 13 | hide_craze | crack | 0.95 | **B. label hallucination** | "图像左下侧可见一条明显的黑色细长裂纹，呈不规则延伸状..." |
| 6 | 25 | hide_craze | crack | 0.95 | **B. label hallucination** | "图像中央存在一条贯穿纵向的明显裂纹，长度超过 5cm..." |
| 7 | 1 | craze | erosion | 0.98 | **C. high-conf type-mismatch** | "前缘大面积结构性破损，白色涂层完全脱落，基材撕裂..." |
| 8 | 15 | craze | erosion | 0.95 | **C. high-conf type-mismatch** | "前缘明显破损与缺口，深色基材外露，符合前缘磨蚀特征" |
| 9 | 0 | corrosion | erosion | 0.95 | **C. high-conf type-mismatch** | "前缘深色纵向条带，涂层磨损剥落，伴侵蚀纹理" |
| 10 | 2 | surface_injure | coating_peel | 0.92 | **D. mid-conf taxonomy drift** | "表面深色斑块，不规则形状，涂层剥落特征" |

### Categorized breakdown

**Archetype A — Crack 漏检（Recall hole, 12 → 10 false negatives）**
- VLM 在 `gt=crack` 的 12 张图里 10 张直接给出 `has_defect=false`，evidence 为空。
- 推断原因：crack 图像通常细节微小（细线），VLM 在 zero-shot 默认模板下倾向给 "无明显结构性损伤" 而拒绝 fire。
- 这是 recall=0.675 的最大单一贡献者。

**Archetype B — Hide-craze 被升级为 crack（hallucinated severity）**
- `hide_craze` (隐性龟裂) 17 例中 8 例被高置信度（0.95）预测为 `crack`。
- VLM 把任何看起来像线条的纹理都升级到 "crack"，而 D5c taxonomy 区分 hide_craze（细网状裂纹）和 crack（贯穿性宏观裂纹）。
- 提示模板缺少类别定义/区分准则。

**Archetype C — 大量真实缺陷被路由到 "erosion"（taxonomy collapse）**
- `craze`、`corrosion`、`thunderstrike` 三类合计 39 例中，约 19 例被预测为 `erosion`，置信度普遍 0.95–0.98。
- VLM 把 "前缘大面积涂层脱落 + 基材外露" 这一通用视觉特征统一映射到 `erosion`，而它本应是 GT 中的 craze（龟裂）/corrosion（腐蚀）/coating_peel 等更具体类别。
- 根因：prompt 没有提供候选类别枚举（class enum）和类别定义。

**Archetype D — Surface_injure 被均匀分散到无关类**
- 10 例 `surface_injure` 散落到 coating_peel(4)/none(3)/lightning_hole(2)/contamination(2)/crack(1)。
- "划痕 / 表面损伤" 是 VLM 词汇里没有显式对应词的类，分类器退化为 "看上去像什么就叫什么"。

**Archetype E — SAM3 0 region（全局 localization 失败）**
- 不在 Top-10 表格中显式列出，但作用于所有 80 例。
- SAM3 用 `"defect"` 单 prompt 完全没找到 region → mAP@0.5=0 → 流水线退化为 "VLM-only image classifier"。

---

## Next Layer Priority

排名按 "失败模式覆盖度 × 实现成本" 综合考虑，Layer 1 显著领先。

### 优先级 1（最高）— Layer 1: Prompt 工程

**针对失败模式**: A、B、C、D、E（覆盖全部）。
**为何最高**:
1. **多 prompt SAM3 直接修复 Archetype E**：用 `["crack", "erosion", "corrosion", "coating peel", "lightning hole", "contamination"]` 多 prompt 调用 SAM3 并合并 mask，理论上能从 0 region 跃升到非零 mAP。这是当前流水线 0→1 的拐点。
2. **类别枚举 + JSON schema 修复 Archetype B/C/D**：在 VLM 提示中加入 `defect_type ∈ {crack, corrosion, craze, hide_craze, surface_injure, thunderstrike, none}` 的硬约束，并在 prompt 内给出每个类的简明定义和区分要点（"hide_craze: 细密网状细纹；crack: 单一贯穿性宏观裂纹"），可直接收敛 19 例 `→erosion` 的 collapse 与 8 例 `hide_craze→crack` 的升级。
3. **Negative example / CoT 修复 Archetype A**：明确指示 VLM "对线条状细节做近距离判断；不要因为面积小就拒绝 fire"。
**预期收益**: defect_type accuracy 从 ~0.15 提升到 ≥0.45；recall 从 0.675 提升到 ≥0.85。
**实现代价**: S（1–2 天）。

### 优先级 2 — Layer 4: Voting / Self-consistency

**针对失败模式**: A（recall hole）、E（SAM3 region 不稳）。
**为何排第二**:
- T13 baseline 用 temperature=0.0、单次采样。把 VLM 改成 temperature=0.3、采 3–5 次后做多数投票，能直接缓解 Archetype A 的 "默认拒绝" 倾向。
- SAM3 多 prompt 输出本身需要一个 NMS / region-vote 合并步骤（与 Layer 1 实现紧耦合，但属 Layer 4 思想）。
**预期收益**: 在 Layer 1 之后再加 +3–5pt accuracy / +5pt recall。
**实现代价**: S–M。

### 优先级 3 — Layer 2: Static Few-shot ICL

**针对失败模式**: A（crack 漏检）、B（hide_craze ≠ crack）、D（surface_injure 无对应词）。
**为何排第三**:
- 给 VLM 提示中嵌入每个 GT 类别的 1–2 张参考图 + 简短描述，明确 hide_craze vs crack vs surface_injure 的视觉差异。
- 对那些 VLM 自身先验里没有清晰表征的类（surface_injure、hide_craze）尤其有效。
- 必须在 Layer 1 之后做：先把类别枚举搞对，few-shot 才有意义。
**实现代价**: M（需要人工挑选 reference image）。

### 优先级 4 — Layer 3: DINOv2 RAG retrieval

**针对失败模式**: B、D（细粒度类别）、长尾。
**为何排第四**:
- Layer 2 是静态 few-shot；Layer 3 用 DINOv2 在训练集（D5c train split）上检索最相似 K 张，动态注入 prompt。
- 在数据规模 ≥ 几千张时收益明显，n=80 评测集上效果有限。
- 依赖 host 端有可索引的 reference pool（当前 D5c 1065 张可用）。
**实现代价**: M。

### 优先级 5 — Layer 5: Severity geometric module

**针对失败模式**: 无法在当前 baseline 上验证。
**为何排第五**:
- Severity 在 D5c 上 GT=null，**实现了也测不出来**。
- 阻塞项：要么获取/标注 D1 + 规则映射的 severity，要么对 D5c 的 80 张人工补 severity 标签。
- 在 severity GT 落地之前，本层不应消耗主线工时。
**实现代价**: 模块本身 M，但需要先解决 GT 问题。

### 暂不优先 — Layer 6 / Layer 7

- **Layer 6 (CLIP/DINOv2 linear probe)**：需要 host-side 训练数据 pipeline，对当前 80 张评测集不划算。
- **Layer 7 (hard-case fallback)**：等 Layer 1–4 上线后，再针对剩余难例设计降级策略。

---

## 不确定性与已知局限 / Caveats

- n=80 切片对 6 类是欠采样的（每类约 5–17 例），confusion matrix 单元格统计噪声大；"hide_craze→crack 8 例" 这类结论具备方向性但不应作为强统计断言。
- 数据集无 negative，导致 precision=1.0 是 trivial 的；增加 negative slice（健康叶片图）后该指标会显著下降，需要在 Layer 1 评测里补做。
- VLM evidence 字段在 false-negative 时为空，无法直接还原拒绝原因；后续应在 prompt 中要求 "即使判定 has_defect=false 也输出 reasoning"，便于错误归因。
- SAM3 0-region 是否仅与 prompt 词 "defect" 有关，还是模型本身在该数据分布下失效，需要在 Layer 1 多 prompt 实验中证伪。

---

## References

- Plan: `/home/edward/.claude/plans/8-jaunty-glade.md`（Layer 0 §3, Layer 1+ §9）
- T14 metrics: `reports/blade/20260507/d5c/baseline/metrics.yaml`
- T14 predictions: `reports/blade/20260507/d5c/baseline/predictions.jsonl`
- T14 error gallery: `reports/blade/20260507/d5c/baseline/report.html`
- Dataset converter: `scripts/blade/converters/d5c_figshare_voc.py`
