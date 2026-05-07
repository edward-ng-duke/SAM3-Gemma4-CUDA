"""Prompt templates for SAM/Gemma blade defect prediction."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# SAM3 baseline prompt (single open-vocabulary phrase).
# ---------------------------------------------------------------------------
SAM3_PROMPT_BASELINE: str = "defect"


# ---------------------------------------------------------------------------
# Severity rubric — read from sibling severity_rubric.md at import time.
# Falls back to a minimal inline rubric if the file is missing so the module
# remains importable in stripped-down deployments.
# ---------------------------------------------------------------------------
_SEVERITY_RUBRIC_FALLBACK = (
    "## none\n- 无缺陷掩膜，表面均匀完整。\n"
    "## mild\n- 缺陷面积 < 1% 弦长²；线状裂纹 < 5 cm；浅表色斑/微小起皮，无基材外露。\n"
    "## moderate\n- 缺陷面积 1%–5% 弦长² 或长度 5–30 cm；明显裂纹/漆面成片剥落/可见基材。\n"
    "## severe\n- 缺陷面积 > 5% 弦长² 或长度 > 30 cm 或多处独立缺陷；贯穿性裂纹/雷击穿孔/结构性损伤。\n"
)


def _load_severity_rubric() -> str:
    rubric_path = Path(__file__).parent / "severity_rubric.md"
    try:
        return rubric_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _SEVERITY_RUBRIC_FALLBACK


_SEVERITY_RUBRIC_TEXT: str = _load_severity_rubric()


# ---------------------------------------------------------------------------
# VLM JSON schema (documentation-only — not used for runtime validation).
# ---------------------------------------------------------------------------
VLM_JSON_SCHEMA: dict[str, str] = {
    "has_defect": "bool",
    "defect_type": "str",
    "severity": "enum[none|mild|moderate|severe]",
    "evidence": "str",
    "confidence": "float[0,1]",
}


# ---------------------------------------------------------------------------
# VLM system prompt (baseline). Chinese, includes task description, JSON
# schema requirement and severity rubric summary.
# ---------------------------------------------------------------------------
VLM_SYSTEM_BASELINE: str = (
    "你是风电叶片缺陷检测专家。给定一张叶片图像与若干 SAM3 候选区域，"
    "你的任务是判断图像中是否存在真实的表面缺陷，若存在则给出缺陷类型与严重度等级。\n\n"
    "判定规则：\n"
    "- has_defect=true 仅在图像中存在可见、非伪影的缺陷时成立；\n"
    "- 镜面反光、阴影、水渍、涂装色差、轻微污垢均不视为缺陷；\n"
    "- defect_type 例如：crack（裂纹）/ erosion（前缘磨蚀）/ delamination（分层）/"
    " coating_peel（涂层剥落）/ lightning_hole（雷击穿孔）/ contamination（污染）等，无缺陷时填 \"none\"；\n"
    "- severity 必须是 none / mild / moderate / severe 四个枚举之一；\n"
    "- evidence：用一句话指出判定依据（位置 + 视觉特征），无缺陷时填 \"\"；\n"
    "- confidence：浮点数，范围 [0, 1]，反映你对最终判定的把握。\n\n"
    "严重度评级标准（4 级，几何特征基于弦长²，视觉特征读取 RGB）：\n"
    f"{_SEVERITY_RUBRIC_TEXT}\n"
    "输出格式（严格 JSON，不允许任何额外文字、Markdown 代码围栏或解释）：\n"
    "{\n"
    '  "has_defect": <bool>,\n'
    '  "defect_type": <string>,\n'
    '  "severity": "none"|"mild"|"moderate"|"severe",\n'
    '  "evidence": <string>,\n'
    '  "confidence": <float in [0,1]>\n'
    "}\n"
)


def build_user_prompt(image_id: str, sam3_regions: list[dict]) -> str:
    """Compose the per-image user prompt for the VLM.

    Parameters
    ----------
    image_id:
        Stable identifier for the image (used by the rater for traceability).
    sam3_regions:
        List of SAM3 candidate region dicts. Each dict may contain ``bbox``
        (4-tuple of ints), ``score`` (float) and ``prompt`` (str) keys; missing
        keys are rendered as ``?``.

    Returns
    -------
    str
        A multi-line prompt ending with ``请输出 JSON。``.
    """

    lines: list[str] = [
        f"Image ID: {image_id}",
        f"SAM3 candidate regions ({len(sam3_regions)}):",
    ]

    if not sam3_regions:
        lines.append("- (无候选区域)")
    else:
        for idx, region in enumerate(sam3_regions):
            prompt = region.get("prompt", "?")
            bbox = region.get("bbox", "?")
            score = region.get("score", "?")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
            lines.append(
                f"- region {idx}: prompt='{prompt}', bbox={bbox}, score={score_str}"
            )

    lines.append("")
    lines.append("请输出 JSON。")
    return "\n".join(lines)
