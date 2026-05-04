import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import base64
import io
import json
import re
import tempfile
import spaces
import gradio as gr
import numpy as np
import torch
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from typing import Iterable

import supervision as sv

from servers_client import (
    sam3_detect,
    sam3_track,
    sam3_video,
    vlm_generate,
    vlm_generate_stream,
    vlm_chat_stream,
    safe_parse_json,
    health as sam3_health,
    ServerError,
    SAM3_SERVER_URL,
    QWEN_BASE_URL,
    QWEN_MODEL,
    DetectedRegion,
)

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_VL = "多模态模型"

print(f"🖥️ Compute device (for Gradio glue only): {DEVICE}")
print("ℹ️  This process holds no model weights.")


colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8", c100="#D3E5F0", c200="#A8CCE1", c300="#7DB3D2",
    c400="#529AC3", c500="#4682B4", c600="#3E72A0", c700="#36638C",
    c800="#2E5378", c900="#264364", c950="#1E3450",
)


class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )


steel_blue_theme = SteelBlueTheme()


css = r"""
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

body, .gradio-container { font-family: 'Outfit', sans-serif !important; }
footer { display: none !important; }

/* Gradio 6.14 ships a 1x1 absolute-positioned hidden tab row used for width
   measurement; it lands at the vertical center of the real tablist and steals
   clicks (it has pointer-events: auto by default). Disable hit-testing on it
   so clicks reach the real tabs underneath. */
.tab-container.visually-hidden,
.tab-container.visually-hidden * { pointer-events: none !important; }

.app-header {
    background: linear-gradient(135deg, #1E3450 0%, #264364 30%, #3E72A0 70%, #4682B4 100%);
    border-radius: 16px; padding: 32px 40px; margin-bottom: 24px;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(30,52,80,0.25);
}
.app-header::before {
    content:''; position:absolute; top:-50%; right:-20%;
    width:400px; height:400px;
    background:radial-gradient(circle,rgba(255,255,255,0.06) 0%,transparent 70%);
    border-radius:50%;
}
.header-content {
    display:flex; align-items:center; gap:24px;
    position:relative; z-index:1;
}
.header-icon-wrap {
    width:64px; height:64px; background:rgba(255,255,255,0.12);
    border-radius:16px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; backdrop-filter:blur(8px); border:1px solid rgba(255,255,255,0.15);
}
.header-icon-wrap svg {
    width:36px; height:36px;
    display:block;
}
.header-text h1 {
    font-size:2rem; font-weight:700; color:#fff;
    margin:0 0 8px 0; letter-spacing:-0.02em; line-height:1.2;
}
.header-meta { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.meta-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(255,255,255,0.12); color:rgba(255,255,255,0.9);
    padding:4px 12px; border-radius:20px;
    font-family:'IBM Plex Mono',monospace; font-size:0.8rem; font-weight:500;
    border:1px solid rgba(255,255,255,0.1); backdrop-filter:blur(4px);
}
.meta-badge svg {
    color:#ffffff !important;
    stroke:#ffffff !important;
}
.meta-sep {
    width:4px; height:4px; background:rgba(255,255,255,0.35);
    border-radius:50%; flex-shrink:0;
}
.meta-cap { color:rgba(255,255,255,0.65); font-size:0.85rem; font-weight:400; }

.tab-intro {
    display:flex; align-items:flex-start; gap:16px;
    background:linear-gradient(135deg,rgba(70,130,180,0.06),rgba(70,130,180,0.02));
    border:1px solid rgba(70,130,180,0.15); border-left:4px solid #4682B4;
    border-radius:10px; padding:18px 22px; margin-bottom:20px;
}
.dark .tab-intro {
    background:linear-gradient(135deg,rgba(70,130,180,0.1),rgba(70,130,180,0.04));
    border-color:rgba(70,130,180,0.25);
}
.intro-icon {
    width:40px; height:40px; background:rgba(70,130,180,0.1);
    border-radius:10px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; margin-top:2px;
}
.intro-icon svg { width:22px; height:22px; color:#4682B4; }
.intro-text { flex:1; }
.intro-text p { margin:0; color:#2E5378; font-size:0.95rem; line-height:1.6; }
.dark .intro-text p { color:#A8CCE1; }
.intro-text p.intro-sub { color:#64748b; font-size:0.85rem; margin-top:4px; }
.dark .intro-text p.intro-sub { color:#94a3b8; }

.section-heading {
    display:flex; align-items:center; gap:14px;
    margin:18px 0 14px 0; padding:0 2px;
}
.heading-icon {
    width:32px; height:32px;
    background:linear-gradient(135deg,#4682B4,#3E72A0);
    border-radius:8px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; box-shadow:0 2px 8px rgba(70,130,180,0.2);
}
.heading-icon svg { width:18px; height:18px; color:#fff; }
.heading-label {
    font-weight:600; font-size:1.05rem;
    color:#1E3450; letter-spacing:-0.01em;
}
.dark .heading-label { color:#D3E5F0; }
.heading-line {
    flex:1; height:1px;
    background:linear-gradient(90deg,rgba(70,130,180,0.2),transparent);
}

.status-indicator {
    display:flex; align-items:center; gap:10px;
    padding:10px 16px; margin-top:10px;
    background:rgba(70,130,180,0.04); border:1px solid rgba(70,130,180,0.12);
    border-radius:8px;
}
.status-dot {
    width:8px; height:8px; background:#22c55e;
    border-radius:50%; flex-shrink:0;
    animation:statusPulse 2s ease-in-out infinite;
}
@keyframes statusPulse {
    0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(34,197,94,0.4); }
    50%     { opacity:0.7; box-shadow:0 0 0 4px rgba(34,197,94,0); }
}
.status-text { font-size:0.85rem; color:#64748b; font-style:italic; }

.card-label {
    display:flex; align-items:center; gap:8px;
    font-weight:600; font-size:0.8rem;
    text-transform:uppercase; letter-spacing:0.06em; color:#4682B4;
    margin-bottom:14px; padding-bottom:10px;
    border-bottom:1px solid rgba(70,130,180,0.1);
}
.card-label svg { width:16px; height:16px; }

.primary {
    border-radius:10px !important; font-weight:600 !important;
    letter-spacing:0.02em !important; transition:all 0.25s ease !important;
}
.primary:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 6px 20px rgba(70,130,180,0.3) !important;
}

.gradio-textbox textarea {
    font-family:'IBM Plex Mono',monospace !important;
    font-size:0.92rem !important; line-height:1.7 !important;
    border-radius:8px !important;
}

label { font-weight:600 !important; }

.section-divider {
    height:1px; background:linear-gradient(90deg,transparent,rgba(70,130,180,0.2),transparent);
    margin:16px 0; border:none;
}

@media (max-width: 768px) {
    .app-header { padding: 20px 24px; }
    .header-text h1 { font-size: 1.5rem; }
    .header-content { flex-direction: column; align-items: flex-start; gap: 16px; }
}
"""


T_LOGO_SVG = """
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path fill="white" d="M13.2 2L5 13h5l-1.2 9L19 10h-5l-.8-8Z"/>
</svg>
"""
SVG_IMAGE = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0 0 22.5 18.75V5.25A2.25 2.25 0 0 0 20.25 3H3.75A2.25 2.25 0 0 0 1.5 5.25v13.5A2.25 2.25 0 0 0 3.75 21Z"/></svg>'
SVG_DETECT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 3.75H6A2.25 2.25 0 0 0 3.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0 1 20.25 6v1.5m0 9V18A2.25 2.25 0 0 1 18 20.25h-1.5m-9 0H6A2.25 2.25 0 0 1 3.75 18v-1.5M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"/></svg>'
SVG_OUTPUT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 0 0-1.883 2.542l.857 6a2.25 2.25 0 0 0 2.227 1.932H19.05a2.25 2.25 0 0 0 2.227-1.932l.857-6a2.25 2.25 0 0 0-1.883-2.542m-16.5 0V6A2.25 2.25 0 0 1 6 3.75h3.879a1.5 1.5 0 0 1 1.06.44l2.122 2.12a1.5 1.5 0 0 0 1.06.44H18A2.25 2.25 0 0 1 20.25 9v.776"/></svg>'
SVG_TEXT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z"/></svg>'
SVG_CHIP = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 0 0 2.25-2.25V6.75a2.25 2.25 0 0 0-2.25-2.25H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Z"/></svg>'
SVG_VIDEO = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="m15.75 10.5 4.72-2.36A.75.75 0 0 1 21.75 8.81v6.38a.75.75 0 0 1-1.28.67l-4.72-2.36m0-3v3m-10.5 6h9A2.25 2.25 0 0 0 16.5 17.25V6.75A2.25 2.25 0 0 0 14.25 4.5h-9A2.25 2.25 0 0 0 3 6.75v10.5A2.25 2.25 0 0 0 5.25 19.5Z"/></svg>'


print(f"[app] SAM3 service: {SAM3_SERVER_URL}")
print(f"[app] VLM (Qwen):   {QWEN_BASE_URL} model={QWEN_MODEL}")


BRIGHT_YELLOW = sv.Color(r=255, g=230, b=0)
BLACK = sv.Color(r=0, g=0, b=0)
MASK_COLORS = [
    (255, 230, 0),
    (255, 99, 132),
    (54, 162, 235),
    (75, 192, 192),
    (153, 102, 255),
    (255, 159, 64),
]

VIDEO_COLORS_BGR = [
    (181, 120, 31),
    (13, 128, 255),
    (43, 161, 43),
    (41, 38, 214),
    (189, 102, 148),
    (74, 87, 140),
]


def clamp_box_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def qwen_filter_regions(image: Image.Image, regions: list, user_prompt: str) -> dict:
    region_descriptions = []
    for idx, reg in enumerate(regions):
        x1, y1, x2, y2 = reg["bbox"]
        region_descriptions.append({
            "region_index": idx,
            "bbox": [x1, y1, x2, y2],
            "sam_score": round(float(reg["score"]), 4),
        })

    instruction = f"""
You are given an image and a list of candidate object regions proposed by a segmentation model.

User request:
"{user_prompt}"

Candidate regions:
{json.dumps(region_descriptions, indent=2)}

Task:
Select all candidate regions that match the user request.

Return ONLY valid JSON in this exact format:
{{
  "selected_region_indexes": [0, 2],
  "reason": "short explanation"
}}

Rules:
- Use only indexes from the candidate list.
- If nothing matches, return an empty list.
- Do not return markdown.
"""

    raw = vlm_generate(
        image,
        instruction,
        max_new_tokens=512,
        temperature=0.2,
        enable_thinking=False,
    )

    parsed = safe_parse_json(raw)
    if not isinstance(parsed, dict):
        parsed = {"selected_region_indexes": [], "reason": "无法解析模型输出。"}

    parsed.setdefault("selected_region_indexes", [])
    parsed.setdefault("reason", "")
    return parsed


def overlay_masks_on_image(base_image: Image.Image, masks: list, opacity: float = 0.45):
    base = base_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.array(mask).astype(np.uint8)

        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

        if mask.shape[::-1] != base.size:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(base.size, Image.NEAREST)
        else:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

        color = MASK_COLORS[i % len(MASK_COLORS)]
        fill = Image.new("RGBA", base.size, color + (0,))
        alpha = mask_pil.point(lambda v: int(opacity * 255) if v > 0 else 0)
        fill.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, fill)

    return Image.alpha_composite(base, overlay).convert("RGB")


def annotate_sam3_candidates(image: Image.Image, boxes: list, scores: list, masks: list):
    img = overlay_masks_on_image(image, masks, opacity=0.35)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = MASK_COLORS[i % len(MASK_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"id={i} | {scores[i]:.2f}"
        tb = draw.textbbox((x1, max(0, y1 - 22)), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((tb[0], tb[1]), label, fill="black", font=font)

    return img


def annotate_final_selection(image: Image.Image, selected_regions: list):
    if not selected_regions:
        return image.convert("RGB")

    img = overlay_masks_on_image(
        image,
        [item["mask"] for item in selected_regions],
        opacity=0.45
    )
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, item in enumerate(selected_regions):
        x1, y1, x2, y2 = item["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 230, 0), width=4)
        label = f"{item['label']} | {item['score']:.2f}"
        tb = draw.textbbox((x1, max(0, y1 - 24)), label, font=font)
        draw.rectangle(tb, fill=(255, 230, 0))
        draw.text((tb[0], tb[1]), label, fill="black", font=font)

    return img


def format_json_output(selected_regions, vl_reason, original_prompt):
    return {
        "prompt": original_prompt,
        "num_selected": len(selected_regions),
        "selected_regions": [
            {
                "region_index": item["region_index"],
                "bbox": item["bbox"],
                "score": round(float(item["score"]), 4),
                "label": item["label"],
            }
            for item in selected_regions
        ],
        "vl_reason": vl_reason,
    }


def calc_timeout_duration(video_file, *args):
    return args[-1] if args else 60


def run_sam3_qwen_detection(image, prompt, conf_thresh):
    if image is None:
        raise gr.Error("请先上传一张图片。")
    if not prompt or not prompt.strip():
        raise gr.Error("请填写文本提示词。")

    image = image.convert("RGB")

    try:
        regions = sam3_detect(image, prompt, conf_threshold=float(conf_thresh))
    except ServerError as e:
        raise gr.Error(f"分割检测服务不可达：{e}")
    except Exception as e:
        raise gr.Error(f"检测过程出错：{e}")

    if len(regions) == 0:
        empty_json = {
            "prompt": prompt,
            "num_selected": 0,
            "selected_regions": [],
            "vl_reason": "分割检测模型未找到任何候选区域。"
        }
        return image, image, json.dumps(empty_json, indent=2), "没有检测到任何目标。"

    candidate_regions = [{
        "region_index": r.region_index,
        "bbox": r.bbox,
        "score": r.score,
        "mask": r.mask,
        "label": prompt,
    } for r in regions]

    sam3_vis = annotate_sam3_candidates(
        image,
        [r["bbox"] for r in candidate_regions],
        [r["score"] for r in candidate_regions],
        [r["mask"] for r in candidate_regions],
    )

    try:
        vl_result = qwen_filter_regions(image, candidate_regions, prompt)
    except ServerError as e:
        raise gr.Error(f"多模态服务不可达：{e}")
    except Exception as e:
        raise gr.Error(f"多模态过滤过程出错：{e}")

    selected_idx = vl_result.get("selected_region_indexes", [])
    reason = vl_result.get("reason", "")

    valid_idx = []
    for idx in selected_idx:
        try:
            idx = int(idx)
            if 0 <= idx < len(candidate_regions):
                valid_idx.append(idx)
        except Exception:
            continue

    seen = set()
    valid_idx = [x for x in valid_idx if not (x in seen or seen.add(x))]

    selected_regions = [candidate_regions[i] for i in valid_idx]
    final_vis = annotate_final_selection(image, selected_regions)
    final_json = format_json_output(selected_regions, reason, prompt)

    status = (
        f"分割检测出 {len(candidate_regions)} 个候选区域，"
        f"{MODEL_VL}选中 {len(selected_regions)} 个。"
    )

    return sam3_vis, final_vis, json.dumps(final_json, indent=2), status


def run_video_segmentation(video_path, prompt, frame_limit, time_limit):
    if not video_path:
        raise gr.Error("请先上传一个视频。")
    if not prompt or not prompt.strip():
        raise gr.Error("请填写文本提示词。")

    try:
        out_path, processed_frames, masked_frames = sam3_video(
            video_path=video_path,
            prompt=prompt,
            frame_limit=int(frame_limit),
            time_limit=int(time_limit),
            render_mode="annotated",
        )
    except ServerError as e:
        return None, f"分割检测服务不可达：{e}"
    except Exception as e:
        return None, f"视频处理出错：{str(e)}"

    return (
        out_path,
        f"视频处理完成：共处理 {processed_frames} 帧，"
        f"其中 {masked_frames} 帧带 mask、轮廓与边界框标注。"
    )


def run_video_segmentation_mask(video_path, prompt, frame_limit, time_limit):
    if not video_path:
        raise gr.Error("请先上传一个视频。")
    if not prompt or not prompt.strip():
        raise gr.Error("请填写文本提示词。")

    try:
        out_path, processed_frames, masked_frames = sam3_video(
            video_path=video_path,
            prompt=prompt,
            frame_limit=int(frame_limit),
            time_limit=int(time_limit),
            render_mode="mask",
        )
    except ServerError as e:
        return None, f"分割检测服务不可达：{e}"
    except Exception as e:
        return None, f"视频 mask 处理出错：{str(e)}"

    return (
        out_path,
        f"视频 mask 处理完成：共处理 {processed_frames} 帧，"
        f"其中 {masked_frames} 帧叠加了 mask。"
    )


def run_image_click_gpu(input_image, x, y, points_state, labels_state):
    if input_image is None:
        return input_image, [], []

    if points_state is None:
        points_state = []
    if labels_state is None:
        labels_state = []

    points_state.append([int(x), int(y)])
    labels_state.append(1)

    try:
        overlay, _has_mask = sam3_track(input_image, points_state, labels_state)
        return overlay, points_state, labels_state
    except ServerError as e:
        print(f"Tracker service error: {e}")
        return input_image, points_state, labels_state
    except Exception as e:
        print(f"Tracker Error: {e}")
        return input_image, points_state, labels_state


def image_click_handler(image, evt: gr.SelectData, points_state, labels_state):
    x, y = evt.index
    return run_image_click_gpu(image, x, y, points_state, labels_state)


# ---------- Image Q&A tab handlers ----------

def _pil_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a `data:image/png;base64,...` URL for OpenAI image_url content."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def run_qa_detect_and_crop(image, prompt, conf_thresh):
    """Detect → crop bbox per region → return gallery + crops state."""
    if image is None:
        raise gr.Error("请先上传一张图片。")
    if not prompt or not prompt.strip():
        raise gr.Error("请填写文本提示词。")

    image = image.convert("RGB")

    try:
        regions = sam3_detect(image, prompt, conf_threshold=float(conf_thresh), return_masks=False)
    except ServerError as e:
        raise gr.Error(f"分割检测服务不可达：{e}")

    if len(regions) == 0:
        return [], [], None, [], "没有检测到任何目标。"

    crops: list[Image.Image] = []
    captions: list[str] = []
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)
        captions.append(f"#{r.region_index} score={r.score:.2f} bbox=({x1},{y1},{x2},{y2})")

    gallery_value = list(zip(crops, captions))
    status = f"分割检测返回 {len(regions)} 个候选区域，已生成 {len(crops)} 张抠图。"
    return gallery_value, crops, None, [], status


def on_qa_gallery_select(evt: gr.SelectData, crops_state):
    if crops_state is None or evt.index >= len(crops_state):
        return None, None
    return crops_state[evt.index], int(evt.index)


def chat_with_qwen(message, history, selected_crop):
    """Multi-turn streaming chat. Image only attached to the first turn."""
    if selected_crop is None:
        raise gr.Error("先在 Gallery 中点选一个候选子图。")
    if not message or not message.strip():
        return history, ""

    history = history or []
    img_data_url = _pil_to_data_url(selected_crop)

    messages = []
    for turn_idx, turn in enumerate(history):
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
        else:
            role, content = turn[0], turn[1]
        if turn_idx == 0 and role == "user":
            messages.append({"role": "user", "content": [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": img_data_url}},
            ]})
        else:
            messages.append({"role": role, "content": content})

    if len(history) == 0:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": img_data_url}},
        ]})
    else:
        messages.append({"role": "user", "content": message})

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield new_history, ""

    partial = ""
    try:
        for chunk_text in vlm_chat_stream(messages, enable_thinking=True):
            partial += chunk_text
            new_history[-1] = {"role": "assistant", "content": partial}
            yield new_history, ""
    except ServerError as e:
        new_history[-1] = {"role": "assistant", "content": partial + f"\n\n[多模态错误：{e}]"}
        yield new_history, ""


def clear_qa_chat():
    return [], ""


def explain_detection(image, prompt, detection_json_text):
    if image is None:
        raise gr.Error("请先上传一张图片。")
    if not detection_json_text or not detection_json_text.strip():
        raise gr.Error("请先运行检测。")

    image = image.convert("RGB")
    explain_prompt = f"""
你将看到一张图片、原始的检测提示词，以及一段 JSON 形式的检测结果。

原始提示词：
{prompt}

检测结果 JSON：
{detection_json_text}

请用中文简要说明：
1. 选中了哪些目标
2. 为什么它们匹配提示词
3. 这个结果是否可靠

回答务必简洁清晰，控制在三段之内。
"""

    full_text = ""
    try:
        for chunk in vlm_generate_stream(
            image,
            explain_prompt,
            max_new_tokens=512,
            temperature=0.6,
        ):
            full_text += chunk
            yield full_text
    except ServerError as e:
        raise gr.Error(f"多模态服务不可达：{e}")


def html_header():
    return f"""
    <div class="app-header">
        <div class="header-content">
            <div class="header-icon-wrap">{T_LOGO_SVG}</div>
            <div class="header-text">
                <h1>图像与视频分割</h1>
                <div class="header-meta">
                    <span class="meta-badge">{SVG_CHIP} 分割检测模型服务</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">候选区域</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">多模态过滤</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">图像 + 视频分割</span>
                </div>
            </div>
        </div>
    </div>
    """


def html_tab_intro(icon_svg, title, description, detail=""):
    sub = f'<p class="intro-sub">{detail}</p>' if detail else ""
    return f"""
    <div class="tab-intro">
        <div class="intro-icon">{icon_svg}</div>
        <div class="intro-text">
            <p><strong>{title}</strong> &mdash; {description}</p>
            {sub}
        </div>
    </div>
    """


def html_section_heading(icon_svg, label):
    return f"""
    <div class="section-heading">
        <div class="heading-icon">{icon_svg}</div>
        <span class="heading-label">{label}</span>
        <div class="heading-line"></div>
    </div>
    """


def html_card_label(icon_svg, label):
    return f'<div class="card-label">{icon_svg}<span>{label}</span></div>'


def html_status_indicator(text):
    return f"""
    <div class="status-indicator">
        <span class="status-dot"></span>
        <span class="status-text">{text}</span>
    </div>
    """


def html_divider():
    return '<div class="section-divider"></div>'


EXAMPLES = [
    ["examples/1.jpg", "grapes", 0.45],
    ["examples/2.jpg", "face", 0.35],
    ["examples/3.jpg", "croissant", 0.30],
]

VIDEO_EXAMPLES = [
    ["examples/1V.mp4", "cheetah", 120, 120],
]


with gr.Blocks() as demo:
    gr.HTML(html_header())

    with gr.Tabs():
        with gr.Tab("图像检测"):
            gr.HTML(html_tab_intro(
                SVG_IMAGE,
                "图像检测",
                "分割检测模型先根据你的文本提示输出候选 mask 和区域。多模态模型再对这些候选做过滤，只保留最匹配你描述的那些。",
                "图像模式：分割检测出候选，多模态过滤最终结果。",
            ))

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(html_card_label(SVG_IMAGE, "输入"))
                    image_input = gr.Image(type="pil", label="上传图片", height=360)

                    prompt_input = gr.Textbox(
                        label="检测提示词",
                        placeholder="例如：穿黑色上衣的人",
                        lines=2,
                    )

                    with gr.Accordion("高级设置", open=False):
                        conf_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.45,
                            step=0.05,
                            label="分割检测置信度阈值",
                        )

                    detect_btn = gr.Button("运行检测", variant="primary")
                    explain_btn = gr.Button("解释结果", variant="secondary")

                    gr.HTML(html_divider())

                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[image_input, prompt_input, conf_slider],
                        label="示例",
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_DETECT, "分割检测候选区域"))
                    sam3_output = gr.Image(label="分割检测结果", height=300)

                    gr.HTML(html_section_heading(SVG_OUTPUT, "多模态过滤后的最终检测"))
                    final_output = gr.Image(label="最终检测结果", height=300)

                    gr.Markdown(
                        f"""
                        ### 使用步骤

                        #### 1. 上传图片 + 写提示词
                        - 上传你要分析的图片，写一句清晰的检测描述。

                        #### 2. 调整分割检测设置
                        - 用 **置信度阈值滑块** 控制分割检测模型的严格程度：
                          - **值越低** → 候选越多，**值越高** → 候选越少越干净。

                        #### 3. 运行检测与解释
                        - 点 **"运行检测"**
                        - **上方面板：** 分割检测候选区域，**下方面板：** 多模态过滤后的最终检测
                        - **JSON 输出：** 结构化结果，含 bbox、score、label
                        - 点 **"解释结果"** 让多模态模型用自然语言说明为什么这么选
                        """
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_TEXT, "结构化输出"))
                    json_output = gr.Textbox(label="检测结果 JSON", lines=18, interactive=True)

                    status_output = gr.Textbox(label="系统状态", interactive=False)

                    gr.HTML(html_status_indicator(
                        "流水线：分割检测出候选 → 多模态过滤相关检测。"
                    ))

                    gr.HTML(html_section_heading(SVG_TEXT, "多模态解释"))
                    explanation_output = gr.Textbox(label="解释", lines=15, interactive=True)

            detect_btn.click(
                fn=run_sam3_qwen_detection,
                inputs=[image_input, prompt_input, conf_slider],
                outputs=[sam3_output, final_output, json_output, status_output],
            )

            explain_btn.click(
                fn=explain_detection,
                inputs=[image_input, prompt_input, json_output],
                outputs=[explanation_output],
            )

        with gr.Tab("视频 mask"):
            gr.HTML(html_tab_intro(
                SVG_VIDEO,
                "视频分割（mask 叠加）",
                "用文本提示在视频每一帧里分割目标物体，仅在原始帧上渲染彩色 mask 叠加层。",
                "视频模式：文本提示分割，仅显示 mask 叠加。",
            ))

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(html_card_label(SVG_VIDEO, "视频输入"))
                    video_input_mask = gr.Video(label="上传视频", format="mp4", height=320)

                    video_prompt_mask = gr.Textbox(
                        label="分割提示词",
                        placeholder="例如：球员、跑动的人、红色汽车",
                        lines=2,
                    )

                    with gr.Accordion("高级设置", open=False):
                        with gr.Row():
                            frame_limiter_mask = gr.Slider(
                                minimum=10,
                                maximum=1000,
                                value=60,
                                step=10,
                                label="最大帧数",
                            )
                            time_limiter_mask = gr.Radio(
                                choices=[60, 120, 180, 240, 300],
                                value=60,
                                label="超时（秒）",
                            )

                    video_btn_mask = gr.Button("运行视频 mask 分割", variant="primary")

                    gr.HTML(html_divider())

                    gr.Examples(
                        examples=VIDEO_EXAMPLES,
                        inputs=[video_input_mask, video_prompt_mask, frame_limiter_mask, time_limiter_mask],
                        label="视频示例",
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_OUTPUT, "处理后视频"))
                    video_output_mask = gr.Video(label="带 mask 的视频", height=420)

                    video_status_mask = gr.Textbox(label="系统状态", interactive=False)

                    gr.HTML(html_status_indicator(
                        "流水线：分割检测视频 session → 提示词条件化 → mask 在帧间传播并叠加渲染。"
                    ))

            video_btn_mask.click(
                fn=run_video_segmentation_mask,
                inputs=[video_input_mask, video_prompt_mask, frame_limiter_mask, time_limiter_mask],
                outputs=[video_output_mask, video_status_mask],
            )

        with gr.Tab("视频标注"):
            gr.HTML(html_tab_intro(
                SVG_VIDEO,
                "视频分割",
                "用文本提示在视频帧间分割目标物体。分割检测模型初始化一个视频 session，然后把分割 mask 在整段视频里传播。",
                "视频模式：文本提示分割，输出带 mask、轮廓与边界框。",
            ))

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(html_card_label(SVG_VIDEO, "视频输入"))
                    video_input = gr.Video(label="上传视频", format="mp4", height=320)

                    video_prompt = gr.Textbox(
                        label="分割提示词",
                        placeholder="例如：球员、跑动的人、红色汽车",
                        lines=2,
                    )

                    with gr.Accordion("高级设置", open=False):
                        with gr.Row():
                            frame_limiter = gr.Slider(
                                minimum=10,
                                maximum=1000,
                                value=60,
                                step=10,
                                label="最大帧数",
                            )
                            time_limiter = gr.Radio(
                                choices=[60, 120, 180, 240, 300],
                                value=60,
                                label="超时（秒）",
                            )

                    video_btn = gr.Button("运行视频分割", variant="primary")

                    gr.HTML(html_divider())

                    gr.Examples(
                        examples=VIDEO_EXAMPLES,
                        inputs=[video_input, video_prompt, frame_limiter, time_limiter],
                        label="视频示例",
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_OUTPUT, "处理后视频"))
                    video_output = gr.Video(label="分割视频", height=420)

                    video_status = gr.Textbox(label="系统状态", interactive=False)

                    gr.HTML(html_status_indicator(
                        "流水线：分割检测视频 session → 提示词条件化 → mask 在帧间传播，输出含轮廓与边界框。"
                    ))

            video_btn.click(
                fn=run_video_segmentation,
                inputs=[video_input, video_prompt, frame_limiter, time_limiter],
                outputs=[video_output, video_status],
            )

        with gr.Tab("点选分割"):
            gr.HTML(html_tab_intro(
                SVG_IMAGE,
                "交互式点选分割",
                "上传图片，然后在你想分割的物体上点击。每次点击都被当作一个前景点累加，分割检测模型会实时更新 mask 预览。",
                "交互模式：累计前景点点击式分割。",
            ))

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(html_card_label(SVG_IMAGE, "交互输入"))
                    img_click_input = gr.Image(
                        type="pil",
                        label="上传图片",
                        interactive=True,
                        height=450
                    )

                    with gr.Row():
                        img_click_clear = gr.Button("清空点位并重置", variant="primary")

                    st_click_points = gr.State([])
                    st_click_labels = gr.State([])

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_OUTPUT, "结果预览"))
                    img_click_output = gr.Image(
                        type="pil",
                        label="分割预览",
                        height=450,
                        interactive=False
                    )

                    gr.HTML(html_status_indicator(
                        "流水线：点击坐标 → 分割检测模型提示编码 → mask 预测叠加。"
                    ))

            img_click_input.select(
                fn=image_click_handler,
                inputs=[img_click_input, st_click_points, st_click_labels],
                outputs=[img_click_output, st_click_points, st_click_labels]
            )

            img_click_clear.click(
                fn=lambda: (None, [], []),
                outputs=[img_click_output, st_click_points, st_click_labels]
            )

        with gr.Tab("图像问答"):
            gr.HTML(html_tab_intro(
                SVG_TEXT,
                "图像问答 —— 抠图 + 多模态对话",
                "上传一张图和一句提示词，分割检测模型会检测出所有候选区域并裁剪出来。在 Gallery 里点选一张，再针对它跟多模态模型聊天提问。",
                "多模态对话已就绪。",
            ))

            qa_crops_state = gr.State([])
            qa_selected_crop = gr.State(None)
            qa_selected_idx = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(html_card_label(SVG_IMAGE, "输入"))
                    qa_image_input = gr.Image(type="pil", label="上传图片", height=320)
                    qa_prompt_input = gr.Textbox(
                        label="检测提示词",
                        placeholder="例如：人、人脸、汽车",
                        lines=2,
                    )
                    with gr.Accordion("高级设置", open=False):
                        qa_conf_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.45, step=0.05,
                            label="分割检测置信度阈值",
                        )
                    qa_detect_btn = gr.Button("检测并抠图", variant="primary")
                    qa_status = gr.Textbox(label="状态", interactive=False)

                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[qa_image_input, qa_prompt_input, qa_conf_slider],
                        label="示例",
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_DETECT, "候选抠图"))
                    qa_gallery = gr.Gallery(
                        label="点选其中一张开始对话",
                        columns=3,
                        height=520,
                        object_fit="contain",
                        show_label=True,
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_OUTPUT, "当前选中"))
                    qa_selected_preview = gr.Image(
                        label="当前选中子图",
                        type="pil",
                        height=240,
                        interactive=False,
                    )
                    gr.HTML(html_section_heading(SVG_TEXT, "与多模态模型对话"))
                    qa_chatbot = gr.Chatbot(
                        label="多模态对话",
                        height=320,
                    )
                    with gr.Row():
                        qa_message = gr.Textbox(
                            placeholder="对选中的子图问点什么……",
                            show_label=False,
                            scale=4,
                        )
                        qa_send = gr.Button("发送", variant="primary", scale=1)
                    qa_clear = gr.Button("清空对话", variant="secondary")

            qa_detect_btn.click(
                fn=run_qa_detect_and_crop,
                inputs=[qa_image_input, qa_prompt_input, qa_conf_slider],
                outputs=[qa_gallery, qa_crops_state, qa_selected_preview, qa_chatbot, qa_status],
            )

            qa_gallery.select(
                fn=on_qa_gallery_select,
                inputs=[qa_crops_state],
                outputs=[qa_selected_preview, qa_selected_idx],
            ).then(
                fn=lambda crop: crop,
                inputs=[qa_selected_preview],
                outputs=[qa_selected_crop],
            )

            qa_send.click(
                fn=chat_with_qwen,
                inputs=[qa_message, qa_chatbot, qa_selected_crop],
                outputs=[qa_chatbot, qa_message],
            )
            qa_message.submit(
                fn=chat_with_qwen,
                inputs=[qa_message, qa_chatbot, qa_selected_crop],
                outputs=[qa_chatbot, qa_message],
            )

            qa_clear.click(
                fn=clear_qa_chat,
                outputs=[qa_chatbot, qa_message],
            )


if __name__ == "__main__":
    # servers.py writes output videos to SAM3_VIDEO_OUT_DIR (/var/sam3_data by
    # default). The path lives outside Gradio's default allowed roots, so we
    # have to add it explicitly or postprocess_data will refuse to move the
    # file into cache.
    _allowed = []
    _video_out = os.environ.get("SAM3_VIDEO_OUT_DIR")
    if _video_out:
        _allowed.append(os.path.abspath(_video_out))

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        css=css,
        mcp_server=True,
        theme=steel_blue_theme,
        show_error=True,
        ssr_mode=False,
        allowed_paths=_allowed or None,
    )
