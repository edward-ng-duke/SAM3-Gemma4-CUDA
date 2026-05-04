"""
Outbound clients for all model services this project talks to.

- SAM3:  HTTP to local servers.py (stateless service holding SAM3 weights).
- VLM:   OpenAI-compatible HTTP to a remote Qwen 3.6 27B endpoint.
         (Qwen is itself multimodal and replaces what used to be Gemma.)

Public entrypoints:
    SAM3:
        sam3_detect(image, prompt, conf_threshold=0.45)         -> list[DetectedRegion]
        sam3_track(image, points, labels)                       -> (overlay_pil, has_mask)
        sam3_video(video_path, prompt, frame_limit=60, ...)     -> (out_path, processed, masked)
        health()                                                 -> {...}
    VLM:
        vlm_generate(image, prompt, ...)                        -> str         (one-shot)
        vlm_generate_stream(image, prompt, ...)                 -> Iterator[str] (SSE-like stream)
        vlm_chat_stream(messages)                               -> Iterator[str] (multi-turn)

Configuration (env vars, all have defaults):
    SAM3_SERVER_URL  default "http://127.0.0.1:5050"
    QWEN_BASE_URL    default "http://10.0.0.94:5000/v1"
    QWEN_MODEL       default "qwen36-27b-fp8"
    QWEN_API_KEY     default "no_need"
"""

import ast
import base64
import io
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from typing import Iterator

import httpx
import numpy as np
from PIL import Image
from openai import OpenAI


SAM3_SERVER_URL = os.environ.get("SAM3_SERVER_URL", "http://127.0.0.1:5050")

QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://10.0.0.94:5000/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen36-27b-fp8")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "no_need")
QWEN_CLIENT = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)


class ServerError(RuntimeError):
    pass


@dataclass
class DetectedRegion:
    region_index: int
    bbox: list           # [x1, y1, x2, y2]
    score: float
    mask: np.ndarray     # HxW bool, may be None if return_masks=False


def _encode_pil_to_b64_png(image: Image.Image) -> str:
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_mask_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.array(img) > 0


def _decode_image_b64_png(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _post(path: str, payload: dict, timeout: float = 600.0) -> dict:
    url = f"{SAM3_SERVER_URL}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
    except httpx.RequestError as e:
        raise ServerError(f"Cannot reach SAM3 server at {url}: {e}") from e

    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise ServerError(f"SAM3 server {path} returned {r.status_code}: {detail}")
    return r.json()


def health() -> dict:
    url = f"{SAM3_SERVER_URL}/health"
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url)
        r.raise_for_status()
        return r.json()
    except httpx.RequestError as e:
        raise ServerError(f"Cannot reach SAM3 server at {url}: {e}") from e


def sam3_detect(
    image: Image.Image,
    prompt: str,
    conf_threshold: float = 0.45,
    mask_threshold: float = 0.5,
    return_masks: bool = True,
) -> list[DetectedRegion]:
    payload = {
        "image_b64": _encode_pil_to_b64_png(image),
        "prompt": prompt,
        "conf_threshold": float(conf_threshold),
        "mask_threshold": float(mask_threshold),
        "return_masks": bool(return_masks),
    }
    data = _post("/v1/sam3/detect", payload)
    regions: list[DetectedRegion] = []
    for r in data.get("regions", []):
        mask_b64 = r.get("mask_b64")
        mask = _decode_mask_b64_png(mask_b64) if mask_b64 else None
        regions.append(DetectedRegion(
            region_index=int(r["region_index"]),
            bbox=[int(v) for v in r["bbox"]],
            score=float(r["score"]),
            mask=mask,
        ))
    return regions


def sam3_track(
    image: Image.Image,
    points: list,
    labels: list,
) -> tuple[Image.Image, bool]:
    payload = {
        "image_b64": _encode_pil_to_b64_png(image),
        "points": [[int(x), int(y)] for x, y in points],
        "labels": [int(v) for v in labels],
    }
    data = _post("/v1/sam3/track", payload)
    overlay = _decode_image_b64_png(data["overlay_image_b64"])
    return overlay, bool(data.get("has_mask", False))


def _stage_video_for_server(video_path: str) -> tuple[str, str | None]:
    """If client and server live in different containers, copy the input video
    into a directory both can read (the shared volume) and return that staged
    path. Returns (path_to_send, path_to_cleanup_or_None).

    Staging dir is `SAM3_VIDEO_STAGE_DIR`, falling back to `SAM3_VIDEO_OUT_DIR`.
    When neither is set we assume client and server share a filesystem and
    pass the path through unchanged.
    """
    abs_path = os.path.abspath(video_path)
    stage_dir = os.environ.get("SAM3_VIDEO_STAGE_DIR") or os.environ.get("SAM3_VIDEO_OUT_DIR")
    if not stage_dir:
        return abs_path, None

    stage_dir = os.path.abspath(stage_dir)
    if abs_path.startswith(stage_dir + os.sep):
        return abs_path, None

    inputs_dir = os.path.join(stage_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    suffix = os.path.splitext(abs_path)[1] or ".mp4"
    staged = os.path.join(inputs_dir, f"{uuid.uuid4().hex}{suffix}")
    shutil.copyfile(abs_path, staged)
    return staged, staged


def sam3_video(
    video_path: str,
    prompt: str,
    frame_limit: int = 60,
    time_limit: int = 60,
    render_mode: str = "annotated",
) -> tuple[str, int, int]:
    if not video_path or not os.path.isfile(video_path):
        raise ServerError(f"input video not found: {video_path}")

    send_path, cleanup_path = _stage_video_for_server(video_path)
    payload = {
        "video_path": send_path,
        "prompt": prompt,
        "frame_limit": int(frame_limit),
        "time_limit": int(time_limit),
        "render_mode": render_mode,
    }
    try:
        data = _post("/v1/sam3/video", payload, timeout=1800.0)
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass
    return (
        data["output_video_path"],
        int(data.get("processed_frames", 0)),
        int(data.get("masked_frames", 0)),
    )


# ---------- VLM (Qwen 3.6 27B, multimodal) ----------
#
# Qwen 3 supports a "thinking" mode where chain-of-thought goes into a separate
# `reasoning` (or `reasoning_content`) field and the final answer in `content`.
# For JSON / structured output we keep thinking OFF (clean content). For free-text
# explanations and chat we leave thinking ON so the user sees the reasoning stream.


def _pil_to_data_url(image: Image.Image) -> str:
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _user_message_with_image(image: Image.Image, prompt: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _pil_to_data_url(image)}},
        ],
    }


def _extract_text_from_message(msg) -> str:
    """Pull final text from an OpenAI ChatCompletionMessage. Prefer content; fall back to reasoning."""
    if hasattr(msg, "model_dump"):
        msg_d = msg.model_dump()
    else:
        msg_d = msg
    content = (msg_d.get("content") or "").strip()
    if content:
        return content
    return (msg_d.get("reasoning") or msg_d.get("reasoning_content") or "").strip()


def _delta_text(delta) -> str:
    """Pull a streaming delta's text. Concatenates reasoning + content if both are present."""
    if hasattr(delta, "model_dump"):
        d = delta.model_dump()
    else:
        d = delta
    return (d.get("reasoning") or d.get("reasoning_content") or "") + (d.get("content") or "")


def vlm_generate(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 768,
    temperature: float = 0.2,
    enable_thinking: bool = False,
) -> str:
    """One-shot VLM call to Qwen. Defaults to thinking-off for clean structured output."""
    try:
        resp = QWEN_CLIENT.chat.completions.create(
            model=QWEN_MODEL,
            messages=[_user_message_with_image(image, prompt)],
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
        )
    except Exception as e:
        raise ServerError(f"VLM (Qwen) request failed: {e}") from e
    if not resp.choices:
        return ""
    return _extract_text_from_message(resp.choices[0].message)


def vlm_generate_stream(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    enable_thinking: bool = True,
) -> Iterator[str]:
    """Streaming VLM call to Qwen. Defaults to thinking-on so the user sees reasoning."""
    try:
        stream = QWEN_CLIENT.chat.completions.create(
            model=QWEN_MODEL,
            messages=[_user_message_with_image(image, prompt)],
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
            stream=True,
        )
    except Exception as e:
        raise ServerError(f"VLM (Qwen) stream request failed: {e}") from e
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            text = _delta_text(chunk.choices[0].delta)
            if text:
                yield text
    except Exception as e:
        raise ServerError(f"VLM (Qwen) stream error: {e}") from e


def vlm_chat_stream(messages: list, enable_thinking: bool = True) -> Iterator[str]:
    """Multi-turn streaming chat. Caller passes a full OpenAI-format messages list.

    Used by the Image Q&A tab — caller controls turn structure (image attached only
    to first turn, etc.). Yields text chunks as they arrive.
    """
    try:
        stream = QWEN_CLIENT.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
            stream=True,
        )
    except Exception as e:
        raise ServerError(f"VLM (Qwen) chat request failed: {e}") from e
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            text = _delta_text(chunk.choices[0].delta)
            if text:
                yield text
    except Exception as e:
        raise ServerError(f"VLM (Qwen) chat stream error: {e}") from e


# ---------- shared helper: parse JSON-ish blobs returned by VLMs ----------


def safe_parse_json(text: str):
    """Strip markdown fences and try json then ast.literal_eval; return {} on failure."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}
