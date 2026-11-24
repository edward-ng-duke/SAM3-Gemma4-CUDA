"""
SAM3 stateless HTTP service.

Run:
    python servers.py
    # or
    SAM3_PORT=5050 python servers.py

Endpoints:
    GET  /health
    POST /v1/sam3/detect   image + text prompt → bboxes + raw masks
    POST /v1/sam3/track    image + click points → server-rendered overlay image
    POST /v1/sam3/video    video file path + text prompt → server-rendered output video path

All callers (app.py, scripts/batch_detect_persons.py) talk to this service via
servers_client.py instead of loading SAM3 in-process. One model copy, many callers.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 必须在第一次 CUDA 分配之前设置（即 import torch 之前）：expandable segments 让
# empty_cache() 能真正把缓存还给驱动，而不是把峰值水位钉死在进程上。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import base64
import gc
import hashlib
import io
import logging
import logging.handlers
import tempfile
import threading
import traceback
from typing import Any, Optional

import cv2
import numpy as np
import torch
import transformers
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel
from safetensors.torch import load_file as load_safetensors
from transformers import (
    AutoConfig,
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
    Sam3VideoModel,
    Sam3VideoProcessor,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
SAM_MODEL_NAME = os.path.join(_HERE, "models", "facebook", "sam3")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VID_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

# Where /v1/sam3/video writes its output mp4. In single-host mode the default
# (system tmp) is fine. In docker-compose with two containers, set this to a
# shared volume mounted in both servers and app containers, so the path the
# server returns is readable by the Gradio container.
VIDEO_OUT_DIR = os.environ.get("SAM3_VIDEO_OUT_DIR", tempfile.gettempdir())
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

VIDEO_COLORS_BGR = [
    (181, 120, 31),
    (13, 128, 255),
    (43, 161, 43),
    (41, 38, 214),
    (189, 102, 148),
    (74, 87, 140),
]


_INFER_LOCK = threading.Lock()

SAM_MODEL: Optional[Sam3Model] = None
SAM_PROCESSOR: Optional[Sam3Processor] = None
TRK_MODEL: Optional[Sam3TrackerModel] = None
TRK_PROCESSOR: Optional[Sam3TrackerProcessor] = None
VID_MODEL: Optional[Sam3VideoModel] = None
VID_PROCESSOR: Optional[Sam3VideoProcessor] = None

# Populated at startup by _load_models(); cached so /health and /v1/sam3/config stay O(1).
MODEL_META: dict[str, Any] = {}
_WEIGHTS_SHA256: Optional[str] = None
_WEIGHTS_SHA256_COMPUTED = False

LOG_DIR = os.environ.get("SAM3_LOG_DIR", os.path.join(_HERE, "logs"))
LOG_PATH = os.path.join(LOG_DIR, "sam3_server.log")

logger = logging.getLogger("sam3.server")


def _setup_logging() -> None:
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=50 * 1024 * 1024, backupCount=5
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:
        # Log dir not writable (rare; e.g., read-only container fs). Stream-only logging
        # still works for journalctl / docker logs, so don't crash startup.
        print(f"[servers] WARN: rotating file log disabled ({e}); using stderr only")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False


_setup_logging()


def _is_cuda_oom(exc: BaseException) -> bool:
    """torch raises OutOfMemoryError on newer versions; older versions raise RuntimeError
    whose message contains 'out of memory'."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _weights_path() -> str:
    return os.path.join(SAM_MODEL_NAME, "model.safetensors")


def _get_weights_sha256() -> Optional[str]:
    """Hash model weights at most once; the file is several GB."""
    global _WEIGHTS_SHA256, _WEIGHTS_SHA256_COMPUTED
    if _WEIGHTS_SHA256_COMPUTED:
        return _WEIGHTS_SHA256

    weights_path = _weights_path()
    if os.environ.get("SAM3_SKIP_WEIGHTS_HASH", "0") != "1" and os.path.exists(weights_path):
        h = hashlib.sha256()
        with open(weights_path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        _WEIGHTS_SHA256 = h.hexdigest()

    _WEIGHTS_SHA256_COMPUTED = True
    return _WEIGHTS_SHA256


def _vision_encoder_shared() -> bool:
    encoders = []
    if SAM_MODEL is not None and hasattr(SAM_MODEL, "vision_encoder"):
        encoders.append(id(SAM_MODEL.vision_encoder))
    if TRK_MODEL is not None and hasattr(TRK_MODEL, "vision_encoder"):
        encoders.append(id(TRK_MODEL.vision_encoder))
    if VID_MODEL is not None and hasattr(VID_MODEL, "detector_model"):
        detector = VID_MODEL.detector_model
        if hasattr(detector, "vision_encoder"):
            encoders.append(id(detector.vision_encoder))
    return len(set(encoders)) <= 1


def _refresh_model_meta() -> None:
    global MODEL_META
    cuda_cap = None
    cuda_device_name = None
    if torch.cuda.is_available():
        try:
            cuda_cap = list(torch.cuda.get_device_capability())
            cuda_device_name = torch.cuda.get_device_name()
        except Exception:
            pass

    MODEL_META = {
        "model_path": SAM_MODEL_NAME,
        "weights_file": _weights_path(),
        "weights_sha256": _get_weights_sha256(),
        "sam3_image_class": type(SAM_MODEL).__name__ if SAM_MODEL is not None else None,
        "sam3_tracker_class": type(TRK_MODEL).__name__ if TRK_MODEL is not None else None,
        "sam3_video_class": type(VID_MODEL).__name__ if VID_MODEL is not None else None,
        "dtype": str(VID_DTYPE),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_capability": cuda_cap,
        "cuda_device_name": cuda_device_name,
        "vision_encoder_shared": _vision_encoder_shared(),
        "startup_preload": os.environ.get("SAM3_PRELOAD", "detect"),
    }


def _preload_targets(raw: Optional[str] = None) -> set[str]:
    raw = (raw if raw is not None else os.environ.get("SAM3_PRELOAD", "detect")).strip().lower()
    if raw in ("", "detect", "image", "sam3_image"):
        return {"image"}
    if raw in ("0", "false", "none", "off"):
        return set()
    if raw in ("1", "true", "all"):
        return {"image", "tracker", "video"}

    targets: set[str] = set()
    for token in raw.replace("+", ",").split(","):
        token = token.strip()
        if token in ("detect", "image", "sam3_image"):
            targets.add("image")
        elif token in ("track", "tracker", "sam3_tracker"):
            targets.update({"image", "tracker"})
        elif token in ("video", "sam3_video"):
            targets.add("video")
        else:
            raise ValueError(
                "SAM3_PRELOAD must be one of detect, track, video, all, none, "
                f"or a comma-separated combination; got {raw!r}"
            )
    return targets


def _ensure_image_model() -> tuple[Sam3Model, Sam3Processor]:
    global SAM_MODEL, SAM_PROCESSOR
    if SAM_MODEL is None:
        if VID_MODEL is not None:
            SAM_MODEL = VID_MODEL.detector_model
        else:
            print(f"[servers] loading Sam3Model from {SAM_MODEL_NAME} ...")
            SAM_MODEL = Sam3Model.from_pretrained(SAM_MODEL_NAME, dtype=VID_DTYPE).to(DEVICE).eval()
    if SAM_PROCESSOR is None:
        SAM_PROCESSOR = Sam3Processor.from_pretrained(SAM_MODEL_NAME)
    _refresh_model_meta()
    return SAM_MODEL, SAM_PROCESSOR


def _ensure_video_model() -> tuple[Sam3VideoModel, Sam3VideoProcessor]:
    global SAM_MODEL, SAM_PROCESSOR, TRK_MODEL, TRK_PROCESSOR, VID_MODEL, VID_PROCESSOR
    if VID_MODEL is None:
        # If the service started in lightweight detect-only mode, drop the standalone
        # detector/tracker before upgrading so we do not retain two backbones.
        if SAM_MODEL is not None or TRK_MODEL is not None:
            print("[servers] upgrading to Sam3VideoModel; releasing standalone image/tracker models first ...")
            SAM_MODEL = None
            TRK_MODEL = None
            TRK_PROCESSOR = None
            gc.collect()
            _release_cached_vram(0)

        print(f"[servers] loading Sam3VideoModel from {SAM_MODEL_NAME} ...")
        VID_MODEL = Sam3VideoModel.from_pretrained(SAM_MODEL_NAME, dtype=VID_DTYPE).to(DEVICE).eval()
        VID_PROCESSOR = Sam3VideoProcessor.from_pretrained(SAM_MODEL_NAME)
        SAM_MODEL = VID_MODEL.detector_model
    if VID_PROCESSOR is None:
        VID_PROCESSOR = Sam3VideoProcessor.from_pretrained(SAM_MODEL_NAME)
    if SAM_PROCESSOR is None:
        SAM_PROCESSOR = Sam3Processor.from_pretrained(SAM_MODEL_NAME)
    _refresh_model_meta()
    return VID_MODEL, VID_PROCESSOR


def _ensure_tracker_model() -> tuple[Sam3TrackerModel, Sam3TrackerProcessor]:
    global TRK_MODEL, TRK_PROCESSOR
    if TRK_MODEL is None:
        sam_model, _ = _ensure_image_model()
        print("[servers] building Sam3TrackerModel with SHARED vision_encoder (no extra backbone in VRAM) ...")
        sam_cfg = AutoConfig.from_pretrained(SAM_MODEL_NAME)
        trk_cfg = sam_cfg.tracker_config if hasattr(sam_cfg, "tracker_config") else sam_cfg
        TRK_MODEL = Sam3TrackerModel(trk_cfg)
        del TRK_MODEL.vision_encoder
        TRK_MODEL.vision_encoder = sam_model.vision_encoder

        raw_sd = load_safetensors(_weights_path())
        tracker_sd = {
            k[len("tracker_model."):]: v
            for k, v in raw_sd.items()
            if k.startswith("tracker_model.")
        }
        missing, unexpected = TRK_MODEL.load_state_dict(tracker_sd, strict=False)
        vision_missing = [m for m in missing if m.startswith("vision_encoder.")]
        other_missing = [m for m in missing if not m.startswith("vision_encoder.")]
        print(
            f"[servers] tracker: loaded {len(tracker_sd)} weights, "
            f"{len(vision_missing)} vision_encoder.* missing (expected, shared), "
            f"{len(other_missing)} other missing, {len(unexpected)} unexpected"
        )
        assert not other_missing, f"unexpected missing keys: {other_missing[:5]}"
        del raw_sd, tracker_sd

        # tracker 的非 vision_encoder 部分搬上 device，参数统一成 VID_DTYPE：按 config
        # 构建的 TRK_MODEL 是 float32，而共享 vision_encoder 输出 bf16 特征，不转换的话
        # forward 在 neck 处报 "Input type (c10::BFloat16) and bias type (float) should
        # be the same"。两个约束：
        #   1) 只转参数、不转 buffer —— 与 from_pretrained(dtype=...) 语义一致，
        #      RoPE cos/sin 等 buffer 必须保持 fp32；
        #   2) 不能对整树 .to(dtype=...) —— 会递归把共享 vision_encoder 的 64 个 fp32
        #      RoPE buffer 降成 bf16，改变 /detect 的检测结果（实测同图 46→42）。
        TRK_MODEL = TRK_MODEL.to(DEVICE)
        for _name, _param in TRK_MODEL.named_parameters():
            if not _name.startswith("vision_encoder.") and _param.is_floating_point():
                _param.data = _param.data.to(VID_DTYPE)
        TRK_MODEL = TRK_MODEL.eval()
        gc.collect()
        _release_cached_vram(0)

    if TRK_PROCESSOR is None:
        TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained(SAM_MODEL_NAME)
    assert SAM_MODEL is None or id(SAM_MODEL.vision_encoder) == id(TRK_MODEL.vision_encoder), (
        "vision_encoder not shared between SAM and TRK"
    )
    _refresh_model_meta()
    return TRK_MODEL, TRK_PROCESSOR


def _load_models(preload: Optional[str] = None) -> None:
    print(f"[servers] device={DEVICE}, vid_dtype={VID_DTYPE}")
    targets = _preload_targets(preload)
    if "video" in targets:
        _ensure_video_model()
    if "image" in targets:
        _ensure_image_model()
    if "tracker" in targets:
        _ensure_tracker_model()
    _refresh_model_meta()

    weights_sha256 = MODEL_META.get("weights_sha256")
    logger.info(
        "models ready: preload=%s loaded(image=%s tracker=%s video=%s), dtype=%s, weights_sha256=%s, transformers=%s",
        preload if preload is not None else os.environ.get("SAM3_PRELOAD", "detect"),
        SAM_MODEL is not None,
        TRK_MODEL is not None,
        VID_MODEL is not None,
        MODEL_META["dtype"],
        (weights_sha256[:12] + "...") if weights_sha256 else "skipped",
        transformers.__version__,
    )
    print(
        "[servers] SAM3 startup preload complete: "
        f"image={SAM_MODEL is not None}, tracker={TRK_MODEL is not None}, video={VID_MODEL is not None}."
    )


# ---------- helpers ----------


def _release_cached_vram(min_waste_bytes: int = 1 << 30) -> None:
    """请求结束后把"已缓存但未分配"的显存还给驱动。

    caching allocator 默认终生保留单次请求的峰值（nvidia-smi 上表现为"泄漏"），
    会挤压同卡其他进程。只在差额超过阈值时触发，小请求的热缓存不受影响。
    """
    if not torch.cuda.is_available():
        return
    if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() >= min_waste_bytes:
        torch.cuda.empty_cache()


def _filter_masks_to_cpu(outputs, conf_threshold: float) -> list[tuple[float, np.ndarray]]:
    """按分数过滤检测结果，把保留的低分辨率 mask（sigmoid 后 float32）搬到 CPU。

    取代 SAM_PROCESSOR.post_process_instance_segmentation(..., target_sizes=...)：
    那个实现会在 GPU 上把全部保留 mask 一次性插值到原图分辨率（float32）再以
    int64 二值化——大图多目标时单次请求瞬时分配 >10 GB，且峰值被 allocator
    永久驻留。这里只做它的过滤部分（数学完全一致），上采样推迟到 detect() 里
    在 CPU 上逐 mask 进行，显存/内存峰值都控制在每 mask 几十 MB。
    """
    scores = outputs.pred_logits[0].sigmoid()
    if getattr(outputs, "presence_logits", None) is not None:
        scores = scores * outputs.presence_logits[0].sigmoid()
    keep = scores > conf_threshold
    kept_scores = scores[keep].float().cpu().numpy()
    kept_masks = outputs.pred_masks[0][keep].sigmoid().float().cpu().numpy()
    return [(float(s), m) for s, m in zip(kept_scores, kept_masks)]


def _decode_b64_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_mask_b64_png(mask: np.ndarray) -> str:
    """Encode binary mask as 1-channel PNG (lossless, compact for binary)."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    elif mask.max() <= 1:
        mask = mask * 255
    img = Image.fromarray(mask, mode="L")
    return _encode_pil_to_b64_png(img)


def _clamp_box_xyxy(box, width, height):
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


def _extract_boxes_from_masks(mask_data, width, height):
    boxes = []
    if mask_data is None:
        return boxes
    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().float().cpu().numpy()
    mask_data = np.array(mask_data)
    if mask_data.ndim == 4:
        mask_data = mask_data[0]
    if mask_data.ndim == 3 and mask_data.shape[0] == 1:
        mask_data = mask_data[0]
    if mask_data.ndim == 2:
        mask_data = np.expand_dims(mask_data, axis=0)
    if mask_data.ndim != 3:
        return boxes

    for single_mask in mask_data:
        single_mask = np.array(single_mask)
        if single_mask.shape[:2] != (height, width):
            single_mask = cv2.resize(
                single_mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        binary = single_mask > 0
        ys, xs = np.where(binary)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append(None)
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes.append(_clamp_box_xyxy([x1, y1, x2, y2], width, height))
    return boxes


def _draw_video_masks_contours_and_boxes(frame_bgr, mask_data, prompt_text, scores=None):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if mask_data is None:
        return out
    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().float().cpu().numpy()
    mask_data = np.array(mask_data)
    if mask_data.ndim == 4:
        mask_data = mask_data.squeeze(1)
    if mask_data.ndim == 2:
        mask_data = np.expand_dims(mask_data, axis=0)
    if mask_data.ndim != 3 or len(mask_data) == 0:
        return out

    boxes = _extract_boxes_from_masks(mask_data, w, h)
    for i in range(len(mask_data)):
        color = VIDEO_COLORS_BGR[i % len(VIDEO_COLORS_BGR)]
        mask = mask_data[i]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        binary = mask > 0
        if not np.any(binary):
            continue
        for c in range(3):
            out[:, :, c] = np.where(
                binary,
                (out[:, :, c].astype(np.float32) * 0.55 + color[c] * 0.45).astype(np.uint8),
                out[:, :, c],
            )
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        box = boxes[i]
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if scores is not None and i < len(scores):
                try:
                    label = f"{prompt_text} {float(scores[i]):.2f}"
                except Exception:
                    label = f"{prompt_text} #{i}"
            else:
                label = f"{prompt_text} #{i}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y_top = max(y1 - th - 10, 0)
            y_bottom = max(y1, th + 10)
            cv2.rectangle(out, (x1, y_top), (x1 + tw + 6, y_bottom), color, -1)
            cv2.putText(out, label, (x1 + 3, max(y1 - 4, th + 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def _apply_mask_overlay(base_image: Image.Image, mask_data, opacity=0.5):
    if isinstance(base_image, np.ndarray):
        base_image = Image.fromarray(base_image)
    base_image = base_image.convert("RGBA")
    if mask_data is None:
        return base_image.convert("RGB")
    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().float().cpu().numpy()
    mask_data = np.array(mask_data).astype(np.uint8)
    if mask_data.ndim == 4:
        mask_data = mask_data[0]
    if mask_data.ndim == 3 and mask_data.shape[0] == 1:
        mask_data = mask_data[0]
    if mask_data.ndim == 2:
        mask_data = [mask_data]
        num_masks = 1
    elif mask_data.ndim == 3:
        num_masks = mask_data.shape[0]
    else:
        return base_image.convert("RGB")

    import matplotlib
    try:
        color_map = matplotlib.colormaps["rainbow"].resampled(max(num_masks, 1))
    except AttributeError:
        import matplotlib.cm as cm
        color_map = cm.get_cmap("rainbow").resampled(max(num_masks, 1))

    rgb_colors = [tuple(int(c * 255) for c in color_map(i)[:3]) for i in range(num_masks)]
    composite_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    for i, single_mask in enumerate(mask_data):
        mask_bitmap = Image.fromarray((single_mask * 255).astype(np.uint8))
        if mask_bitmap.size != base_image.size:
            mask_bitmap = mask_bitmap.resize(base_image.size, resample=Image.NEAREST)
        fill_color = rgb_colors[i]
        color_fill = Image.new("RGBA", base_image.size, fill_color + (0,))
        mask_alpha = mask_bitmap.point(lambda v: int(v * opacity) if v > 0 else 0)
        color_fill.putalpha(mask_alpha)
        composite_layer = Image.alpha_composite(composite_layer, color_fill)

    return Image.alpha_composite(base_image, composite_layer).convert("RGB")


def _draw_points_on_image(image, points):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    for pt in points:
        x, y = pt
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 255, 0), outline=(0, 0, 0), width=2)
    return draw_img


# ---------- request / response models ----------


class DetectRequest(BaseModel):
    image_b64: str
    prompt: str
    conf_threshold: float = 0.45
    mask_threshold: float = 0.5
    return_masks: bool = True


class Region(BaseModel):
    region_index: int
    bbox: list[int]
    score: float
    mask_b64: Optional[str] = None


class DetectResponse(BaseModel):
    width: int
    height: int
    regions: list[Region]


class TrackRequest(BaseModel):
    image_b64: str
    points: list[list[int]]   # [[x,y], ...] cumulative click coordinates
    labels: list[int]         # [1,1,...] foreground/background labels


class TrackResponse(BaseModel):
    overlay_image_b64: str    # PNG of input with mask overlay + points drawn
    has_mask: bool


class VideoRequest(BaseModel):
    video_path: str           # local path readable by the server
    prompt: str
    frame_limit: int = 60
    time_limit: int = 60      # not currently enforced server-side; reserved for future
    render_mode: str = "annotated"   # "annotated" | "mask"


class VideoResponse(BaseModel):
    output_video_path: str
    processed_frames: int
    masked_frames: int
    status: str


# ---------- FastAPI ----------


app = FastAPI(title="SAM3 stateless service")


@app.on_event("startup")
def on_startup():
    _load_models()


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so client gets a structured JSON body instead of opaque 500.
    HTTPException is handled by FastAPI itself and never reaches here."""
    tb = traceback.format_exc()
    if _is_cuda_oom(exc):
        # Try to free fragmentation so the next request has a chance.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.error("CUDA OOM on %s: %s\n%s", request.url.path, exc, tb)
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"cuda_oom: {exc}",
                "error_class": type(exc).__name__,
                "retriable": True,
            },
        )
    logger.error("unhandled exception on %s: %s\n%s", request.url.path, exc, tb)
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_class": type(exc).__name__,
            "traceback_tail": "\n".join(tb.splitlines()[-8:]),
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models_loaded": {
            "sam3_image": SAM_MODEL is not None,
            "sam3_tracker": TRK_MODEL is not None,
            "sam3_video": VID_MODEL is not None,
        },
        "model": MODEL_META,
    }


@app.get("/v1/sam3/config")
def sam3_config():
    """Exposes processor + tokenizer constants and DetectRequest defaults so client
    teams can confirm field names, image_size, tokenizer max_length, and the only
    two thresholds the server honors. Pair with /openapi.json for the full schema."""
    if SAM_PROCESSOR is None:
        raise HTTPException(503, "SAM3 image processor not loaded")
    image_proc = getattr(SAM_PROCESSOR, "image_processor", None)
    image_size = getattr(image_proc, "size", None) if image_proc is not None else None
    tokenizer = getattr(SAM_PROCESSOR, "tokenizer", None)
    return {
        "openapi_url": "/openapi.json",
        "docs_url": "/docs",
        "detect": {
            "endpoint": "/v1/sam3/detect",
            "method": "POST",
            "field_names": ["image_b64", "prompt", "conf_threshold", "mask_threshold", "return_masks"],
            "defaults": {
                "conf_threshold": DetectRequest.model_fields["conf_threshold"].default,
                "mask_threshold": DetectRequest.model_fields["mask_threshold"].default,
                "return_masks": DetectRequest.model_fields["return_masks"].default,
            },
            "prompt_is_singular_string": True,
            "thresholds_in_use": ["conf_threshold", "mask_threshold"],
            "no_hardcoded_min_score": True,
            "no_topk_filter": True,
            "no_relevance_filter": True,
        },
        "processor": {
            "image_processor_size": image_size,
            "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
            "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
            "tokenizer_truncates_silently": True,
        },
        "concurrency": {
            "infer_lock": True,
            "note": "All /detect, /track, /video inference is serialized behind a single lock.",
        },
        "model": MODEL_META,
    }


@app.post("/v1/sam3/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(400, "prompt is required")

    image = _decode_b64_image(req.image_b64)
    w, h = image.size

    with _INFER_LOCK:
        sam_model, sam_processor = _ensure_image_model()
        inputs = sam_processor(images=image, text=req.prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = sam_model(**inputs)
        scored_masks = _filter_masks_to_cpu(outputs, float(req.conf_threshold))
        del inputs, outputs
        _release_cached_vram()

    regions: list[Region] = []
    for score, mask_lowres in scored_masks:
        # 与库实现同序：先 bilinear 上采样 sigmoid 浮点 mask，再按 mask_threshold 二值化。
        mask = cv2.resize(mask_lowres, (w, h), interpolation=cv2.INTER_LINEAR) > float(req.mask_threshold)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        bbox = _clamp_box_xyxy([xs.min(), ys.min(), xs.max(), ys.max()], w, h)
        regions.append(Region(
            region_index=len(regions),
            bbox=bbox,
            score=score,
            mask_b64=_encode_mask_b64_png(mask) if req.return_masks else None,
        ))

    return DetectResponse(width=w, height=h, regions=regions)


@app.post("/v1/sam3/track", response_model=TrackResponse)
def track(req: TrackRequest):
    if len(req.points) == 0:
        raise HTTPException(400, "points must not be empty")
    if len(req.points) != len(req.labels):
        raise HTTPException(400, "points and labels length mismatch")

    image = _decode_b64_image(req.image_b64)

    input_points = [[req.points]]
    input_labels = [[req.labels]]

    with _INFER_LOCK:
        trk_model, trk_processor = _ensure_tracker_model()
        inputs = trk_processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = trk_model(**inputs, multimask_output=False)

        masks = trk_processor.post_process_masks(
            outputs.pred_masks.float().cpu(),
            inputs["original_sizes"],
            binarize=True,
        )[0]
        del inputs, outputs
        _release_cached_vram()

    if masks is None or len(masks) == 0:
        overlay = _draw_points_on_image(image, req.points)
        return TrackResponse(overlay_image_b64=_encode_pil_to_b64_png(overlay), has_mask=False)

    overlay = _apply_mask_overlay(image, masks[0])
    overlay = _draw_points_on_image(overlay, req.points)
    return TrackResponse(overlay_image_b64=_encode_pil_to_b64_png(overlay), has_mask=True)


@app.post("/v1/sam3/video", response_model=VideoResponse)
def video(req: VideoRequest):
    if not req.video_path or not os.path.isfile(req.video_path):
        raise HTTPException(400, f"video_path not readable: {req.video_path}")
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(400, "prompt is required")
    if req.render_mode not in ("annotated", "mask"):
        raise HTTPException(400, "render_mode must be 'annotated' or 'mask'")

    cap = cv2.VideoCapture(req.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if fps <= 0:
        fps = 24.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_rgb = []
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (req.frame_limit > 0 and counter >= req.frame_limit):
            break
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        counter += 1
    cap.release()

    if len(frames_rgb) == 0:
        raise HTTPException(400, "no readable frames in video")

    out_path = tempfile.mktemp(suffix=".mp4", prefix="sam3_video_", dir=VIDEO_OUT_DIR)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (vid_w, vid_h),
    )

    processed = 0
    masked = 0

    with _INFER_LOCK:
        vid_model, vid_processor = _ensure_video_model()
        session = vid_processor.init_video_session(
            video=frames_rgb,
            inference_device=DEVICE,
            dtype=VID_DTYPE,
        )
        session = vid_processor.add_text_prompt(inference_session=session, text=req.prompt)

        for model_out in vid_model.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=len(frames_rgb),
        ):
            post = vid_processor.postprocess_outputs(session, model_out)
            f_idx = model_out.frame_idx
            frame_rgb = frames_rgb[f_idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            masks = post.get("masks") if "masks" in post else None
            if masks is not None and hasattr(masks, "ndim") and masks.ndim == 4:
                masks = masks.squeeze(1)

            # Truthy check: tracker may return an empty (0, H, W) tensor for frames where
            # no instance has been propagated yet — those should NOT count as "masked".
            has_masks = masks is not None and getattr(masks, "shape", (0,))[0] > 0

            if req.render_mode == "annotated":
                if has_masks:
                    scores = post.get("scores", None)
                    out_bgr = _draw_video_masks_contours_and_boxes(frame_bgr, masks, req.prompt, scores=scores)
                    masked += 1
                else:
                    out_bgr = frame_bgr
            else:  # "mask"
                if has_masks:
                    pil = _apply_mask_overlay(Image.fromarray(frame_rgb), masks)
                    out_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    masked += 1
                else:
                    out_bgr = frame_bgr

            writer.write(out_bgr)
            processed += 1

        # session 持有全部帧张量 + 各 frame 的 vision feature 缓存（随帧数可达数 GB），
        # 在锁内显式释放，避免峰值驻留到下一个请求。
        del session
        _release_cached_vram()

    writer.release()

    return VideoResponse(
        output_video_path=out_path,
        processed_frames=processed,
        masked_frames=masked,
        status=f"render_mode={req.render_mode}; processed {processed}, masked {masked}",
    )


if __name__ == "__main__":
    port = int(os.environ.get("SAM3_PORT", "5050"))
    host = os.environ.get("SAM3_HOST", "0.0.0.0")
    print(f"[servers] starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
