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

import base64
import io
import tempfile
import threading
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
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


def _load_models():
    global SAM_MODEL, SAM_PROCESSOR, TRK_MODEL, TRK_PROCESSOR, VID_MODEL, VID_PROCESSOR

    print(f"[servers] device={DEVICE}, vid_dtype={VID_DTYPE}")

    # 1) 一次性加载完整 Sam3VideoModel，覆盖 detector + tracker_video + tracker_neck
    #    所有 namespace（detector_model.*, tracker_model.*, tracker_neck.*）都被 load。
    print(f"[servers] loading Sam3VideoModel from {SAM_MODEL_NAME} (single source for all three endpoints) ...")
    VID_MODEL = Sam3VideoModel.from_pretrained(SAM_MODEL_NAME, dtype=VID_DTYPE).to(DEVICE).eval()
    VID_PROCESSOR = Sam3VideoProcessor.from_pretrained(SAM_MODEL_NAME)

    # 2) /v1/sam3/detect 用的 Sam3Model = VID_MODEL.detector_model（共享，不复制）
    SAM_MODEL = VID_MODEL.detector_model
    SAM_PROCESSOR = Sam3Processor.from_pretrained(SAM_MODEL_NAME)

    # 3) /v1/sam3/track 用的 Sam3TrackerModel：单独实例化（不同 forward 签名），
    #    但把它的 vision_encoder 替换成 VID_MODEL.detector_model.vision_encoder 的同一引用。
    print("[servers] building Sam3TrackerModel with SHARED vision_encoder (no extra backbone in VRAM) ...")
    sam_cfg = AutoConfig.from_pretrained(SAM_MODEL_NAME)
    trk_cfg = sam_cfg.tracker_config if hasattr(sam_cfg, "tracker_config") else sam_cfg
    TRK_MODEL = Sam3TrackerModel(trk_cfg)
    # 替换 vision_encoder 引用 —— 自动接入 nn.Module tree（PyTorch 标准行为）
    del TRK_MODEL.vision_encoder
    TRK_MODEL.vision_encoder = VID_MODEL.detector_model.vision_encoder

    # 只 load tracker_model.* 部分权重（不含 vision_encoder，因为它已经共享自 detector）
    raw_sd = load_safetensors(os.path.join(SAM_MODEL_NAME, "model.safetensors"))
    tracker_sd = {
        k[len("tracker_model."):]: v
        for k, v in raw_sd.items()
        if k.startswith("tracker_model.")
    }
    missing, unexpected = TRK_MODEL.load_state_dict(tracker_sd, strict=False)
    # missing 必含所有 vision_encoder.* 键（这是预期的，它们走共享引用了）
    vision_missing = [m for m in missing if m.startswith("vision_encoder.")]
    other_missing = [m for m in missing if not m.startswith("vision_encoder.")]
    print(
        f"[servers] tracker: loaded {len(tracker_sd)} weights, "
        f"{len(vision_missing)} vision_encoder.* missing (expected, shared), "
        f"{len(other_missing)} other missing, {len(unexpected)} unexpected"
    )
    assert not other_missing, f"unexpected missing keys: {other_missing[:5]}"

    # tracker 的非 vision_encoder 子模块需要搬上 device。.to(DEVICE) 会递归，
    # 但共享的 vision_encoder 已经在 DEVICE 上，.to() 对它是 no-op（不会重复分配）。
    TRK_MODEL = TRK_MODEL.to(DEVICE).eval()
    TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained(SAM_MODEL_NAME)

    # 4) 健全性自检：vision_encoder 内存里只有 1 份（id 相同）
    assert id(SAM_MODEL.vision_encoder) == id(TRK_MODEL.vision_encoder), "vision_encoder not shared between SAM and TRK"
    print(f"[servers] vision_encoder backbone shared across detect/track/video (single VRAM copy)")
    print("[servers] all SAM3 models ready.")


# ---------- helpers ----------


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
        mask_data = mask_data.detach().cpu().numpy()
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
        mask_data = mask_data.detach().cpu().numpy()
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
        mask_data = mask_data.detach().cpu().numpy()
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
    }


@app.post("/v1/sam3/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if SAM_MODEL is None or SAM_PROCESSOR is None:
        raise HTTPException(503, "SAM3 image model not loaded")
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(400, "prompt is required")

    image = _decode_b64_image(req.image_b64)
    w, h = image.size

    with _INFER_LOCK:
        inputs = SAM_PROCESSOR(images=image, text=req.prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = SAM_MODEL(**inputs)

        processed = SAM_PROCESSOR.post_process_instance_segmentation(
            outputs,
            threshold=float(req.conf_threshold),
            mask_threshold=float(req.mask_threshold),
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

    raw_masks = processed.get("masks")
    raw_scores = processed.get("scores")
    if raw_masks is None or raw_scores is None or len(raw_scores) == 0:
        return DetectResponse(width=w, height=h, regions=[])

    masks_np = raw_masks.detach().cpu().numpy()
    scores_np = raw_scores.detach().cpu().numpy()

    regions: list[Region] = []
    for idx, mask in enumerate(masks_np):
        if mask.ndim == 3:
            mask = np.squeeze(mask, axis=0)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        bbox = _clamp_box_xyxy([xs.min(), ys.min(), xs.max(), ys.max()], w, h)
        regions.append(Region(
            region_index=len(regions),
            bbox=bbox,
            score=float(scores_np[idx]),
            mask_b64=_encode_mask_b64_png(mask) if req.return_masks else None,
        ))

    return DetectResponse(width=w, height=h, regions=regions)


@app.post("/v1/sam3/track", response_model=TrackResponse)
def track(req: TrackRequest):
    if TRK_MODEL is None or TRK_PROCESSOR is None:
        raise HTTPException(503, "SAM3 tracker model not loaded")
    if len(req.points) == 0:
        raise HTTPException(400, "points must not be empty")
    if len(req.points) != len(req.labels):
        raise HTTPException(400, "points and labels length mismatch")

    image = _decode_b64_image(req.image_b64)

    input_points = [[req.points]]
    input_labels = [[req.labels]]

    with _INFER_LOCK:
        inputs = TRK_PROCESSOR(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = TRK_MODEL(**inputs, multimask_output=False)

        masks = TRK_PROCESSOR.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            binarize=True,
        )[0]

    if masks is None or len(masks) == 0:
        overlay = _draw_points_on_image(image, req.points)
        return TrackResponse(overlay_image_b64=_encode_pil_to_b64_png(overlay), has_mask=False)

    overlay = _apply_mask_overlay(image, masks[0])
    overlay = _draw_points_on_image(overlay, req.points)
    return TrackResponse(overlay_image_b64=_encode_pil_to_b64_png(overlay), has_mask=True)


@app.post("/v1/sam3/video", response_model=VideoResponse)
def video(req: VideoRequest):
    if VID_MODEL is None or VID_PROCESSOR is None:
        raise HTTPException(503, "SAM3 video model not loaded")
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
        session = VID_PROCESSOR.init_video_session(
            video=frames_rgb,
            inference_device=DEVICE,
            dtype=VID_DTYPE,
        )
        session = VID_PROCESSOR.add_text_prompt(inference_session=session, text=req.prompt)

        for model_out in VID_MODEL.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=len(frames_rgb),
        ):
            post = VID_PROCESSOR.postprocess_outputs(session, model_out)
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
