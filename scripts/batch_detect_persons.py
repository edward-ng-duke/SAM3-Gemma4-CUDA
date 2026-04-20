import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


PROMPTS = ["person", "face", "head", "hands", "arm", "shoulder", "torso", "legs", "feet"]
DEFAULT_CONF_THRESH = 0.45
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


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


def detect_with_prompt(image_pil, prompt, conf_thresh, sam_model, sam_processor, device):
    import torch
    import numpy as np

    model_inputs = sam_processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        sam_outputs = sam_model(**model_inputs)
    processed = sam_processor.post_process_instance_segmentation(
        sam_outputs,
        threshold=float(conf_thresh),
        mask_threshold=0.5,
        target_sizes=model_inputs.get("original_sizes").tolist(),
    )[0]
    raw_masks = processed.get("masks", None)
    raw_scores = processed.get("scores", None)
    if raw_masks is None or raw_scores is None or len(raw_scores) == 0:
        return []
    raw_masks_np = raw_masks.detach().cpu().numpy()
    raw_scores_np = raw_scores.detach().cpu().numpy()
    w, h = image_pil.size[0], image_pil.size[1]
    out = []
    for idx, mask in enumerate(raw_masks_np):
        if mask.ndim == 3:
            mask = np.squeeze(mask, axis=0)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        bbox = _clamp_box_xyxy([xs.min(), ys.min(), xs.max(), ys.max()], w, h)
        out.append({
            "prompt": prompt,
            "bbox": bbox,
            "score": float(raw_scores_np[idx]),
            "mask": mask,
        })
    return out


def collect_all_detections(image_pil, prompts, conf_thresh, sam_model, sam_processor, device):
    raise NotImplementedError("T3 will implement this")


def cluster_persons_with_gemma(image_pil, regions):
    raise NotImplementedError("T4 will implement this")


def visualize_persons(image_pil, regions, persons):
    raise NotImplementedError("T5 will implement this")


def process_image(image_path, out_dir, conf_thresh, sam_model, sam_processor, vl_model, vl_processor, device):
    raise NotImplementedError("T6 will implement this")


def main():
    parser = argparse.ArgumentParser(
        description="Batch detect persons in images using SAM3 + Gemma 4."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory of images to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write JSON + visualizations",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=DEFAULT_CONF_THRESH,
        help="SAM3 confidence threshold",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if JSON exists",
    )
    args = parser.parse_args()

    raise NotImplementedError("T7 will wire this up")


if __name__ == "__main__":
    main()
