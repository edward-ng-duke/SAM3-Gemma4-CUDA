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
PERSON_COLORS = [
    (255, 230, 0),    # yellow (from MASK_COLORS)
    (255, 99, 132),   # pink
    (54, 162, 235),   # blue
    (75, 192, 192),   # teal
    (153, 102, 255),  # purple
    (255, 159, 64),   # orange
    (46, 204, 113),   # green
    (231, 76, 60),    # red
    (26, 188, 156),   # turquoise
    (241, 196, 15),   # dark yellow
    (142, 68, 173),   # violet
    (52, 152, 219),   # sky blue
]


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
    regions = []
    per_prompt_counts = {}
    for prompt in prompts:
        got = detect_with_prompt(image_pil, prompt, conf_thresh, sam_model, sam_processor, device)
        per_prompt_counts[prompt] = len(got)
        regions.extend(got)
    for i, region in enumerate(regions):
        region["region_index"] = i
    return regions, per_prompt_counts


def cluster_persons_with_gemma(image_pil, regions):
    if len(regions) == 0:
        return {"num_persons": 0, "reasoning": "no regions to cluster", "persons": []}

    import torch
    from app import VL_MODEL, VL_PROCESSOR, build_vl_inputs, safe_parse_json

    regions_for_prompt = [{k: v for k, v in r.items() if k != "mask"} for r in regions]
    regions_json = json.dumps(regions_for_prompt, indent=2)

    instruction = (
        "You are given an image and a list of detected regions. Each region was produced by a\n"
        "segmentation model prompted with a body-part keyword (person, face, head, hands, arm,\n"
        "shoulder, torso, legs, feet). Multiple regions may correspond to the same physical\n"
        "person, and some regions may be false positives.\n"
        "\n"
        "Detected regions:\n"
        f"{regions_json}\n"
        "\n"
        "Task: Count how many DISTINCT PHYSICAL PERSONS are visible in the image, and group\n"
        "the regions by person.\n"
        "\n"
        "Rules:\n"
        "- Two regions belong to the same person if spatially consistent with one body.\n"
        "- Clear false positives should NOT be assigned to any person.\n"
        "- A person counts even if only partially visible (only a hand, only a leg, only a\n"
        "  shoulder — still counts as 1 person).\n"
        "- If no persons are visible, return num_persons: 0 and empty persons list.\n"
        "\n"
        "Return ONLY valid JSON (no markdown fences):\n"
        "{\"num_persons\": <int>, \"reasoning\": \"<1-3 sentences in Chinese>\",\n"
        " \"persons\": [{\"person_id\": <int>, \"region_indexes\": [<int>, ...]}]}"
    )

    inputs = build_vl_inputs(image_pil, instruction)
    with torch.inference_mode():
        gen_ids = VL_MODEL.generate(
            **inputs,
            max_new_tokens=768,
            use_cache=True,
            temperature=0.2,
            do_sample=False,
        )
    raw = VL_PROCESSOR.batch_decode(
        gen_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    try:
        parsed = safe_parse_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("parsed output is not a dict")

        try:
            num_persons = int(parsed.get("num_persons", -1))
        except (TypeError, ValueError):
            num_persons = -1

        reasoning_val = parsed.get("reasoning", "")
        if isinstance(reasoning_val, str):
            reasoning = reasoning_val
        else:
            try:
                reasoning = str(reasoning_val)
            except Exception:
                reasoning = ""

        persons_raw = parsed.get("persons", [])
        if not isinstance(persons_raw, list):
            persons_raw = []

        n_regions = len(regions)
        persons = []
        for p in persons_raw:
            if not isinstance(p, dict):
                continue
            try:
                pid = int(p.get("person_id", -1))
            except (TypeError, ValueError):
                continue
            ridx_raw = p.get("region_indexes", [])
            if not isinstance(ridx_raw, list):
                ridx_raw = []
            region_indexes = []
            for idx in ridx_raw:
                try:
                    idx_int = int(idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx_int < n_regions:
                    region_indexes.append(idx_int)
            persons.append({"person_id": pid, "region_indexes": region_indexes})

        return {"num_persons": num_persons, "reasoning": reasoning, "persons": persons}
    except Exception:
        return {
            "num_persons": -1,
            "reasoning": f"parse_error: {raw[:120]}",
            "persons": [],
        }


def visualize_persons(image_pil, regions, persons):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    # Build region_index -> (person_id, color) map
    index_to_person = {}
    for person in persons:
        try:
            pid = int(person.get("person_id", -1))
        except (TypeError, ValueError):
            pid = -1
        color = PERSON_COLORS[pid % len(PERSON_COLORS)] if pid >= 0 else (128, 128, 128)
        for ri in person.get("region_indexes", []):
            try:
                ri_int = int(ri)
            except (TypeError, ValueError):
                continue
            index_to_person[ri_int] = (pid, color)

    base = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # Paint masks
    for region in regions:
        ri = region.get("region_index", -1)
        pid, color = index_to_person.get(ri, (None, (128, 128, 128)))
        mask = region.get("mask")
        if mask is None:
            continue
        mask_arr = np.asarray(mask).astype(np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr.squeeze()
        if mask_arr.shape[:2] != (base.size[1], base.size[0]):
            mpil = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize(base.size, Image.NEAREST)
        else:
            mpil = Image.fromarray((mask_arr * 255).astype(np.uint8))
        fill = Image.new("RGBA", base.size, color + (0,))
        alpha = mpil.point(lambda v: int(0.45 * 255) if v > 0 else 0)
        fill.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, fill)

    composed = Image.alpha_composite(base, overlay).convert("RGB")

    # Draw bboxes + labels
    draw = ImageDraw.Draw(composed)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for region in regions:
        ri = region.get("region_index", -1)
        pid, color = index_to_person.get(ri, (None, (128, 128, 128)))
        x1, y1, x2, y2 = region["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"P{pid}/{region['prompt']}" if pid is not None else f"?/{region['prompt']}"
        tb = draw.textbbox((x1, max(0, y1 - 22)), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((tb[0], tb[1]), label, fill="black" if sum(color) > 400 else "white", font=font)

    return composed


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
