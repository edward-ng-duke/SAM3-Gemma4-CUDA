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


def detect_with_prompt(image_pil, prompt, conf_thresh, sam_model, sam_processor, device):
    raise NotImplementedError("T2 will implement this")


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
