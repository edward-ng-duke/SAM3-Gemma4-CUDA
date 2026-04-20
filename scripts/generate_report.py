import argparse
import json
import os
from pathlib import Path


def load_all_jsons(out_dir):
    raise NotImplementedError("T10 will implement this")


def write_summary(items, out_dir):
    raise NotImplementedError("T10 will implement this")


def render_html(items, out_dir):
    raise NotImplementedError("T11 will implement this")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from batch detection JSONs.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory containing json/ subdirectory (same as batch_detect_persons --output-dir)")
    args = parser.parse_args()
    raise NotImplementedError("T13 will wire this up")


if __name__ == "__main__":
    main()
