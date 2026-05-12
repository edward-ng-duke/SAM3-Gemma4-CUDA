#!/usr/bin/env python3
"""SAM3 server diagnostic probe.

Designed to be handed to the server-side team running the SAM3 deployment
at `http://10.0.0.93:5050` (or any other reachable host). Produces a single
self-contained text report covering the three things the demo team needs:

  1. The /health and /openapi.json (or /docs) response, to confirm what
     model/head is loaded and what the real request schema is.
  2. The HTTP 500 reproduction: which prompts crash the server, with
     response bodies attached (we hope for a Python traceback).
  3. The HTTP 200 + `regions=[]` reproduction: same prompts that should
     be hitting the open-vocab head but return empty, across multiple
     conf_threshold values, image sizes, and (optionally) the 4 WTBD
     samples shipped with the demo.

Usage (on the SAM3 server box, or any host that can reach it):

    # default endpoint = http://10.0.0.93:5050
    python3 sam3_diag.py

    # custom endpoint (e.g. from the SAM3 server itself, hit localhost)
    python3 sam3_diag.py --endpoint http://127.0.0.1:5050

    # also include the 4 WTBD samples if you have them in CWD/samples/
    python3 sam3_diag.py --include-wtbd

Dependencies:
  - Python 3.8+
  - stdlib only (urllib, base64, json, gzip)
  - Pillow OPTIONAL — only used to build a synthetic 8x8 test image;
    falls back to a hardcoded base64 PNG if Pillow is unavailable.

Output:
  - sam3_diag_<UTC-timestamp>.txt in CWD
  - Paste/upload that file back to the demo team.

This script is read-only against the server. It does not modify state.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# ---------------------------------------------------------------------------
# 8x8 white PNG. Precomputed so this script has zero non-stdlib deps in the
# minimal path. To regenerate:
#     from PIL import Image; import base64, io
#     b = io.BytesIO(); Image.new("RGB",(8,8),(255,255,255)).save(b,"PNG")
#     print(base64.b64encode(b.getvalue()).decode())
# ---------------------------------------------------------------------------
WHITE_8X8_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAGUlEQVR4nGP8//8/AwMDExMDEwMDEwAAAAD//wMABXgC"
    "AmJ7yiwAAAAASUVORK5CYII="
)


def _fmt_dur(seconds: float) -> str:
    return f"{seconds*1000:7.0f} ms"


def _http_get(url: str, timeout: float = 15.0) -> tuple[int, str, dict[str, str]]:
    """GET with no auth, return (status, body_text, headers). Never raises."""
    req = urllib.request.Request(url, method="GET")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body, dict(resp.headers)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return e.code, body, dict(e.headers or {})
    except Exception as e:
        return -1, f"<exception: {type(e).__name__}: {e}>", {}
    finally:
        _ = time.perf_counter() - t0


def _http_post_json(
    url: str, body: dict[str, Any], timeout: float = 60.0
) -> tuple[int, str, float]:
    """POST JSON, return (status, body_text, wall_seconds). Never raises."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            txt = resp.read().decode("utf-8", errors="replace")
            return resp.status, txt, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        txt = ""
        try:
            txt = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return e.code, txt, time.perf_counter() - t0
    except Exception as e:
        return -1, f"<exception: {type(e).__name__}: {e}>", time.perf_counter() - t0


def _png_b64_synthetic_8x8() -> str:
    try:
        from PIL import Image  # type: ignore
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return WHITE_8X8_PNG_B64


def _png_b64_file(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

# Prompts we've observed deterministically returning HTTP 500 on this server.
SUSPECT_500_PROMPTS = [
    "blade",
    "line",
    "black line",
    "split",
    "surface damage",
    "paint peeling",
    "rust",
]

# Prompts we've observed returning HTTP 200 but with empty regions.
SUSPECT_EMPTY_PROMPTS = [
    "crack",
    "corrosion",
    "craze",
]

# Domain-neutral / canary prompts.
CANARY_PROMPTS = [
    "object",
    "wind turbine",
]


def run_request(
    endpoint: str,
    image_b64: str,
    prompt: str,
    *,
    conf_threshold: float = 0.05,
    mask_threshold: float = 0.5,
    return_masks: bool = False,
    retries: int = 2,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Send detect request `retries+1` times. Capture deterministic vs flaky."""
    url = endpoint.rstrip("/") + "/v1/sam3/detect"
    attempts: list[dict[str, Any]] = []
    body = {
        "image_b64": image_b64,
        "prompt": prompt,
        "conf_threshold": conf_threshold,
        "mask_threshold": mask_threshold,
        "return_masks": return_masks,
    }
    for attempt in range(retries + 1):
        status, txt, wall = _http_post_json(url, body, timeout=timeout)
        parsed_regions = None
        max_score = None
        parse_err = None
        if status == 200:
            try:
                d = json.loads(txt)
                regs = d.get("regions", [])
                parsed_regions = len(regs)
                if regs:
                    max_score = max(float(r.get("score", 0.0)) for r in regs)
            except Exception as e:
                parse_err = f"{type(e).__name__}: {e}"
        attempts.append({
            "attempt": attempt,
            "status": status,
            "wall_seconds": round(wall, 3),
            "body_len": len(txt),
            "body_head": txt[:400],
            "regions": parsed_regions,
            "max_score": max_score,
            "parse_err": parse_err,
        })
        time.sleep(0.2)
    return {
        "prompt": prompt,
        "conf_threshold": conf_threshold,
        "image_b64_len": len(image_b64),
        "attempts": attempts,
    }


def section(out: list[str], title: str) -> None:
    out.append("")
    out.append("=" * 78)
    out.append(f" {title}")
    out.append("=" * 78)


def summarize_attempts(at: list[dict[str, Any]]) -> str:
    parts = []
    for a in at:
        if a["status"] == 200:
            parts.append(f"200 [{a['regions']}r max={a['max_score']}] {_fmt_dur(a['wall_seconds'])}")
        else:
            head = (a["body_head"] or "").splitlines()[0][:80] if a["body_head"] else "<empty>"
            parts.append(f"{a['status']} {_fmt_dur(a['wall_seconds'])} body={head!r}")
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--endpoint",
        default=os.environ.get("SAM3_ENDPOINT", "http://10.0.0.93:5050"),
        help="SAM3 base URL (no trailing slash). Default from $SAM3_ENDPOINT.",
    )
    p.add_argument("--retries", type=int, default=2, help="Per request retry count (default 2 → 3 attempts each).")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    p.add_argument(
        "--include-wtbd",
        action="store_true",
        help="Also probe with the 4 wtbd_*.png samples under ./samples/ if present.",
    )
    p.add_argument(
        "--wtbd-dir",
        default="samples",
        help="Directory containing wtbd_<class>.png samples (default ./samples).",
    )
    p.add_argument(
        "--conf-thresholds",
        type=str,
        default="0.45,0.20,0.05",
        help="Comma-separated conf_threshold values to sweep.",
    )
    p.add_argument("--out", default=None, help="Output report path (default sam3_diag_<UTC>.txt).")
    args = p.parse_args(argv)

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or f"sam3_diag_{ts}.txt"

    thresholds = [float(x) for x in args.conf_thresholds.split(",")]

    out: list[str] = []
    out.append("SAM3 SERVER DIAGNOSTIC REPORT")
    out.append("")
    out.append(f"  generated_utc:    {ts}")
    out.append(f"  endpoint:         {args.endpoint}")
    out.append(f"  retries_per_req:  {args.retries}")
    out.append(f"  thresholds:       {thresholds}")
    out.append(f"  include_wtbd:     {args.include_wtbd}")
    out.append(f"  python_version:   {sys.version.splitlines()[0]}")

    # ---------------------------------------------------------------- /health
    section(out, "1. GET /health")
    status, body, _ = _http_get(args.endpoint.rstrip("/") + "/health", timeout=args.timeout)
    out.append(f"HTTP {status}")
    out.append(body[:2000])

    # --------------------------------------------------------- /openapi.json
    section(out, "2. GET /openapi.json (the schema-of-record — most important!)")
    status, body, _ = _http_get(args.endpoint.rstrip("/") + "/openapi.json", timeout=args.timeout)
    out.append(f"HTTP {status}  body_len={len(body)}")
    if status == 200:
        try:
            d = json.loads(body)
            out.append(json.dumps(d, indent=2, ensure_ascii=False)[:20000])
        except Exception as e:
            out.append(f"<unparseable as json: {e}>")
            out.append(body[:8000])
    else:
        out.append(body[:8000])

    # Fallback: /docs
    section(out, "3. GET /docs (FastAPI Swagger UI, HTML — fallback if openapi.json missing)")
    status, body, _ = _http_get(args.endpoint.rstrip("/") + "/docs", timeout=args.timeout)
    out.append(f"HTTP {status}  body_len={len(body)}")
    out.append(body[:2000])

    # -------------------------------------------- Minimal 8×8 white PNG probe
    section(out, "4. Minimal 8×8 white PNG probe — known-bad and known-empty prompts")
    out.append("Goal: get the server traceback for HTTP 500 prompts (see /var/log/...")
    out.append("or `docker logs --tail 500 <container>` AT THE SAME TIME this script runs).")
    out.append("")
    tiny_b64 = _png_b64_synthetic_8x8()
    out.append(f"tiny image: 8x8 white PNG, base64_len={len(tiny_b64)}")
    out.append("")

    for prompt_set_name, prompts in [
        ("SUSPECT_500", SUSPECT_500_PROMPTS),
        ("SUSPECT_EMPTY", SUSPECT_EMPTY_PROMPTS),
        ("CANARY", CANARY_PROMPTS),
    ]:
        out.append(f"-- prompt set: {prompt_set_name} --")
        for prompt in prompts:
            for thr in thresholds:
                res = run_request(
                    args.endpoint, tiny_b64, prompt,
                    conf_threshold=thr,
                    retries=args.retries,
                    timeout=args.timeout,
                )
                out.append(f"  prompt={prompt!r:22s} conf={thr:.2f}  ->  {summarize_attempts(res['attempts'])}")
                # Dump full body for any non-200 with non-trivial content
                for a in res["attempts"]:
                    if a["status"] != 200 and len(a["body_head"]) > 0:
                        out.append(f"     [body for {prompt!r} attempt {a['attempt']} HTTP={a['status']}]")
                        out.append("       " + a["body_head"].replace("\n", "\n       "))
                        break

    # ----------------------------------------------- WTBD images (optional)
    if args.include_wtbd:
        section(out, "5. WTBD wind-turbine-blade samples (1024×1024, 6–10 m drone shots)")
        out.append("These are the real-world inputs failing in production. If they probe")
        out.append("differently than the 8×8 synthetic, it's a model+data scale issue and")
        out.append("we should add sliding-window / SAHI tile inference.")
        out.append("")
        wtbd_files = sorted(
            f for f in os.listdir(args.wtbd_dir)
            if f.startswith("wtbd_") and f.endswith(".png")
        ) if os.path.isdir(args.wtbd_dir) else []
        out.append(f"wtbd files found in {args.wtbd_dir}/: {wtbd_files}")
        for fname in wtbd_files:
            path = os.path.join(args.wtbd_dir, fname)
            b64 = _png_b64_file(path)
            if b64 is None:
                out.append(f"  ! could not read {path}")
                continue
            out.append("")
            out.append(f"-- image: {fname} ({len(b64)} base64 bytes) --")
            # Mix one HTTP-500 and one HTTP-200-empty prompt per image; at lowest threshold only
            for prompt in ["crack", "corrosion", "craze", "surface damage"]:
                res = run_request(
                    args.endpoint, b64, prompt,
                    conf_threshold=0.05,
                    retries=0,
                    timeout=args.timeout,
                )
                out.append(f"  prompt={prompt!r:22s} conf=0.05  ->  {summarize_attempts(res['attempts'])}")
    else:
        section(out, "5. WTBD samples (skipped — pass --include-wtbd to enable)")

    # ---------------------------------------------- What we'd love to also see
    section(out, "6. What we ALSO need from the server box (not gathered by this script)")
    out.append("""
Please paste the following alongside this report when sending back:

  (a) Process / container state
        ps -ef | grep -i sam3
        ls -la /<sam3-weights-dir>/
        sha256sum /<sam3-weights-dir>/*.pt

  (b) Recent server logs covering the moment this script ran (timestamp at
      the top of the report). Look for tracebacks on the HTTP 500 prompts.
        journalctl -u <sam3-service-name> -n 500 --no-pager
        # or
        docker logs <sam3-container> --tail 500
        # or
        tail -n 1000 /var/log/sam3/server.log /var/log/sam3/error.log

  (c) Server-side hardcoded threshold/postprocessing constants (grep is fine):
        grep -rnE '(score|conf)_thr|min_score|top_k|filter_by|relevance' \\
            /<sam3-server-src>/

  (d) Loaded model config (what head was mounted?):
        cat /<sam3-server-src>/config*.yaml
        # or
        cat /<sam3-service-launch-cmd>

  (e) [If easy] Raw model output for one WTBD image with --no-postprocess,
      top 20 candidates with raw scores. If the demo team can't reach this
      box, a small extra debug endpoint is the long-term fix:
        POST /v1/sam3/debug_detect  -> returns candidates pre-NMS-and-threshold

Send the report file + (a)-(e) outputs back to the demo team. Cheers.
""")

    # ---------------------------------------------------------- write file
    final = "\n".join(out) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final)
    print(f"wrote {out_path}  ({len(final)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
