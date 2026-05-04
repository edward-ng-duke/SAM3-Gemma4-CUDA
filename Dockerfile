# syntax=docker/dockerfile:1.6
#
# Fully offline single-container image bundling:
#   - Python 3.10 venv with torch (cu124) + transformers + Gradio + FastAPI + OpenAI SDK
#   - SAM3 weights at /app/models/facebook/sam3
#   - tini as PID 1, /app/entrypoint.sh runs servers.py (5050) in background
#     then execs app.py (Gradio, 7860) in the foreground — one container, two
#     processes, talking over 127.0.0.1.
#
# Build:    docker build -t sam3-cuda:latest .
# Run:      docker compose up
# Once built, the image runs offline (no internet needed at runtime).
# The Qwen 3.6 VLM is an EXTERNAL service; configure with QWEN_BASE_URL env at runtime.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

# =====================================================================
# Stage 1: builder — install all Python deps into /opt/venv
# =====================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3 → python3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Self-contained venv at /opt/venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade 'pip>=24.0' wheel

# Torch first (it's huge; isolate this layer for cache reuse)
RUN pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0

# Project deps (editable list comes from requirements.txt; we add gradio[mcp] + spaces)
COPY requirements.txt /tmp/requirements.txt
RUN pip install \
        -r /tmp/requirements.txt \
        'gradio[mcp]' \
        spaces \
        fastapi \
        'uvicorn[standard]' \
        httpx \
        'openai>=1.40.0'

# =====================================================================
# Stage 2: runtime — slim image with only what's needed to run the services
# =====================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Bring the venv (with all wheels already installed) from builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Code (ordered: small files first, then bulky model files for layer cache hits)
COPY servers.py servers_client.py app.py requirements.txt /app/
COPY scripts/ /app/scripts/
COPY examples/ /app/examples/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# SAM3 model weights (≈6.5 GB) — pinned into the image so it runs offline.
# Gemma is intentionally NOT copied (the project no longer uses it).
COPY models/facebook/sam3/ /app/models/facebook/sam3/

# Ports: 5050 = SAM3 HTTP API (loopback inside container, optionally exposed),
#        7860 = Gradio UI (mapped to GRADIO_HOST_PORT on the host).
EXPOSE 5050 7860

# tini as PID 1 → forwards signals + reaps zombies. entrypoint.sh starts
# servers.py in the background, waits for /health, then execs app.py.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/entrypoint.sh"]
