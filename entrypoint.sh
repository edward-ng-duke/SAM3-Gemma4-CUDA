#!/usr/bin/env bash
# Single-container entrypoint: run servers.py in the background, wait until it
# is healthy on 127.0.0.1:$SAM3_PORT, then exec app.py in the foreground.
#
# tini is PID 1 (set in Dockerfile ENTRYPOINT); this script is its child and
# forwards SIGTERM/SIGINT to the background servers.py before letting tini
# reap everything.

set -euo pipefail

SAM3_PORT="${SAM3_PORT:-5050}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
SAM3_OUT_DIR="${SAM3_VIDEO_OUT_DIR:-/var/sam3_data}"

mkdir -p "${SAM3_OUT_DIR}/inputs"

echo "[entrypoint] starting servers.py on 127.0.0.1:${SAM3_PORT} ..."
python /app/servers.py &
SERVERS_PID=$!

trap 'echo "[entrypoint] received signal, terminating servers.py (pid ${SERVERS_PID}) ..."; kill -TERM "${SERVERS_PID}" 2>/dev/null || true; wait "${SERVERS_PID}" 2>/dev/null || true; exit 0' TERM INT

echo "[entrypoint] waiting for SAM3 health on 127.0.0.1:${SAM3_PORT} (up to 90s) ..."
for i in $(seq 1 45); do
    if curl -sf "http://127.0.0.1:${SAM3_PORT}/health" 2>/dev/null | grep -q '"sam3_video":true'; then
        echo "[entrypoint] SAM3 ready after ~$((i * 2))s"
        break
    fi
    if ! kill -0 "${SERVERS_PID}" 2>/dev/null; then
        echo "[entrypoint] servers.py exited prematurely (pid ${SERVERS_PID}); see logs above"
        exit 1
    fi
    sleep 2
done

if ! curl -sf "http://127.0.0.1:${SAM3_PORT}/health" 2>/dev/null | grep -q '"sam3_video":true'; then
    echo "[entrypoint] SAM3 not ready within 90s; see container logs above"
    kill -TERM "${SERVERS_PID}" 2>/dev/null || true
    exit 1
fi

echo "[entrypoint] launching Gradio app.py on 0.0.0.0:${GRADIO_PORT} ..."
exec python /app/app.py
