SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

UV         := /home/edward/.local/bin/uv
VENV       := .venv
PY         := $(VENV)/bin/python
PORT       := 7860
SAM3_PORT  := 5050
HOST       := 0.0.0.0

MODELS_SAM   := models/facebook/sam3
MODELS_GEMMA := models/google/gemma-4-E2B-it
DATA_DIR ?= /home/edward/research/lianzhong-project/data/反向教学教材
OUT_DIR  ?= /home/edward/research/lianzhong-project/data/outputs/反向教学教材

.PHONY: help dev dev-orchestrate run serve venv install models gemma-model download clean-venv clean-models status stop detect report \
        docker-build docker-up docker-down docker-logs docker-restart docker-clean docker-shell \
        deploy deploy-rebuild deploy-verify \
        eval-blade

help:
	@echo "Targets:"
	@echo "  make dev          — one-shot: stop + venv + deps + SAM3 model + start servers.py + run app.py"
	@echo "  make serve        — launch SAM3 stateless service (servers.py) on $(SAM3_PORT) (foreground)"
	@echo "  make run          — just launch app.py on $(PORT) (assumes servers.py is up)"
	@echo "  make venv         — create .venv via uv"
	@echo "  make install      — install torch (cu124) + requirements + spaces + gradio[mcp]"
	@echo "  make models       — download facebook/sam3 into ./models (Gemma is no longer needed)"
	@echo "  make gemma-model  — (optional, unused) download google/gemma-4-E2B-it"
	@echo "  make stop         — kill any app.py / servers.py / anything on $(PORT) or $(SAM3_PORT)"
	@echo "  make status       — show venv / models / port status"
	@echo "  make clean-venv   — remove .venv"
	@echo "  make clean-models — remove ./models (careful, re-downloads SAM3 ~10GB)"
	@echo "  make detect       — batch-process images in DATA_DIR (needs serve)"
	@echo "  make report       — generate HTML report from OUT_DIR/json/"
	@echo ""
	@echo "Docker (fully offline) targets:"
	@echo "  make deploy       — up + verify using existing sam3-cuda:latest image (no rebuild)"
	@echo "  make deploy-rebuild — force docker-build first, then up + verify (use after Dockerfile/requirements changes)"
	@echo "  make docker-build — build sam3-cuda:latest with SAM3 weights baked in (slow, ~15min)"
	@echo "  make docker-up    — docker compose up -d  (single 'sam3' container, two processes)"
	@echo "  make docker-down  — docker compose down"
	@echo "  make docker-logs  — docker compose logs -f"
	@echo "  make docker-restart — docker compose restart"
	@echo "  make docker-shell — open bash inside the sam3 container for debugging"
	@echo "  make docker-clean — docker compose down -v --rmi local (drops image)"

dev: stop venv install models dev-orchestrate

dev-orchestrate:
	@echo "[dev] starting servers.py in background (logs → /tmp/sam3-servers.log) ..."
	@if [ -f .env ]; then set -a; . <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env); set +a; fi
	@rm -f /tmp/sam3-servers.log
	@SAM3_PORT=$(SAM3_PORT) nohup $(PY) servers.py > /tmp/sam3-servers.log 2>&1 < /dev/null &
	@disown || true
	@echo "[dev] waiting for SAM3 service on $(SAM3_PORT) (up to 90s) ..."
	@for i in $$(seq 1 45); do \
		if curl -sf http://127.0.0.1:$(SAM3_PORT)/health 2>/dev/null | grep -q '"sam3_video":true'; then \
			echo "[dev] SAM3 service ready"; \
			break; \
		fi; \
		if grep -qE "Traceback|FATAL|CUDA out of memory" /tmp/sam3-servers.log 2>/dev/null; then \
			echo "[dev] servers.py failed to start:"; \
			tail -30 /tmp/sam3-servers.log; \
			exit 1; \
		fi; \
		sleep 2; \
	done
	@if ! curl -sf http://127.0.0.1:$(SAM3_PORT)/health 2>/dev/null | grep -q '"sam3_video":true'; then \
		echo "[dev] timed out waiting for SAM3"; tail -30 /tmp/sam3-servers.log; exit 1; \
	fi
	@echo "[dev] launching app.py on $(PORT)  (Ctrl+C to stop app; servers.py keeps running — use 'make stop' to fully clean up)"
	@if [ -f .env ]; then set -a; . <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env); set +a; fi; \
	$(PY) app.py
	@echo "[dev] app.py exited. servers.py still running on $(SAM3_PORT). Run 'make stop' to fully clean up."

venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(UV) venv --python 3.10 $(VENV)

install: venv
	$(UV) pip install --python $(PY) \
		--index-url https://download.pytorch.org/whl/cu124 \
		torch torchvision
	$(UV) pip install --python $(PY) \
		-r requirements.txt "gradio[mcp]" spaces

models: $(MODELS_SAM)/config.json

gemma-model: $(MODELS_GEMMA)/config.json

$(MODELS_SAM)/config.json:
	if [ -f .env ]; then set -a; . <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env); set +a; fi; \
	mkdir -p models; \
	/home/edward/miniconda3/bin/hf download facebook/sam3 --local-dir $(MODELS_SAM)

$(MODELS_GEMMA)/config.json:
	if [ -f .env ]; then set -a; . <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env); set +a; fi; \
	mkdir -p models; \
	/home/edward/miniconda3/bin/hf download google/gemma-4-E2B-it --local-dir $(MODELS_GEMMA)

download: models

run:
	if [ -f .env ]; then set -a; . <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env); set +a; fi; \
	$(PY) app.py

serve:
	if [ -f .env ]; then set -a; . <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env); set +a; fi; \
	SAM3_PORT=$(SAM3_PORT) $(PY) servers.py

detect:
	set -a; source .env; set +a; \
	$(PY) scripts/batch_detect_persons.py --input-dir "$(DATA_DIR)" --output-dir "$(OUT_DIR)"

report:
	$(PY) scripts/generate_report.py --output-dir "$(OUT_DIR)"

# ---------- docker (fully offline image) ----------
# We pass --env-file .env.docker to bypass the project-local .env which
# contains free-form notes that docker compose can't parse.
DC := docker compose --env-file .env.docker

docker-build:
	$(DC) build

docker-up:
	$(DC) up -d
	@echo "[docker] container started. Tail logs with 'make docker-logs'."
	@set -a; [ -f .env.docker ] && . ./.env.docker; set +a; \
	echo "[docker] Gradio: http://localhost:$${GRADIO_HOST_PORT:-17860}   SAM3 health: http://localhost:$${SAM3_HOST_PORT:-5050}/health"

docker-down:
	$(DC) down

docker-logs:
	$(DC) logs -f --tail=200

docker-restart:
	$(DC) restart

docker-shell:
	$(DC) exec sam3 bash

docker-clean:
	$(DC) down -v --rmi local

# Default deploy: assume sam3-cuda:latest is already built (it bakes ~6.5GB SAM3
# weights + CUDA base, so rebuilding is expensive). If the image is missing we
# fail loudly and tell the user to run `make deploy-rebuild` (or `make docker-build`).
# If you changed Dockerfile/requirements.txt, use `make deploy-rebuild`.
deploy:
	@if ! docker image inspect sam3-cuda:latest >/dev/null 2>&1; then \
	  echo "[deploy] sam3-cuda:latest not found locally."; \
	  echo "[deploy] Run 'make deploy-rebuild' to build it (slow, ~15min, pulls CUDA base)."; \
	  exit 1; \
	fi
	@echo "[deploy] using existing sam3-cuda:latest (no rebuild)"
	@$(MAKE) --no-print-directory docker-up
	@$(MAKE) --no-print-directory deploy-verify

deploy-rebuild: docker-build docker-up deploy-verify

deploy-verify:
	@echo "[deploy] waiting for SAM3 API on host (up to 120s) ..."
	@set -a; [ -f .env.docker ] && . ./.env.docker; set +a; \
	SAM3=$${SAM3_HOST_PORT:-5050}; UI=$${GRADIO_HOST_PORT:-17860}; \
	for i in $$(seq 1 60); do \
	  if curl -sf "http://127.0.0.1:$$SAM3/health" 2>/dev/null | grep -q '"sam3_video":true'; then \
	    echo "[deploy] SAM3 API ready at http://localhost:$$SAM3"; \
	    echo "[deploy] Gradio UI: http://localhost:$$UI"; \
	    echo "[deploy] SAM3 API: http://localhost:$$SAM3  (docs: docs/SAM3_API.md)"; \
	    exit 0; \
	  fi; \
	  sleep 2; \
	done; \
	echo "[deploy] SAM3 API NOT ready within 120s — run 'make docker-logs' to inspect"; \
	exit 1

stop:
	@echo "[stop] killing app.py / servers.py processes..."
	@pkill -f "python .*app\.py" 2>/dev/null; true
	@pkill -f "python .*servers\.py" 2>/dev/null; true
	@sleep 1
	@for P in $(PORT) $(SAM3_PORT); do \
		pids="$$(ss -tlnpH 2>/dev/null | awk -v p=$$P '$$4 ~ ":"p"$$"{print $$0}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)"; \
		if [ -n "$$pids" ]; then \
			echo "[stop] port $$P held by: $$pids — sending TERM"; \
			kill $$pids 2>/dev/null || true; \
			sleep 2; \
			pids="$$(ss -tlnpH 2>/dev/null | awk -v p=$$P '$$4 ~ ":"p"$$"{print $$0}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)"; \
			if [ -n "$$pids" ]; then echo "[stop] still up, sending KILL to $$pids"; kill -9 $$pids 2>/dev/null || true; sleep 1; fi; \
		fi; \
		if ss -tlnp 2>/dev/null | grep -q ":$$P "; then echo "[stop] still listening on $$P"; else echo "[stop] port $$P free"; fi; \
	done

status:
	@printf "venv:        %s\n" "$$( [ -x $(PY) ] && echo present || echo missing )"
	@printf "torch:       %s\n" "$$( $(PY) -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>/dev/null || echo missing )"
	@printf "SAM3:        %s\n" "$$( [ -f $(MODELS_SAM)/config.json ] && du -sh $(MODELS_SAM) | cut -f1 || echo missing )"
	@printf "Gemma4:      %s\n" "$$( [ -f $(MODELS_GEMMA)/config.json ] && echo "$$(du -sh $(MODELS_GEMMA) | cut -f1) (unused, can rm)" || echo 'missing (not needed)' )"
	@printf "app port:    %s\n" "$$( ss -tlnp 2>/dev/null | grep -q :$(PORT) && echo listening || echo free )"
	@printf "sam3 port:   %s\n" "$$( ss -tlnp 2>/dev/null | grep -q :$(SAM3_PORT) && echo listening || echo free )"
	@printf "Qwen VLM:    %s\n" "$$( curl -sf -m 3 http://10.0.0.94:5000/v1/models >/dev/null 2>&1 && echo reachable || echo unreachable )"

clean-venv:
	rm -rf $(VENV)

clean-models:
	rm -rf models

# ---------- blade defect evaluation ----------
# Usage: make eval-blade DATASET=<path> PREDICTOR=dummy OUT=<path>
PREDICTOR ?= dummy

eval-blade:
	@if [ -z "$(DATASET)" ] || [ -z "$(OUT)" ]; then \
		echo "Usage: make eval-blade DATASET=<path> PREDICTOR=dummy OUT=<path>"; \
		exit 2; \
	fi
	$(PY) -m scripts.blade --dataset "$(DATASET)" --predictor "$(PREDICTOR)" --out "$(OUT)"
