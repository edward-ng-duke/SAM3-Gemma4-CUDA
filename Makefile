SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

UV      := /home/edward/.local/bin/uv
VENV    := .venv
PY      := $(VENV)/bin/python
PORT    := 7860
HOST    := 0.0.0.0

MODELS_SAM   := models/facebook/sam3
MODELS_GEMMA := models/google/gemma-4-E2B-it
DATA_DIR ?= /home/edward/research/lianzhong-project/data/反向教学教材
OUT_DIR  ?= /home/edward/research/lianzhong-project/data/outputs/反向教学教材

.PHONY: help dev run venv install models download clean-venv clean-models status stop detect report

help:
	@echo "Targets:"
	@echo "  make dev       — stateless: stop anything on $(PORT) / any app.py, then venv + deps + models + launch"
	@echo "  make run       — just launch app.py (assumes env already set up)"
	@echo "  make venv      — create .venv via uv"
	@echo "  make install   — install torch (cu124) + requirements + spaces + gradio[mcp]"
	@echo "  make models    — download facebook/sam3 + google/gemma-4-E2B-it into ./models"
	@echo "  make stop      — kill any app.py / anything holding port $(PORT)"
	@echo "  make status    — show venv / models / port status"
	@echo "  make clean-venv     — remove .venv"
	@echo "  make clean-models   — remove ./models (careful, re-downloads 16GB)"
	@echo "  make detect    — batch-process images in DATA_DIR → JSONs + overlays in OUT_DIR"
	@echo "  make report    — generate HTML report from OUT_DIR/json/"

dev: stop venv install models run

venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(UV) venv --python 3.10 $(VENV)

install: venv
	$(UV) pip install --python $(PY) \
		--index-url https://download.pytorch.org/whl/cu124 \
		torch torchvision
	$(UV) pip install --python $(PY) \
		-r requirements.txt "gradio[mcp]" spaces

models: $(MODELS_SAM)/config.json $(MODELS_GEMMA)/config.json

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

detect:
	set -a; source .env; set +a; \
	$(PY) scripts/batch_detect_persons.py --input-dir "$(DATA_DIR)" --output-dir "$(OUT_DIR)"

report:
	$(PY) scripts/generate_report.py --output-dir "$(OUT_DIR)"

stop:
	@echo "[stop] killing app.py processes..."
	@pkill -f "python .*app\.py" 2>/dev/null; true
	@sleep 1
	@pids="$$(ss -tlnpH 2>/dev/null | awk '$$4 ~ /:$(PORT)$$/{print $$0}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)"; \
	if [ -n "$$pids" ]; then \
		echo "[stop] port $(PORT) held by: $$pids — sending TERM"; \
		kill $$pids 2>/dev/null || true; \
		sleep 2; \
		pids="$$(ss -tlnpH 2>/dev/null | awk '$$4 ~ /:$(PORT)$$/{print $$0}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)"; \
		if [ -n "$$pids" ]; then echo "[stop] still up, sending KILL to $$pids"; kill -9 $$pids 2>/dev/null || true; sleep 1; fi; \
	fi
	@ss -tlnp 2>/dev/null | grep -q ":$(PORT) " && echo "[stop] still listening on $(PORT)" || echo "[stop] port $(PORT) free"

status:
	@printf "venv:     %s\n" "$$( [ -x $(PY) ] && echo present || echo missing )"
	@printf "torch:    %s\n" "$$( $(PY) -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>/dev/null || echo missing )"
	@printf "SAM3:     %s\n" "$$( [ -f $(MODELS_SAM)/config.json ] && du -sh $(MODELS_SAM) | cut -f1 || echo missing )"
	@printf "Gemma4:   %s\n" "$$( [ -f $(MODELS_GEMMA)/config.json ] && du -sh $(MODELS_GEMMA) | cut -f1 || echo missing )"
	@printf "port:     %s\n" "$$( ss -tlnp 2>/dev/null | grep -q :$(PORT) && echo listening || echo free )"

clean-venv:
	rm -rf $(VENV)

clean-models:
	rm -rf models
