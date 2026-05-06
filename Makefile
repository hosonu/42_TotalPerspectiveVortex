# ==============================================================================
# 42 Total Perspective Vortex - Makefile
# ==============================================================================

# Variables
LOCAL_BIN 	:= ./bin
VENV 		:= ./.venv
UV 			:= $(LOCAL_BIN)/uv
UV_PY 		:= ./uv_python
UV_CACHE 	:= ./uv_cache
PYTHON      := $(UV) run python
FLAKE8      := $(UV) run flake8
PEP8        := $(UV) run autopep8

# Virtual environment
export UV_PYTHON_INSTALL_DIR := $(UV_PY)
export UV_CACHE_DIR := $(UV_CACHE)

# Directories
SRC_DIR     := src
SCRIPTS_DIR := scripts
MODEL_FILE  := saved_bci_model.pkl

# Entry Point
CLI_PY      := $(SRC_DIR)/mybci.py
VIS_PY		:= $(SRC_DIR)/visualize.py

# Parameters (can be overridden: make train RUNS="3 7 11" SUBJECT=5)
SUBJECT     ?= 1
RUNS        ?= 4 8 12
BONUS ?= 0

ifeq ($(BONUS),1)
    BONUS_FLAG := --bonus
else
    BONUS_FLAG :=
endif

# Rules
.PHONY: all help install sync run main train predict lint format fmt check clean fclean re

all: install

help:
	@echo "Total Perspective Vortex — Commands"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install dependencies using uv"
	@echo ""
	@echo "  Execution:"
	@echo "    make run            Full evaluation for all subjects (no arguments)"
	@echo "    make train          Train model with RUNS='$(RUNS)'"
	@echo "    make predict        Playback simulation with RUNS='$(RUNS)'"
	@echo "    make visualize      Visualize raw/filtered EEG data with RUNS='$(RUNS)'"
	@echo ""
	@echo "  Development:"
	@echo "    make lint           Run flake8 on src and scripts"
	@echo "    make fmt            Run autopep8 formatter"
	@echo "    make check          Run verify_setup and lint"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean          Remove python cache files"
	@echo "    make fclean         Complete cleanup including .venv and saved model"
	@echo "    make re             Full reinstallation"

$(UV):
	@echo "Downloading uv locally into ./bin..."
	@mkdir -p $(LOCAL_BIN)
	curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$(shell pwd)/bin" INSTALLER_NO_MODIFY_PATH=1 UV_NO_MODIFY_PATH=1 sh

# Dependency Management
pre:
	@echo "Creating directories in this project directory..."
	@mkdir -p $(UV_PY)
	@mkdir -p $(UV_CACHE)

sync: pre $(UV)
	@echo "Syncing environment with uv.lock..."
	$(UV) sync

install: sync
	@echo "Done! Environment is ready."

# Main Tasks
run main: sync
	@echo "==> Running bulk evaluation on all subjects..."
	$(PYTHON) $(CLI_PY) $(BONUS_FLAG)

train: sync
	@echo "==> Training model..."
	$(PYTHON) $(CLI_PY) $(RUNS) train --subject $(SUBJECT) $(BONUS_FLAG)

predict: sync
	@echo "==> Running playback simulation..."
	$(PYTHON) $(CLI_PY) $(RUNS) predict --subject $(SUBJECT) $(BONUS_FLAG)

visualize: sync
	@echo "==> Visualizing raw and filtered EEG data..."
	$(PYTHON) $(VIS_PY) $(RUNS) --subject $(SUBJECT)

bonus:
	@echo "==> Evaluating bonus pipeline on all subjects..."
	$(PYTHON) $(CLI_PY) --bonus

# Development Tools
lint:
	$(FLAKE8) $(SRC_DIR) $(SCRIPTS_DIR)

format fmt:
	$(PEP8) -i -r $(SRC_DIR) $(SCRIPTS_DIR)

check: sync lint
	$(PYTHON) $(SCRIPTS_DIR)/verify_setup.py

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

fclean: clean
	rm -rf .venv
	rm -f $(MODEL_FILE)

re: fclean all