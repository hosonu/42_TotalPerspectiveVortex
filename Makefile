# Run from the repository root:  make <target>
UV      ?= uv
SUBJECT ?= 1
SRC     := src/bci
DEV     := dev
FLAKE8  := $(UV) run flake8
PEP8    := $(UV) run autopep8

.PHONY: help install sync run main verify test-lib cv explore extract-epochs clean lint format fmt check

help:
	@echo "42 Total Perspective Vortex — common targets"
	@echo ""
	@echo "  make install / sync   uv sync (deps + editable install of bci)"
	@echo "  make run / main       full evaluation: src/main.py"
	@echo "  make verify          dev/verify_setup.py"
	@echo "  make test-lib        dev/test_libraries.py"
	@echo "  make cv              dev/run_motor_imagery_cv.py (SUBJECT=$(SUBJECT))"
	@echo "  make explore         dev/explore_data.py"
	@echo "  make extract-epochs  dev/extract_epochs.py (SUBJECT=$(SUBJECT))"
	@echo "  make lint            flake8 on $(SRC), src/main.py, $(DEV)"
	@echo "  make format / fmt    autopep8 -i -r on $(SRC), src/main.py, $(DEV)"
	@echo "  make check           verify + lint"
	@echo "  make clean           remove __pycache__ under src and dev"

install sync:
	$(UV) sync

run main: sync
	$(UV) run python src/main.py

verify: sync
	$(UV) run python dev/verify_setup.py

test-lib: sync
	$(UV) run python dev/test_libraries.py

cv: sync
	$(UV) run python dev/run_motor_imagery_cv.py $(SUBJECT)

explore: sync
	$(UV) run python dev/explore_data.py $(SUBJECT)

extract-epochs: sync
	$(UV) run python dev/extract_epochs.py $(SUBJECT)

lint:
	$(FLAKE8) $(SRC) src/main.py $(DEV)

format fmt: sync
	$(PEP8) -i -r $(SRC) src/main.py $(DEV)

check: verify lint

clean:
	@find src $(DEV) -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null; true
	@find src $(DEV) -name '*.pyc' -delete 2>/dev/null; true
