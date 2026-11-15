# Convenience commands for local development.
# Windows PowerShell: run the commands directly (see README).
# Git Bash / WSL: `make` works with these targets (forward-slash paths).

PYTHON ?= python
VENV ?= .venv
VENV_PY = $(VENV)/Scripts/python.exe

.PHONY: help install install-dev data validate train experiments api test lint format dvc docker drift all clean

help:
	@echo "install      create the venv and install inference dependencies"
	@echo "install-dev  install dev dependencies (training + tests + dvc + lint)"
	@echo "data         run the DVC prepare stage (dvc repro prepare)"
	@echo "validate     run Great Expectations on data/raw"
	@echo "train        train + log to MLflow (dvc repro train)"
	@echo "experiments  run all 6 MLflow experiments"
	@echo "api          start the FastAPI service on :8000"
	@echo "test         run pytest"
	@echo "lint         flake8 + black --check"
	@echo "format       black + isort"
	@echo "dvc          dvc repro (prepare + validate + train)"
	@echo "docker       build and run the Docker image"
	@echo "drift        compute the PSI drift report against a running API"
	@echo "presentation rebuild docs/presentation.pptx from the current artifacts"

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

install-dev: install
	$(VENV_PY) -m pip install -r requirements-dev.txt

data:
	$(VENV_PY) -m dvc repro prepare

validate:
	$(VENV_PY) -m src.data.validation --input data/raw/UCI_Credit_Card.csv \
		--output data/expectations/validation_result.json

train:
	$(VENV_PY) -m src.models.train --params params.yaml

experiments:
	$(VENV_PY) -m src.models.experiment

api:
	$(VENV_PY) -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

test:
	$(VENV_PY) -m pytest -q

lint:
	$(VENV_PY) -m flake8 src tests scripts
	$(VENV_PY) -m black --check src tests scripts

format:
	$(VENV_PY) -m black src tests scripts

dvc:
	$(VENV_PY) -m dvc repro

docker:
	powershell -ExecutionPolicy Bypass -File scripts/docker_build_run.ps1

drift:
	$(VENV_PY) -m src.monitoring.drift --api-url http://127.0.0.1:8000 --sample-size 1000

presentation:
	$(VENV_PY) scripts/build_presentation.py

all: validate dvc test
