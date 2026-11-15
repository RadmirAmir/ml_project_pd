# Convenience commands for local development (Windows: use `make` from Git Bash/WSL
# or run the underlying commands directly).

PYTHON ?= python
VENV ?= .venv
ACTIVATE = $(VENV)/Scripts/activate

.PHONY: help install install-dev data validate train experiments api test lint format dvc docker drift all clean

help:
	@echo "install      create the venv and install dev dependencies"
	@echo "data         run the DVC prepare stage (dvc repro prepare)"
	@echo "validate     run Great Expectations on data/raw"
	@echo "train        train + log to MLflow (dvc repro train)"
	@echo "experiments  run all MLflow experiments"
	@echo "api          start the FastAPI service on :8000"
	@echo "test         run pytest"
	@echo "lint         flake8 + black --check"
	@echo "format       black + isort"
	@echo "dvc          dvc repro (prepare + validate + train)"
	@echo "docker       build and run the Docker image"
	@echo "drift        compute the PSI drift report against a running API"

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/python -m pip install --upgrade pip
	$(VENV)/Scripts/python -m pip install -r requirements.txt

install-dev: install
	$(VENV)/Scripts/python -m pip install -r requirements-dev.txt

data:
	$(VENV)/Scripts/python -m dvc repro prepare

validate:
	$(VENV)/Scripts/python -m src.data.validation --input data/raw/UCI_Credit_Card.csv \
		--output data/expectations/validation_result.json

train:
	$(VENV)/Scripts/python -m src.models.train --params params.yaml

experiments:
	$(VENV)/Scripts/python -m src.models.experiment

api:
	$(VENV)/Scripts/python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

test:
	$(VENV)/Scripts/python -m pytest -q

lint:
	$(VENV)/Scripts/python -m flake8 src tests
	$(VENV)/Scripts/python -m black --check --diff src tests

format:
	$(VENV)/Scripts/python -m black src tests

dvc:
	$(VENV)/Scripts/python -m dvc repro

docker:
	powershell -ExecutionPolicy Bypass -File scripts/docker_build_run.ps1

drift:
	$(VENV)/Scripts/python -m src.monitoring.drift --api-url http://127.0.0.1:8000 --sample-size 1000

all: validate dvc test
