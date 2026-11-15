"""Project-wide configuration: paths, constants and small helpers.

Every module resolves paths through this file so that the project can be
executed from any working directory (CLI, DVC stage, Docker container or a
pytest session).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Thread control
# --------------------------------------------------------------------------- #
# The hyper-parameter search runs several fits in parallel. Letting every fit
# also grab every BLAS/OpenMP thread oversubscribes the CPU and makes training
# an order of magnitude slower, so the project pins the low-level thread pools
# to one thread and parallelises at the joblib level instead.
for _thread_env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env_var, "1")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# <repo>/src/config.py -> <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPECTATIONS_DIR = DATA_DIR / "expectations"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = PROJECT_ROOT / "metrics"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

PARAMS_FILE = PROJECT_ROOT / "params.yaml"

RAW_DATA_FILE = RAW_DATA_DIR / "UCI_Credit_Card.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
FEATURE_METADATA_FILE = PROCESSED_DATA_DIR / "feature_metadata.json"

MODEL_FILE = MODELS_DIR / "credit_default_model.joblib"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"
METRICS_FILE = METRICS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
ID_COLUMN = "ID"
TARGET_COLUMN = "default.payment.next.month"

#: The 23 raw predictors shipped with the UCI dataset (in their original order).
RAW_FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

#: Columns that must be present in a raw dataframe before feature engineering.
REQUIRED_RAW_COLUMNS = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]

#: Every column of the UCI dataset is integer-valued; the ingestion layer casts
#: them back to ``int64`` because the CSV mixes ``10000`` and ``5e+05`` formats.
INTEGER_RAW_COLUMNS = [ID_COLUMN, *RAW_FEATURE_COLUMNS, TARGET_COLUMN]

# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_EXPERIMENT_NAME = "credit_default_pd_model"


def get_tracking_uri() -> str:
    """Return the MLflow tracking URI (override with ``MLFLOW_TRACKING_URI``)."""
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def get_experiment_name() -> str:
    """Return the MLflow experiment name (override with ``MLFLOW_EXPERIMENT_NAME``)."""
    return os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_directories() -> None:
    """Create output directories that are not tracked by git."""
    for directory in (
        PROCESSED_DATA_DIR,
        EXPECTATIONS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_params(path: Path | str | None = None) -> dict:
    """Load ``params.yaml`` as a plain dictionary."""
    import yaml

    params_path = Path(path) if path is not None else PARAMS_FILE
    if not params_path.exists():
        raise FileNotFoundError(f"parameters file not found: {params_path}")
    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_json(path: Path | str) -> dict:
    """Read a JSON document from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict, path: Path | str) -> Path:
    """Write ``payload`` as an indented UTF-8 JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return destination
