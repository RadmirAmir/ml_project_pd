"""FastAPI scoring service for the credit-default (PD) model.

Run locally::

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Endpoints
---------
``GET  /``             – human friendly banner;
``GET  /health``       – liveness probe (also reports whether a model is loaded);
``GET  /model/info``   – metadata of the loaded model;
``POST /predict``      – single client scoring;
``POST /predict/batch``– batch scoring (up to 5000 clients).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClientData,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import MODEL_FILE
from src.models.predict import CreditDefaultScorer, ModelNotFoundError

LOGGER = logging.getLogger(__name__)

#: Model path can be overridden with ``MODEL_PATH`` (used by Docker Compose and tests).
DEFAULT_MODEL_PATH = MODEL_FILE

_scorer: Optional[CreditDefaultScorer] = None


def model_path() -> Path:
    """Return the configured model bundle path (read on every call)."""
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def get_scorer() -> CreditDefaultScorer:
    """Return the process-wide scorer, raising HTTP 503 when it is unavailable."""
    if _scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"model is not loaded (expected bundle at {model_path()}). "
                "Train the model first: `python -m src.models.train`."
            ),
        )
    return _scorer


def load_scorer(path: Path | str | None = None, strict: bool = False) -> Optional[CreditDefaultScorer]:
    """Load the model bundle into the module-level cache."""
    global _scorer
    bundle_path = Path(path) if path is not None else model_path()
    try:
        _scorer = CreditDefaultScorer.load(bundle_path)
        LOGGER.info("model loaded from %s (threshold=%.2f)", bundle_path, _scorer.threshold)
    except (ModelNotFoundError, OSError, ValueError) as error:
        _scorer = None
        LOGGER.error("could not load the model: %s", error)
        if strict:
            raise
    return _scorer


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once on startup (and release it on shutdown)."""
    global _scorer
    load_scorer(model_path())
    yield
    _scorer = None


app = FastAPI(
    title="Credit Default Prediction API",
    description="PD scoring service for credit card clients (UCI Default of Credit Card Clients).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def read_root() -> dict:
    """Small banner so the service is easy to smoke-test from a browser."""
    return {
        "message": "Credit Default Prediction API is alive!",
        "docs": "/docs",
        "predict_endpoint": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe used by Docker / Kubernetes."""
    scorer = _scorer
    return HealthResponse(
        status="ok" if scorer is not None else "degraded",
        model_loaded=scorer is not None,
        model_name=(scorer.metadata.get("model_name") if scorer else None),
        threshold=(scorer.threshold if scorer else None),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    """Return the metadata stored in the model bundle."""
    scorer = get_scorer()
    metadata = scorer.metadata
    return ModelInfoResponse(
        model_name=metadata.get("model_name"),
        threshold=scorer.threshold,
        trained_at=metadata.get("trained_at"),
        dataset_hash=metadata.get("dataset_hash"),
        sklearn_version=metadata.get("sklearn_version"),
        features={
            "numeric": list(scorer.numeric_features),
            "categorical": list(scorer.categorical_features),
        },
        test_metrics=metadata.get("test_metrics", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(client: ClientData) -> PredictionResponse:
    """Score a single client and return the class and the probability of default."""
    scorer = get_scorer()
    try:
        result = scorer.score_records([client.to_feature_dict()])[0]
    except ValueError as error:  # pragma: no cover - defensive
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PredictionResponse(**result, model_name=scorer.metadata.get("model_name"))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["scoring"])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score several clients in one call."""
    scorer = get_scorer()
    records = [record.to_feature_dict() for record in payload.records]
    predictions = [
        PredictionResponse(**item, model_name=scorer.metadata.get("model_name"))
        for item in scorer.score_records(records)
    ]
    probabilities = [item.default_probability for item in predictions]
    return BatchPredictionResponse(
        predictions=predictions,
        n_records=len(predictions),
        mean_probability=round(sum(probabilities) / len(probabilities), 6),
        n_flagged=sum(item.default_prediction for item in predictions),
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(_request, error: ModelNotFoundError) -> JSONResponse:
    """Return 503 instead of a 500 when the artifact is missing."""
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(error)})
