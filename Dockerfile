# --------------------------------------------------------------------------- #
# Credit Default Prediction API – production image
#
# Build (inference image, fast):
#   docker build -t credit-default-api:latest .
#
# Build an image that can also *train* (MLflow + Great Expectations included):
#   docker build --build-arg REQUIREMENTS=requirements-train.txt -t credit-default-api:train .
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS base

ARG REQUIREMENTS=requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/credit_default_model.joblib

WORKDIR /app

# curl is used by the container health check below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
RUN pip install --no-cache-dir -r "${REQUIREMENTS}"

# Application code and the trained artifact (built by `dvc repro` / `train`).
COPY src ./src
COPY models ./models
COPY params.yaml ./
COPY pyproject.toml ./

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
