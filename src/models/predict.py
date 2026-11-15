"""Scoring utilities shared by the API, the CLI and the monitoring script.

The saved artifact is a *bundle* (a plain dictionary) rather than a bare
estimator: it keeps the fitted sklearn pipeline together with the feature
schema, the decision threshold and the metrics, so a service can start scoring
without any extra configuration file.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_FILE, RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.features.build_features import build_features, build_inference_frame, get_feature_lists

LOGGER = logging.getLogger(__name__)

RISK_BANDS = (
    (0.00, 0.10, "low"),
    (0.10, 0.25, "medium"),
    (0.25, 0.50, "high"),
    (0.50, 1.01, "critical"),
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the serialized model bundle is missing."""


class CreditDefaultScorer:
    """Thin wrapper around the persisted pipeline bundle.

    Examples
    --------
    >>> scorer = CreditDefaultScorer.load()          # doctest: +SKIP
    >>> scorer.score_records([{"LIMIT_BAL": 20000}])  # doctest: +SKIP
    """

    def __init__(self, model, threshold: float = 0.5, metadata: dict | None = None):
        self.model = model
        self.threshold = float(threshold)
        self.metadata = metadata or {}
        self.numeric_features, self.categorical_features = get_feature_lists()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: Path | str | None = None) -> "CreditDefaultScorer":
        """Load the joblib bundle from ``path`` (defaults to ``models/``)."""
        bundle_path = Path(path or MODEL_FILE)
        if not bundle_path.exists():
            raise ModelNotFoundError(f"model bundle not found: {bundle_path}. Run the `train` stage first.")
        bundle = joblib.load(bundle_path)
        if isinstance(bundle, dict) and "model" in bundle:
            metadata = {key: value for key, value in bundle.items() if key != "model"}
            return cls(bundle["model"], threshold=bundle.get("threshold", 0.5), metadata=metadata)
        # Backwards compatibility: a bare estimator.
        return cls(bundle, threshold=0.5)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _model_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a model-ready frame from raw *or* already-featurised input."""
        columns = self.numeric_features + self.categorical_features
        if all(column in data.columns for column in columns):
            return data[columns].copy()
        missing_raw = [column for column in RAW_FEATURE_COLUMNS if column not in data.columns]
        if missing_raw:
            raise ValueError("input is missing raw columns: " + ", ".join(missing_raw))
        return build_features(data)[columns].copy()

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Return the probability of default for every row of ``data``."""
        if data.empty:
            raise ValueError("cannot score an empty dataframe")
        matrix = self._model_matrix(data)
        return np.asarray(self.model.predict_proba(matrix))[:, 1]

    def risk_band(self, probability: float) -> str:
        """Map a PD to a coarse risk band used by the business."""
        for lower, upper, label in RISK_BANDS:
            if lower <= probability < upper:
                return label
        return RISK_BANDS[-1][2]

    def score_records(self, records: list[dict]) -> list[dict]:
        """Score JSON-like records (partial payloads allowed)."""
        if not records:
            return []
        frame = pd.concat([build_inference_frame(record) for record in records], ignore_index=True)
        probabilities = self.predict_proba(frame)
        predictions = (probabilities >= self.threshold).astype(int)
        return [
            {
                "default_prediction": int(prediction),
                "default_probability": round(float(probability), 6),
                "risk_band": self.risk_band(float(probability)),
                "threshold": self.threshold,
            }
            for prediction, probability in zip(predictions, probabilities)
        ]

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a dataframe and append prediction columns."""
        probabilities = self.predict_proba(df)
        scored = df.copy()
        scored["default_probability"] = probabilities
        scored["default_prediction"] = (probabilities >= self.threshold).astype(int)
        scored["risk_band"] = [self.risk_band(value) for value in probabilities]
        return scored


def score_file(input_path: Path | str, output_path: Path | str, model_path: Path | str | None = None) -> int:
    """CLI helper: score a CSV file and write predictions next to the inputs."""
    scorer = CreditDefaultScorer.load(model_path)
    df = pd.read_csv(input_path)
    scored = scorer.score_frame(df)

    keep = [column for column in (TARGET_COLUMN, "LIMIT_BAL", "AGE") if column in scored.columns]
    keep += ["default_probability", "default_prediction", "risk_band"]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scored[keep].to_csv(destination, index=False)
    LOGGER.info("scored %s rows -> %s", len(scored), destination)
    return len(scored)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a CSV file with the trained PD model")
    parser.add_argument("--input", required=True, help="CSV with raw client features")
    parser.add_argument("--output", required=True, help="where to write the scored CSV")
    parser.add_argument("--model", default=str(MODEL_FILE), help="joblib bundle path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args(argv)
    score_file(args.input, args.output, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
