"""Data-drift monitoring for the PD model.

The script mimics production traffic: it takes a slice of the hold-out test
sample ("new clients"), sends it to the running FastAPI service and compares the
resulting distribution with the training reference using the Population
Stability Index (PSI) – the standard credit-risk drift metric – plus a
Kolmogorov-Smirnov test.

Typical demo::

    uvicorn src.api.app:app --port 8000          # terminal 1
    python -m src.monitoring.drift --api-url http://127.0.0.1:8000 --sample-size 1000

Exit codes: ``0`` – no significant drift, ``2`` – PSI alert triggered,
``3`` – the API could not be reached (unless ``--offline``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    MODEL_FILE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
    write_json,
)
from src.models.predict import CreditDefaultScorer

LOGGER = logging.getLogger(__name__)

#: PSI interpretation used in retail credit risk.
PSI_THRESHOLDS = {"stable": 0.10, "moderate": 0.25}

#: Features monitored in addition to the score distribution.
MONITORED_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "BILL_AMT1",
    "PAY_AMT1",
    "CREDIT_UTILIZATION",
    "DELINQ_COUNT",
]


# --------------------------------------------------------------------------- #
# PSI / KS
# --------------------------------------------------------------------------- #
def _proportions(expected: np.ndarray, actual: np.ndarray, bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    expected_share = expected_counts / max(1, expected_counts.sum())
    actual_share = actual_counts / max(1, actual_counts.sum())
    epsilon = 1e-6
    expected_share = np.clip(expected_share, epsilon, None)
    actual_share = np.clip(actual_share, epsilon, None)
    return expected_share, actual_share


def calculate_psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions.

    Bins are derived from the reference (expected) quantiles so that every
    reference bucket holds roughly the same number of observations.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        raise ValueError("PSI needs two non-empty samples")

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(expected, quantiles))
    if bin_edges.size < 3:  # constant / almost-constant reference distribution
        low = min(expected.min(), actual.min()) - 0.5
        high = max(expected.max(), actual.max()) + 0.5
        bin_edges = np.linspace(low, high, bins + 1)
    else:
        bin_edges[0] = min(bin_edges[0], actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

    expected_share, actual_share = _proportions(expected, actual, bin_edges)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def calculate_psi_categorical(expected, actual) -> float:
    """PSI for discrete/categorical columns based on category shares."""
    expected = pd.Series(expected).astype(str)
    actual = pd.Series(actual).astype(str)
    categories = sorted(set(expected.unique()) | set(actual.unique()))
    epsilon = 1e-6
    psi = 0.0
    for category in categories:
        expected_share = max(expected.eq(category).mean(), epsilon)
        actual_share = max(actual.eq(category).mean(), epsilon)
        psi += (actual_share - expected_share) * np.log(actual_share / expected_share)
    return float(psi)


def ks_test(expected, actual) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test (statistic, p-value)."""
    from scipy import stats

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    result = stats.ks_2samp(expected, actual)
    return float(result.statistic), float(result.pvalue)


def classify_psi(psi: float) -> str:
    """Turn a PSI value into the standard stability verdict."""
    if psi < PSI_THRESHOLDS["stable"]:
        return "stable"
    if psi < PSI_THRESHOLDS["moderate"]:
        return "moderate_shift"
    return "significant_shift"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class DriftReport:
    """Container for the drift analysis results."""

    reference_size: int
    current_size: int
    score_psi: float
    score_ks: float
    score_ks_pvalue: float
    feature_psi: dict = field(default_factory=dict)
    threshold: float = 0.5
    performance: dict = field(default_factory=dict)
    reference_mean_pd: float = 0.0
    current_mean_pd: float = 0.0
    current_default_rate: float | None = None
    alerts: list = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return classify_psi(self.score_psi)

    @property
    def has_alert(self) -> bool:
        return bool(self.alerts)

    def to_dict(self) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "score_psi": round(self.score_psi, 6),
            "score_verdict": self.verdict,
            "score_ks": round(self.score_ks, 6),
            "score_ks_pvalue": round(self.score_ks_pvalue, 6),
            "feature_psi": {name: round(value, 6) for name, value in self.feature_psi.items()},
            "threshold": self.threshold,
            "reference_mean_pd": round(self.reference_mean_pd, 6),
            "current_mean_pd": round(self.current_mean_pd, 6),
            "current_default_rate": self.current_default_rate,
            "performance": self.performance,
            "alerts": self.alerts,
        }


def _score_locally(scorer: CreditDefaultScorer, df: pd.DataFrame) -> np.ndarray:
    return scorer.predict_proba(df)


def _score_via_api(scorer: CreditDefaultScorer, df: pd.DataFrame, api_url: str, batch_size: int = 200) -> np.ndarray:
    """Send records to ``POST /predict/batch`` and collect the probabilities."""
    import requests

    url = api_url.rstrip("/") + "/predict/batch"
    raw_columns = [
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
    available = [column for column in raw_columns if column in df.columns]
    probabilities: list[float] = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size][available].to_dict(orient="records")
        response = requests.post(url, json={"records": chunk}, timeout=60)
        response.raise_for_status()
        probabilities.extend(item["default_probability"] for item in response.json()["predictions"])
    return np.asarray(probabilities, dtype=float)


def analyse_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    threshold: float = 0.5,
) -> DriftReport:
    """Compute PSI/KS statistics for the score and the monitored features."""
    feature_psi: dict[str, float] = {}
    for feature in MONITORED_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        reference_values = reference_df[feature].to_numpy(dtype=float)
        current_values = current_df[feature].to_numpy(dtype=float)
        feature_psi[feature] = calculate_psi(reference_values, current_values)

    score_psi = calculate_psi(reference_scores, current_scores)
    score_ks, score_ks_pvalue = ks_test(reference_scores, current_scores)

    performance: dict = {}
    if TARGET_COLUMN in current_df.columns:
        from sklearn.metrics import roc_auc_score

        y_true = current_df[TARGET_COLUMN].astype(int).to_numpy()
        predictions = (current_scores >= threshold).astype(int)
        performance = {
            "roc_auc": float(roc_auc_score(y_true, current_scores)),
            "default_rate": float(y_true.mean()),
            "flagged_rate": float(predictions.mean()),
        }

    alerts = []
    if score_psi >= PSI_THRESHOLDS["moderate"]:
        alerts.append(f"score PSI={score_psi:.4f} >= {PSI_THRESHOLDS['moderate']} (significant shift)")
    elif score_psi >= PSI_THRESHOLDS["stable"]:
        alerts.append(f"score PSI={score_psi:.4f} >= {PSI_THRESHOLDS['stable']} (moderate shift)")
    for feature, psi in feature_psi.items():
        if psi >= PSI_THRESHOLDS["moderate"]:
            alerts.append(f"{feature} PSI={psi:.4f} >= {PSI_THRESHOLDS['moderate']}")

    return DriftReport(
        reference_size=int(len(reference_df)),
        current_size=int(len(current_df)),
        score_psi=score_psi,
        score_ks=score_ks,
        score_ks_pvalue=score_ks_pvalue,
        feature_psi=feature_psi,
        threshold=threshold,
        performance=performance,
        reference_mean_pd=float(np.mean(reference_scores)),
        current_mean_pd=float(np.mean(current_scores)),
        current_default_rate=float(current_df[TARGET_COLUMN].mean()) if TARGET_COLUMN in current_df else None,
        alerts=alerts,
    )


def plot_drift(report: DriftReport, path: Path | str) -> Path:
    """Bar chart of the PSI values against the alert thresholds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = {"score": report.score_psi, **report.feature_psi}
    labels = list(values)
    scores = [values[label] for label in labels]
    colours = ["#c44e52" if value >= PSI_THRESHOLDS["moderate"] else "#4c72b0" for value in scores]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, scores, color=colours)
    ax.axhline(PSI_THRESHOLDS["stable"], color="orange", linestyle="--", lw=1, label="0.10 – moderate")
    ax.axhline(PSI_THRESHOLDS["moderate"], color="red", linestyle="--", lw=1, label="0.25 – significant")
    ax.set_ylabel("PSI")
    ax.set_title("Population Stability Index: reference vs. simulated new data")
    ax.tick_params(axis="x", rotation=25)
    for index, value in enumerate(scores):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _inject_drift(df: pd.DataFrame, factor: float = 1.6, share: float = 0.5, seed: int = 7) -> pd.DataFrame:
    """Artificially shift a share of the clients (used by the demo/alert test).

    The perturbation mimics a worsening cohort: larger balances, smaller
    payments and more months of delay. That moves both the feature
    distributions and the predicted PD, which is what the alert path must catch.

    pandas 3 no longer upcasts silently, so the perturbed amount columns are
    converted to float before the multiplication.
    """
    rng = np.random.default_rng(seed)
    drifted = df.copy()
    mask = rng.random(len(drifted)) < share

    for column in ("LIMIT_BAL", "BILL_AMT1"):
        if column in drifted.columns:
            drifted[column] = drifted[column].astype(float)
            drifted.loc[mask, column] = drifted.loc[mask, column] * factor

    if "PAY_AMT1" in drifted.columns:
        drifted["PAY_AMT1"] = drifted["PAY_AMT1"].astype(float)
        drifted.loc[mask, "PAY_AMT1"] = drifted.loc[mask, "PAY_AMT1"] / factor

    if "PAY_0" in drifted.columns:
        # +2 months of delay for the drifting cohort, clipped to the valid range.
        drifted["PAY_0"] = np.clip(drifted["PAY_0"].astype(int) + np.where(mask, 2, 0), -2, 8).astype(int)
    return drifted


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="base URL of the scoring API")
    parser.add_argument("--reference", default=str(TRAIN_FILE), help="processed training data (reference)")
    parser.add_argument("--current", default=str(TEST_FILE), help="processed test data (simulated traffic)")
    parser.add_argument("--model", default=str(MODEL_FILE), help="model bundle used to score the reference")
    parser.add_argument("--sample-size", type=int, default=1000, help="number of records to send to the API")
    parser.add_argument("--offline", action="store_true", help="score locally instead of calling the API")
    parser.add_argument("--inject-drift", action="store_true", help="perturb the sample to trigger an alert")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--report", default=str(REPORTS_DIR / "drift_report.json"), help="JSON report path")
    parser.add_argument("--figure", default=str(FIGURES_DIR / "drift_psi.png"), help="PNG report path")
    parser.add_argument("--fail-on-alert", action="store_true", help="exit with code 2 when PSI is critical")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args(argv)

    scorer = CreditDefaultScorer.load(args.model)
    reference_df = pd.read_csv(args.reference)
    current_df = pd.read_csv(args.current)

    if args.inject_drift:
        current_df = _inject_drift(current_df, seed=args.seed)
        LOGGER.warning("synthetic drift injected into the simulated traffic")

    reference_sample = reference_df.sample(
        n=min(len(reference_df), max(args.sample_size, 1000)), random_state=args.seed
    ).reset_index(drop=True)
    current_sample = current_df.sample(
        n=min(len(current_df), args.sample_size), random_state=args.seed + 1
    ).reset_index(drop=True)

    reference_scores = _score_locally(scorer, reference_sample)

    if args.offline:
        LOGGER.info("offline mode: scoring the simulated traffic locally")
        current_scores = _score_locally(scorer, current_sample)
    else:
        try:
            LOGGER.info("sending %s records to %s", len(current_sample), args.api_url)
            current_scores = _score_via_api(scorer, current_sample, args.api_url)
        except Exception as error:  # requests errors, timeouts, 4xx/5xx
            LOGGER.error("API request failed: %s", error)
            return 3

    report = analyse_drift(
        reference_df=reference_sample,
        current_df=current_sample,
        reference_scores=reference_scores,
        current_scores=current_scores,
        threshold=scorer.threshold,
    )

    write_json(report.to_dict(), args.report)
    plot_drift(report, args.figure)

    rows = [("score", report.score_psi, report.verdict)]
    rows += [(feature, psi, classify_psi(psi)) for feature, psi in report.feature_psi.items()]
    width = max(len(name) for name, _, _ in rows)

    print(f"\n{'metric':<{width}}  {'PSI':>8}  verdict")
    print("-" * (width + 22))
    for name, psi, verdict in rows:
        print(f"{name:<{width}}  {psi:>8.4f}  {verdict}")
    print(f"\nKS statistic = {report.score_ks:.4f} (p = {report.score_ks_pvalue:.4g})")
    print(f"mean PD: reference {report.reference_mean_pd:.4f} -> current {report.current_mean_pd:.4f}")
    if report.performance:
        print(
            f"live ROC-AUC = {report.performance['roc_auc']:.4f} | default rate = {report.performance['default_rate']:.4f}"
        )
    if report.alerts:
        print("\nALERTS:")
        for alert in report.alerts:
            print(f"  - {alert}")

    if args.fail_on_alert and report.has_alert:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
