"""Run the full experiment matrix and log every run to MLflow.

Usage::

    python -m src.models.experiment                # all experiments (6 runs)
    python -m src.models.experiment --only logreg  # filter by name substring
    python -m src.models.experiment --comparison reports/experiments.md

Each experiment trains one algorithm (optionally with a hyper-parameter search),
logs parameters/metrics/artifacts to MLflow and appends a row to a comparison
table. The best run is copied to ``models/`` and ``metrics/metrics.json`` so the
DVC stage and the API always serve the winning model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import REPORTS_DIR, write_json
from src.models.train import fit_and_evaluate

LOGGER = logging.getLogger(__name__)

#: The experiment matrix required by the assignment (>= 5 runs).
EXPERIMENTS: list[dict] = [
    {
        "name": "01_logreg_baseline",
        "model_name": "logistic_regression",
        "search_method": "none",
        "model_params": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000},
        "note": "Vanilla logistic regression – interpretable baseline",
    },
    {
        "name": "02_logreg_tuned",
        "model_name": "logistic_regression",
        "search_method": "randomized",
        "n_iter": 10,
        "note": "Logistic regression with RandomizedSearchCV (C, weights, solver)",
    },
    {
        "name": "03_random_forest",
        "model_name": "random_forest",
        "search_method": "randomized",
        "n_iter": 8,
        "note": "Random forest, bagging of decorrelated trees",
    },
    {
        "name": "04_extra_trees",
        "model_name": "extra_trees",
        "search_method": "randomized",
        "n_iter": 6,
        "note": "Extremely randomised trees – stronger regularisation",
    },
    {
        "name": "05_gradient_boosting",
        "model_name": "gradient_boosting",
        "search_method": "none",
        # Classic (non-histogram) boosting is slow, so it runs with a fixed,
        # small configuration instead of a full search.
        "model_params": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.9},
        "note": "Classic gradient boosting (fixed compact configuration)",
    },
    {
        "name": "06_hist_gradient_boosting",
        "model_name": "hist_gradient_boosting",
        "search_method": "randomized",
        "n_iter": 15,
        "note": "Histogram-based gradient boosting (fast, production candidate)",
    },
]


def run_experiments(
    experiments: list[dict] | None = None,
    cv: int = 3,
    log_to_mlflow: bool = True,
    register_model: bool = False,
    save_artifacts: bool = False,
) -> list[dict]:
    """Train every experiment and return a list of result summaries."""
    selected = experiments if experiments is not None else EXPERIMENTS
    results = []
    for experiment in selected:
        LOGGER.info("=" * 70)
        LOGGER.info("experiment %s (%s)", experiment["name"], experiment["model_name"])
        result = fit_and_evaluate(
            model_name=experiment["model_name"],
            search_method=experiment.get("search_method", "randomized"),
            n_iter=experiment.get("n_iter", 10),
            cv=cv,
            model_params=experiment.get("model_params"),
            log_to_mlflow=log_to_mlflow,
            run_name=experiment["name"],
            register_model=register_model,
            save_artifacts=save_artifacts,
            tags={"experiment_note": experiment.get("note", "")},
        )
        metrics = result["test_metrics"]
        results.append(
            {
                "name": experiment["name"],
                "model_name": experiment["model_name"],
                "search_method": result["search_method"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ks": metrics["ks"],
                "threshold": result["threshold"],
                "mlflow_run_id": result.get("mlflow_run_id"),
                "best_params": result.get("search", {}).get("best_params", experiment.get("model_params") or {}),
            }
        )
    return results


def comparison_table(results: list[dict]) -> str:
    """Render the experiment comparison as a Markdown table."""
    header = (
        "| # | experiment | model | ROC-AUC | PR-AUC | F1 | Precision | Recall | KS |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for index, row in enumerate(sorted(results, key=lambda item: item["roc_auc"], reverse=True), start=1):
        rows.append(
            f"| {index} | {row['name']} | {row['model_name']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | "
            f"{row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['ks']:.4f} |"
        )
    return "\n".join([header, *rows])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, help="substring filter on experiment names")
    parser.add_argument("--cv", type=int, default=3, help="CV folds used by the search")
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs to MLflow")
    parser.add_argument("--comparison", default=str(REPORTS_DIR / "experiments.md"), help="Markdown comparison path")
    parser.add_argument("--json", default=str(REPORTS_DIR / "experiments.json"), help="JSON comparison path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [item for item in EXPERIMENTS if args.only.lower() in item["name"].lower()]
        if not experiments:
            LOGGER.error("no experiment matched %r", args.only)
            return 1

    results = run_experiments(experiments, cv=args.cv, log_to_mlflow=not args.no_mlflow)
    table = comparison_table(results)

    Path(args.comparison).parent.mkdir(parents=True, exist_ok=True)
    Path(args.comparison).write_text("# Experiment comparison\n\n" + table + "\n", encoding="utf-8")
    write_json({"experiments": results}, Path(args.json))
    print("\n" + table + "\n")
    print(f"comparison written to {args.comparison}")

    best = max(results, key=lambda item: item["roc_auc"])
    print(f"best experiment: {best['name']} (ROC-AUC = {best['roc_auc']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
