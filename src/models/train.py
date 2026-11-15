"""Train the credit-default (PD) model and log the experiment to MLflow.

Python entry point used by the DVC ``train`` stage::

    python -m src.models.train --model gradient_boosting --search randomized

What the stage produces:

* ``models/credit_default_model.joblib`` – serving bundle (pipeline + metadata);
* ``models/model_card.json``             – human readable model card;
* ``metrics/metrics.json``               – DVC metric of the test split;
* ``reports/figures/*.png``              – ROC / PR / confusion matrix / importances;
* one MLflow run with parameters, metrics, artifacts and a registered model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    METRICS_FILE,
    MODEL_CARD_FILE,
    MODEL_FILE,
    PARAMS_FILE,
    TEST_FILE,
    TRAIN_FILE,
    get_experiment_name,
    get_tracking_uri,
    load_params,
    write_json,
)
from src.data.make_dataset import load_processed_data, split_features_target
from src.features.build_features import get_feature_lists
from src.models.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_score_distribution,
    plot_threshold_sweep,
    top_k_metrics,
)
from src.models.pipeline import create_pipeline, search_hyperparameters

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "credit_default_model"


def _file_digest(path: Path) -> str:
    """SHA-256 of a file (used for experiment reproducibility metadata)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    """Return the current git commit hash (``unknown`` outside a repository)."""
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def _log_sklearn_model(
    mlflow,
    pipeline,
    artifact_name: str = "model",
    registered_model_name: str | None = None,
    signature=None,
    input_example=None,
):
    """Call ``mlflow.sklearn.log_model`` across MLflow API versions.

    ``serialization_format="pickle"`` is used deliberately: the default skops
    format refuses to persist histogram-based gradient boosting models unless
    their internal predictors are declared trusted, and the pickle format is the
    one consumed by the FastAPI service (``joblib``) — keeping a single
    serialization story for the artifact and the registry copy.
    """
    import inspect

    import mlflow.sklearn

    parameters = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs = {"serialization_format": "pickle"}
    if "name" in parameters:
        kwargs["name"] = artifact_name
    else:  # pragma: no cover - older MLflow releases
        kwargs["artifact_path"] = artifact_name
    if registered_model_name and "registered_model_name" in parameters:
        kwargs["registered_model_name"] = registered_model_name
    if signature is not None:
        kwargs["signature"] = signature
    if input_example is not None:
        kwargs["input_example"] = input_example
    return mlflow.sklearn.log_model(pipeline, **kwargs)


def fit_and_evaluate(
    model_name: str = "gradient_boosting",
    search_method: str = "randomized",
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    random_state: int = 42,
    model_params: dict | None = None,
    log_to_mlflow: bool = True,
    run_name: str | None = None,
    register_model: bool = True,
    save_artifacts: bool = True,
    output_model_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
    train_path: Path | str | None = None,
    test_path: Path | str | None = None,
    tags: dict | None = None,
) -> dict:
    """Train one configuration and return a result dictionary.

    The function is deliberately side-effect free when ``log_to_mlflow`` and
    ``save_artifacts`` are disabled, which is what the unit tests rely on.
    """
    train_df = load_processed_data(train_path or TRAIN_FILE)
    test_df = load_processed_data(test_path or TEST_FILE)
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    numeric_features, categorical_features = get_feature_lists()
    pipeline = create_pipeline(
        model_name=model_name,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        **(model_params or {}),
    )

    search_info: dict = {}
    if search_method not in {"none", "manual"} and not model_params:
        LOGGER.info("running %s hyper-parameter search for %s", search_method, model_name)
        pipeline, search_info = search_hyperparameters(
            pipeline,
            x_train,
            y_train,
            model_name=model_name,
            method=search_method,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        LOGGER.info("fitting %s without hyper-parameter search", model_name)
        pipeline.fit(x_train, y_train)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    # The decision threshold is tuned on the *training* predictions so that
    # the test split stays untouched by any model-selection decision.
    train_proba = pipeline.predict_proba(x_train)[:, 1]
    train_metrics = compute_metrics(y_train, train_proba, threshold=0.5)
    threshold = float(train_metrics["best_f1_threshold"])

    test_proba = pipeline.predict_proba(x_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_proba, threshold=threshold)
    test_metrics_at_05 = compute_metrics(y_test, test_proba, threshold=0.5, include_threshold_search=False)
    business_metrics = top_k_metrics(y_test, test_proba)

    figures: dict[str, Path] = {}
    if save_artifacts:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        figures["roc_curve"] = plot_roc_curve(y_test, test_proba, FIGURES_DIR / "roc_curve.png")
        figures["pr_curve"] = plot_precision_recall_curve(y_test, test_proba, FIGURES_DIR / "pr_curve.png")
        figures["confusion_matrix"] = plot_confusion_matrix(
            y_test, test_proba, FIGURES_DIR / "confusion_matrix.png", threshold=threshold
        )
        figures["threshold_sweep"] = plot_threshold_sweep(y_test, test_proba, FIGURES_DIR / "threshold_sweep.png")
        figures["score_distribution"] = plot_score_distribution(
            y_test, test_proba, FIGURES_DIR / "score_distribution.png"
        )
        importance = plot_feature_importance(pipeline, FIGURES_DIR / "feature_importance.png")
        if importance is not None:
            figures["feature_importance"] = importance

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dataset_hash = _file_digest(Path(train_path or TRAIN_FILE))

    result = {
        "model_name": model_name,
        "search_method": search_method,
        "search": search_info,
        "threshold": threshold,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_metrics_at_0.5": test_metrics_at_05,
        "business_metrics": business_metrics,
        "figures": {name: str(path) for name, path in figures.items()},
        "trained_at": trained_at,
        "dataset_hash": dataset_hash,
        "n_features": len(numeric_features) + len(categorical_features),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    if save_artifacts:
        bundle = {
            "model": pipeline,
            "model_name": model_name,
            "threshold": threshold,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": numeric_features + categorical_features,
            "trained_at": trained_at,
            "dataset_hash": dataset_hash,
            "sklearn_version": _sklearn_version(),
            "test_metrics": test_metrics,
        }
        model_path = Path(output_model_path or MODEL_FILE)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)

        metrics_target = Path(metrics_path or METRICS_FILE)
        metrics_payload = {
            "model_name": model_name,
            "trained_at": trained_at,
            "threshold": threshold,
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
            "ks": test_metrics["ks"],
            "f1": test_metrics["f1"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "accuracy": test_metrics["accuracy"],
            "brier_score": test_metrics["brier_score"],
            "best_f1_threshold": test_metrics["best_f1_threshold"],
            "roc_auc_at_0.5": test_metrics_at_05["roc_auc"],
            "f1_at_0.5": test_metrics_at_05["f1"],
            **business_metrics,
        }
        write_json(metrics_payload, metrics_target)

        model_card = {
            "name": MODEL_NAME,
            "version": 1,
            "model_name": model_name,
            "description": "Probability of default (PD) model for credit card clients.",
            "trained_at": trained_at,
            "dataset_hash_sha256": dataset_hash,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "features": {"numeric": numeric_features, "categorical": categorical_features},
            "hyperparameters": pipeline.named_steps["classifier"].get_params(),
            "decision_threshold": threshold,
            "metrics": test_metrics,
            "business_metrics": business_metrics,
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": _sklearn_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "git_commit": _git_commit(),
        }
        write_json(model_card, Path(MODEL_CARD_FILE))
        result["model_path"] = str(model_path)

    # ------------------------------------------------------------------ #
    # MLflow
    # ------------------------------------------------------------------ #
    if log_to_mlflow:
        import mlflow

        mlflow.set_tracking_uri(get_tracking_uri())
        mlflow.set_experiment(get_experiment_name())
        run_label = run_name or f"{model_name}-{search_method}"
        with mlflow.start_run(run_name=run_label) as run:
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "search_method": search_method,
                    "n_iter": n_iter,
                    "cv_folds": cv,
                    "scoring": scoring,
                    "random_state": random_state,
                    "decision_threshold": threshold,
                    "n_features": result["n_features"],
                    "train_rows": result["train_rows"],
                }
            )
            mlflow.log_params(search_info.get("best_params", {}))
            if search_info.get("best_cv_score") is not None:
                mlflow.log_metric("cv_best_score", search_info["best_cv_score"])

            flat_metrics = {
                f"test_{key}": value
                for key, value in test_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            flat_metrics.update(
                {
                    f"train_{key}": value
                    for key, value in train_metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
            flat_metrics.update(business_metrics)
            flat_metrics["test_roc_auc_at_0.5"] = test_metrics_at_05["roc_auc"]
            flat_metrics["test_f1_at_0.5"] = test_metrics_at_05["f1"]
            mlflow.log_metrics(flat_metrics)

            mlflow.set_tags(
                {
                    "model_family": model_name,
                    "dataset_hash": dataset_hash,
                    "git_commit": _git_commit(),
                    "host": socket.gethostname(),
                    "python_version": platform.python_version(),
                    "sklearn_version": _sklearn_version(),
                    **(tags or {}),
                }
            )

            for figure in figures.values():
                mlflow.log_artifact(str(figure), artifact_path="figures")
            if save_artifacts:
                mlflow.log_dict(test_metrics, "metrics/test_metrics.json")
                mlflow.log_dict(model_card, "model_card.json")
                try:
                    from mlflow.models import infer_signature

                    sample = x_train.head(5)
                    signature = infer_signature(sample, pipeline.predict_proba(sample)[:, 1])
                except Exception:  # pragma: no cover - signature is a nice-to-have
                    signature, sample = None, None
                _log_sklearn_model(
                    mlflow,
                    pipeline,
                    artifact_name="model",
                    registered_model_name="CreditDefaultModel" if register_model else None,
                    signature=signature,
                    input_example=sample,
                )

            result["mlflow_run_id"] = run.info.run_id
            LOGGER.info("logged MLflow run %s", run.info.run_id)

    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="classifier key (default: params.yaml)")
    parser.add_argument("--params", default=str(PARAMS_FILE), help="params.yaml location")
    parser.add_argument("--search", default=None, choices=["randomized", "grid", "none"], help="search method")
    parser.add_argument("--n-iter", type=int, default=None, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv", type=int, default=None, help="number of CV folds")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    parser.add_argument("--no-register", action="store_true", help="do not register the model")
    parser.add_argument("--output-model", default=str(MODEL_FILE), help="path of the joblib bundle")
    parser.add_argument("--metrics", default=str(METRICS_FILE), help="path of the metrics JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)
    params = load_params(args.params) if Path(args.params).exists() else {}
    train_params = params.get("train", {})
    data_params = params.get("data", {})

    result = fit_and_evaluate(
        model_name=args.model or train_params.get("model", "gradient_boosting"),
        search_method=args.search or train_params.get("search", "randomized"),
        n_iter=args.n_iter if args.n_iter is not None else int(train_params.get("n_iter", 20)),
        cv=args.cv if args.cv is not None else int(train_params.get("cv", 5)),
        scoring=train_params.get("scoring", "roc_auc"),
        n_jobs=int(train_params.get("n_jobs", -1)),
        random_state=int(data_params.get("random_state", 42)),
        log_to_mlflow=not args.no_mlflow,
        run_name=args.run_name,
        register_model=not args.no_register,
        output_model_path=args.output_model,
        metrics_path=args.metrics,
    )

    metrics = result["test_metrics"]
    LOGGER.info(
        "test metrics: ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Precision=%.4f | Recall=%.4f | KS=%.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        metrics["ks"],
    )
    print(
        json.dumps(
            {key: result[key] for key in ("model_name", "threshold", "test_metrics", "mlflow_run_id") if key in result},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
