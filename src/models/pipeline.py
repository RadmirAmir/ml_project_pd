"""Sklearn pipelines and hyper-parameter search spaces for the PD model.

The pipeline always has the classic three parts required by the assignment:

1. ``preprocessor`` – a ``ColumnTransformer`` with an ``Imputer`` + ``Scaler``
   for numeric columns and an ``Imputer`` + ``OneHotEncoder`` for categorical
   ones.
2. ``classifier``   – the estimator under test.

Keeping the preprocessing inside the pipeline means the exact same object is
fitted on train, scored on test and served by the API: there is no train/serve
skew and no leaking of the test distribution into the scaler statistics.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import loguniform

from src.features.build_features import get_feature_lists

RANDOM_STATE = 42


def _make_classifier(model_name: str, **model_params):
    """Instantiate a classifier from the registry by name."""
    registry = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "hist_gradient_boosting": HistGradientBoostingClassifier,
    }
    if model_name not in registry:
        raise ValueError(f"unknown model {model_name!r}; expected one of {sorted(registry)}")
    return registry[model_name](**model_params)


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build the numeric/categorical preprocessing block."""
    if numeric_features is None or categorical_features is None:
        default_numeric, default_categorical = get_feature_lists()
        numeric_features = numeric_features or default_numeric
        categorical_features = categorical_features or default_categorical

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=0.005,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_pipeline(
    model_name: str = "gradient_boosting",
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    random_state: int = RANDOM_STATE,
    **model_params,
) -> Pipeline:
    """Create the full ``preprocessor -> classifier`` pipeline.

    Parameters
    ----------
    model_name:
        Key of the classifier registry.
    numeric_features, categorical_features:
        Column groups; defaults to the schema produced by ``build_features``.
    random_state:
        Seed forwarded to the estimator (only when the estimator accepts it).
    **model_params:
        Extra keyword arguments for the classifier, e.g. ``n_estimators=300``.
    """
    if numeric_features is None or categorical_features is None:
        numeric_features, categorical_features = get_feature_lists()

    # Every estimator in the registry accepts ``random_state``. It matters even
    # for the histogram-based booster: its internal early-stopping split is
    # random, so without a seed the same params.yaml would produce a slightly
    # different model (and different metrics) on every run.
    model_params.setdefault("random_state", random_state)
    if model_name in {"random_forest", "extra_trees"}:
        model_params.setdefault("n_jobs", -1)
    if model_name == "logistic_regression":
        model_params.setdefault("max_iter", 2000)
        model_params.setdefault("solver", "lbfgs")

    classifier = _make_classifier(model_name, **model_params)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


#: Search spaces used by ``GridSearchCV`` / ``RandomizedSearchCV``.
SEARCH_SPACES: dict[str, dict] = {
    "logistic_regression": {
        "classifier__C": loguniform(1e-3, 1e2),
        "classifier__class_weight": [None, "balanced"],
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [4, 6, 8, None],
        "classifier__min_samples_leaf": [1, 5, 20, 50],
        "classifier__max_features": ["sqrt", "log2", 0.5],
        "classifier__class_weight": [None, "balanced_subsample"],
    },
    "extra_trees": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [6, 10, None],
        "classifier__min_samples_leaf": [1, 5, 20],
        "classifier__max_features": ["sqrt", 0.5],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.02, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4],
        "classifier__subsample": [0.8, 1.0],
        "classifier__min_samples_leaf": [1, 10, 30],
    },
    "hist_gradient_boosting": {
        "classifier__learning_rate": [0.02, 0.05, 0.1],
        "classifier__max_depth": [None, 3, 5],
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__min_samples_leaf": [20, 50, 100],
        "classifier__l2_regularization": [0.0, 1.0, 10.0],
    },
}


def get_search_space(model_name: str) -> dict:
    """Return the hyper-parameter space for ``model_name``."""
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"no search space for {model_name!r}")
    return dict(SEARCH_SPACES[model_name])


def search_hyperparameters(
    pipeline: Pipeline,
    x_train,
    y_train,
    model_name: str,
    method: str = "randomized",
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
):
    """Run ``GridSearchCV`` or ``RandomizedSearchCV`` over the model space."""
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

    param_space = get_search_space(model_name)
    cross_validation = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if method == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_space,
            scoring=scoring,
            cv=cross_validation,
            n_jobs=n_jobs,
            refit=True,
            verbose=0,
        )
    elif method == "randomized":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cross_validation,
            n_jobs=n_jobs,
            refit=True,
            random_state=random_state,
            verbose=0,
        )
    elif method in {"none", "manual"}:
        pipeline.fit(x_train, y_train)
        return pipeline, {}
    else:
        raise ValueError("method must be 'grid', 'randomized' or 'none'")

    search.fit(x_train, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "cv_results": _serialisable_cv_results(search.cv_results_),
    }


def _serialisable_cv_results(cv_results: dict) -> list[dict]:
    """Reduce ``GridSearchCV.cv_results_`` to a compact, JSON-friendly list."""
    rows = []
    for index, params in enumerate(cv_results["params"]):
        rows.append(
            {
                "params": {key: _to_python(value) for key, value in params.items()},
                "mean_test_score": float(cv_results["mean_test_score"][index]),
                "std_test_score": float(cv_results["std_test_score"][index]),
                "rank_test_score": int(cv_results["rank_test_score"][index]),
            }
        )
    return rows


def _to_python(value):
    """Convert numpy scalars to Python types (keeps JSON/MLflow logging happy)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the expanded feature names produced by the preprocessor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - defensive, depends on sklearn version
        return []


def predict_scores(pipeline: Pipeline, x) -> np.ndarray:
    """Return the probability of default for every row."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x)[:, 1]
    decision = pipeline.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))
