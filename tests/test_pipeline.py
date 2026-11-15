"""Unit tests for the sklearn pipelines and the hyper-parameter search."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.pipeline import (
    SEARCH_SPACES,
    build_preprocessor,
    create_pipeline,
    get_feature_names,
    get_search_space,
    predict_scores,
    search_hyperparameters,
)


def test_create_pipeline_has_required_steps() -> None:
    """The pipeline is preprocessor -> classifier, as required by the brief."""
    pipeline = create_pipeline(model_name="gradient_boosting", n_estimators=10)

    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformers = {name: (transformer, columns) for name, transformer, columns in preprocessor.transformers}
    numeric_pipeline, _ = transformers["num"]
    categorical_pipeline, _ = transformers["cat"]

    assert [name for name, _ in numeric_pipeline.steps] == ["imputer", "scaler"]
    assert [name for name, _ in categorical_pipeline.steps] == ["imputer", "onehot"]


@pytest.mark.parametrize(
    "model_name",
    ["logistic_regression", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting"],
)
def test_pipeline_fits_and_predicts_probabilities(model_name: str, train_test_frames) -> None:
    """Every registered estimator trains and returns calibrated-looking PDs."""
    from src.data.make_dataset import split_features_target

    train_df, test_df = train_test_frames
    x_train, y_train = split_features_target(train_df)
    x_test, _ = split_features_target(test_df)

    params = {"n_estimators": 10} if model_name in {"random_forest", "extra_trees", "gradient_boosting"} else {}
    pipeline = create_pipeline(model_name=model_name, **params)
    pipeline.fit(x_train, y_train)
    probabilities = predict_scores(pipeline, x_test)

    assert probabilities.shape == (len(x_test),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert len(np.unique(probabilities)) > 1


def test_unknown_categories_are_ignored(train_test_frames) -> None:
    """``handle_unknown='ignore'`` lets a new category reach the model safely."""
    from src.data.make_dataset import split_features_target

    train_df, _ = train_test_frames
    x_train, y_train = split_features_target(train_df)
    pipeline = create_pipeline(model_name="logistic_regression")
    pipeline.fit(x_train, y_train)

    unseen = x_train.head(3).copy()
    unseen["EDUCATION"] = 4
    unseen["AGE_BIN"] = "99+"
    probabilities = pipeline.predict_proba(unseen)[:, 1]

    assert probabilities.shape == (3,)


def test_missing_numeric_values_are_imputed(train_test_frames) -> None:
    """NaNs in numeric columns are imputed instead of crashing the scorer."""
    from src.data.make_dataset import split_features_target

    train_df, _ = train_test_frames
    x_train, y_train = split_features_target(train_df)
    pipeline = create_pipeline(model_name="logistic_regression")
    pipeline.fit(x_train, y_train)

    sample = x_train.head(2).copy()
    sample.loc[sample.index[0], "LIMIT_BAL"] = np.nan
    probabilities = pipeline.predict_proba(sample)[:, 1]

    assert np.isfinite(probabilities).all()


def test_search_spaces_cover_every_model() -> None:
    """Each registered estimator has a non-empty hyper-parameter space."""
    assert set(SEARCH_SPACES) == {
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
    }
    for model_name in SEARCH_SPACES:
        assert get_search_space(model_name)


def test_unknown_model_raises() -> None:
    """An unknown estimator name fails fast with a helpful message."""
    with pytest.raises(ValueError, match="unknown model"):
        create_pipeline(model_name="magic_boost")
    with pytest.raises(ValueError, match="no search space"):
        get_search_space("magic_boost")


def test_randomized_search_returns_best_estimator(train_test_frames) -> None:
    """RandomizedSearchCV returns a refitted pipeline plus search metadata."""
    from src.data.make_dataset import split_features_target

    train_df, _ = train_test_frames
    x_train, y_train = split_features_target(train_df)
    pipeline = create_pipeline(model_name="logistic_regression")

    best, info = search_hyperparameters(
        pipeline,
        x_train,
        y_train,
        model_name="logistic_regression",
        method="randomized",
        n_iter=3,
        cv=2,
        n_jobs=1,
    )

    assert hasattr(best, "predict_proba")
    assert info["best_cv_score"] > 0.5
    assert "classifier__C" in info["best_params"]
    assert len(info["cv_results"]) == 3


def test_preprocessor_expands_categorical_features(train_test_frames) -> None:
    """One-hot encoding expands the categorical block into named columns."""
    from src.data.make_dataset import split_features_target

    train_df, _ = train_test_frames
    x_train, y_train = split_features_target(train_df)
    pipeline = create_pipeline(model_name="logistic_regression")
    pipeline.fit(x_train, y_train)

    names = get_feature_names(pipeline)

    assert len(names) > x_train.shape[1]
    assert any(name.startswith("SEX_") for name in names)


def test_build_preprocessor_with_custom_columns() -> None:
    """The preprocessor honours an explicit column selection."""
    preprocessor = build_preprocessor(numeric_features=["LIMIT_BAL"], categorical_features=["SEX"])
    transformers = {name: (transformer, columns) for name, transformer, columns in preprocessor.transformers}

    assert transformers["num"][1] == ["LIMIT_BAL"]
    assert transformers["cat"][1] == ["SEX"]
