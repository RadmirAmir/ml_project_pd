"""Unit tests for the feature engineering layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    BILL_COLUMNS,
    PAY_STATUS_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.features.build_features import (
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    FeatureEngineeringError,
    build_features,
    build_inference_frame,
    clean_data,
    get_feature_lists,
    select_model_matrix,
)


def test_clean_data_fixes_undocumented_categories(synthetic_raw: pd.DataFrame) -> None:
    """EDUCATION 0/5/6 -> 4 and MARRIAGE 0 -> 3 (documented UCI quirk)."""
    frame = synthetic_raw.head(10).copy()
    frame.loc[frame.index[:3], "EDUCATION"] = [0, 5, 6]
    frame.loc[frame.index[3], "MARRIAGE"] = 0

    cleaned = clean_data(frame)

    assert cleaned.loc[cleaned.index[:3], "EDUCATION"].eq(4).all()
    assert cleaned.loc[cleaned.index[3], "MARRIAGE"] == 3
    # the original frame must not be mutated
    assert frame.loc[frame.index[0], "EDUCATION"] == 0


def test_clean_data_clips_pay_status(synthetic_raw: pd.DataFrame) -> None:
    """Out-of-range PAY_* codes are clipped to the documented [-2, 8] range."""
    frame = synthetic_raw.head(5).copy()
    frame.loc[frame.index[0], "PAY_0"] = 99
    frame.loc[frame.index[1], "PAY_0"] = -99

    cleaned = clean_data(frame)

    assert cleaned.loc[cleaned.index[0], "PAY_0"] == 8
    assert cleaned.loc[cleaned.index[1], "PAY_0"] == -2


def test_build_features_adds_all_declared_columns(synthetic_raw: pd.DataFrame) -> None:
    """Every engineered column exists and the raw columns survive."""
    features = build_features(synthetic_raw)

    for column in RAW_FEATURE_COLUMNS + ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES:
        assert column in features.columns, column
    assert TARGET_COLUMN in features.columns
    assert len(features) == len(synthetic_raw)


def test_build_features_are_finite(synthetic_raw: pd.DataFrame) -> None:
    """No infinities leak from the divisions in the engineered features."""
    features = build_features(synthetic_raw)
    numeric_block = features[ENGINEERED_NUMERIC_FEATURES].to_numpy(dtype=float)
    assert np.isfinite(numeric_block).all()


def test_zero_bills_do_not_produce_infinities() -> None:
    """A client with zero balance still gets a finite utilisation and ratio."""
    frame = pd.DataFrame(
        [
            {
                "ID": 1,
                "LIMIT_BAL": 50_000,
                "SEX": 1,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 30,
                **{column: 0 for column in BILL_COLUMNS},
                **{column: 0 for column in ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6")},
                **{column: 0 for column in PAY_STATUS_COLUMNS},
                TARGET_COLUMN: 0,
            }
        ]
    )

    features = build_features(frame)

    assert features.loc[0, "CREDIT_UTILIZATION"] == 0
    assert features.loc[0, "PAYMENT_RATIO"] == 0
    assert features.loc[0, "LIMIT_PER_AGE"] == pytest.approx(50_000 / 30)


def test_credit_utilisation_is_computed_correctly() -> None:
    """Utilisation = mean(bill) / LIMIT_BAL for a hand-checked row."""
    bills = {f"BILL_AMT{i}": value for i, value in enumerate([60_000, 30_000, 0, 0, 0, 0], start=1)}
    frame = pd.DataFrame(
        [
            {
                "ID": 1,
                "LIMIT_BAL": 100_000,
                "SEX": 1,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 40,
                **bills,
                **{f"PAY_AMT{i}": 0 for i in range(1, 7)},
                **{column: 0 for column in PAY_STATUS_COLUMNS},
                TARGET_COLUMN: 0,
            }
        ]
    )

    features = build_features(frame)

    assert features.loc[0, "CREDIT_UTILIZATION"] == pytest.approx(90_000 / 6 / 100_000)
    assert features.loc[0, "BILL_AMT_MAX"] == 60_000
    assert features.loc[0, "BILL_AMT_MIN"] == 0


def test_bill_trend_sign_follows_growth() -> None:
    """An increasing bill sequence yields a positive slope, a falling one negative."""
    base = {
        "ID": 1,
        "LIMIT_BAL": 100_000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 40,
        **{f"PAY_AMT{i}": 0 for i in range(1, 7)},
        **{column: 0 for column in PAY_STATUS_COLUMNS},
        TARGET_COLUMN: 0,
    }
    growing = build_features(pd.DataFrame([{**base, **{f"BILL_AMT{i}": i * 1000 for i in range(1, 7)}}]))
    falling = build_features(pd.DataFrame([{**base, **{f"BILL_AMT{i}": (7 - i) * 1000 for i in range(1, 7)}}]))

    assert growing.loc[0, "BILL_AMT_TREND"] > 0
    assert falling.loc[0, "BILL_AMT_TREND"] < 0


def test_delinquency_counters() -> None:
    """DELINQ_COUNT / SEVERE_DELINQ_COUNT count the months with a delay."""
    frame = pd.DataFrame(
        [
            {
                "ID": 1,
                "LIMIT_BAL": 100_000,
                "SEX": 1,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 40,
                **{f"BILL_AMT{i}": 1000 for i in range(1, 7)},
                **{f"PAY_AMT{i}": 100 for i in range(1, 7)},
                "PAY_0": 3,
                "PAY_2": 1,
                "PAY_3": 0,
                "PAY_4": -1,
                "PAY_5": 2,
                "PAY_6": -2,
                TARGET_COLUMN: 1,
            }
        ]
    )

    features = build_features(frame)

    assert features.loc[0, "DELINQ_COUNT"] == 3  # PAY_0, PAY_2, PAY_5
    assert features.loc[0, "SEVERE_DELINQ_COUNT"] == 2  # PAY_0, PAY_5
    assert features.loc[0, "MAX_DELINQ"] == 3


def test_age_binning_boundaries() -> None:
    """Ages land in the documented bins (24 -> '<25', 25 -> '25-29')."""
    rows = []
    for index, age in enumerate([24, 25, 30, 55], start=1):
        rows.append(
            {
                "ID": index,
                "LIMIT_BAL": 100_000,
                "SEX": 1,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": age,
                **{f"BILL_AMT{i}": 1000 for i in range(1, 7)},
                **{f"PAY_AMT{i}": 100 for i in range(1, 7)},
                **{column: 0 for column in PAY_STATUS_COLUMNS},
                TARGET_COLUMN: 0,
            }
        )

    features = build_features(pd.DataFrame(rows))

    assert features["AGE_BIN"].tolist() == ["<25", "25-29", "30-34", "55+"]


def test_missing_columns_raise(synthetic_raw: pd.DataFrame) -> None:
    """Feature engineering refuses incomplete input instead of guessing."""
    broken = synthetic_raw.drop(columns=["AGE"])
    with pytest.raises(FeatureEngineeringError, match="AGE"):
        build_features(broken)


def test_empty_frame_raises(synthetic_raw: pd.DataFrame) -> None:
    """An empty dataframe is a programming error, not a silent empty result."""
    with pytest.raises(FeatureEngineeringError):
        build_features(synthetic_raw.iloc[0:0])


def test_select_model_matrix_uses_declared_order(synthetic_raw: pd.DataFrame) -> None:
    """The model matrix contains exactly the declared features in order."""
    features = build_features(synthetic_raw)
    matrix = select_model_matrix(features)
    numeric_features, categorical_features = get_feature_lists()

    assert list(matrix.columns) == numeric_features + categorical_features
    assert TARGET_COLUMN not in matrix.columns


def test_build_inference_frame_accepts_partial_payload() -> None:
    """The API payload may omit the payment history (defaults are applied)."""
    frame = build_inference_frame(
        {
            "LIMIT_BAL": 20_000,
            "SEX": 2,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 24,
            "PAY_0": 2,
            "BILL_AMT1": 3913,
            "PAY_AMT1": 0,
        }
    )

    numeric_features, categorical_features = get_feature_lists()
    assert len(frame) == 1
    assert set(numeric_features + categorical_features).issubset(frame.columns)
    assert frame["AGE_BIN"].iloc[0] == "<25"
