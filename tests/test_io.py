"""Tests for the ingestion helpers (scientific notation, dtype recovery)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.io import coerce_numeric_dtypes, load_raw_csv


def test_scientific_notation_amounts_become_integers(coerced_frame: pd.DataFrame) -> None:
    """``5e+05`` is parsed and cast back to int64."""
    assert str(coerced_frame["LIMIT_BAL"].dtype) == "int64"
    assert coerced_frame["LIMIT_BAL"].tolist() == [500_000, 100_000]
    assert str(coerced_frame["BILL_AMT1"].dtype) == "int64"


def test_non_integral_values_stay_float() -> None:
    """A genuine fractional amount keeps its float dtype (so GE can flag it)."""
    frame = pd.DataFrame({"LIMIT_BAL": [100_000.5, 200_000.0]})

    coerced = coerce_numeric_dtypes(frame)

    assert str(coerced["LIMIT_BAL"].dtype) == "float64"


def test_nulls_keep_the_column_nullable() -> None:
    """A column with nulls is not silently cast to int."""
    frame = pd.DataFrame({"AGE": [25.0, np.nan, 40.0]})

    coerced = coerce_numeric_dtypes(frame)

    assert coerced["AGE"].isna().sum() == 1
    assert str(coerced["AGE"].dtype) == "float64"


def test_unknown_columns_are_untouched() -> None:
    """Columns outside the schema are not modified by the ingestion layer."""
    frame = pd.DataFrame({"CUSTOM": [1.5, 2.5]})

    coerced = coerce_numeric_dtypes(frame)

    assert coerced["CUSTOM"].tolist() == [1.5, 2.5]


def test_load_raw_csv_missing_file(tmp_path) -> None:
    """A missing file raises FileNotFoundError with a clear message."""
    with pytest.raises(FileNotFoundError, match="raw dataset not found"):
        load_raw_csv(tmp_path / "absent.csv")


def test_load_raw_csv_roundtrip(tmp_path, synthetic_raw: pd.DataFrame) -> None:
    """Reading a CSV written from a frame restores the integer dtypes."""
    path = tmp_path / "sample.csv"
    synthetic_raw.to_csv(path, index=False)

    loaded = load_raw_csv(path)

    assert loaded.shape == synthetic_raw.shape
    assert str(loaded["LIMIT_BAL"].dtype) in {"int64", "Int64"}
