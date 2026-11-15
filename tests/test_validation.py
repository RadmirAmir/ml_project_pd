"""Tests for the Great Expectations data contract.

These are the tests the assignment asks for: they fail when a new data
delivery contains anomalies (nulls, out-of-range values, wrong dtypes or an
unexpected schema), and they pass on the clean dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    COLUMN_SPECS,
    DataValidationError,
    build_expectation_suite,
    export_suite,
    validate_dataframe,
)
from tests.conftest import make_synthetic_raw

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_suite_contains_the_expected_check_families() -> None:
    """The suite covers existence, nullability, dtypes, ranges and sets."""
    _, suite = build_expectation_suite()
    expectation_types = {item["type"] for item in suite.to_json_dict()["expectations"]}

    assert "expect_table_row_count_to_be_between" in expectation_types
    assert "expect_column_to_exist" in expectation_types
    assert "expect_column_values_to_not_be_null" in expectation_types
    assert "expect_column_values_to_be_of_type" in expectation_types
    assert "expect_column_values_to_be_between" in expectation_types
    assert "expect_column_values_to_be_in_set" in expectation_types
    assert "expect_column_values_to_be_unique" in expectation_types


def test_clean_data_passes_validation() -> None:
    """A clean 1 200-row sample satisfies every expectation."""
    summary = validate_dataframe(make_synthetic_raw(rows=1200))

    assert summary["success"] is True
    assert summary["failed_expectations"] == []
    assert summary["evaluated_expectations"] >= 90


def test_nulls_fail_validation() -> None:
    """Injected nulls break the not-null contract."""
    frame = make_synthetic_raw(rows=1200)
    frame.loc[frame.index[:5], "LIMIT_BAL"] = np.nan

    summary = validate_dataframe(frame)

    assert summary["success"] is False
    failed = {(item["expectation"], item["column"]) for item in summary["failed_expectations"]}
    assert ("expect_column_values_to_not_be_null", "LIMIT_BAL") in failed


def test_out_of_range_age_fails_validation() -> None:
    """An impossible age (150) is rejected."""
    frame = make_synthetic_raw(rows=1200)
    frame.loc[frame.index[0], "AGE"] = 150

    summary = validate_dataframe(frame)

    assert summary["success"] is False
    assert any(
        item["expectation"] == "expect_column_values_to_be_between" and item["column"] == "AGE"
        for item in summary["failed_expectations"]
    )


def test_unexpected_sex_code_fails_validation() -> None:
    """A category outside the documented set (SEX = 5) is rejected."""
    frame = make_synthetic_raw(rows=1200)
    frame.loc[frame.index[0], "SEX"] = 5

    summary = validate_dataframe(frame)

    assert summary["success"] is False
    assert any(
        item["expectation"] == "expect_column_values_to_be_in_set" and item["column"] == "SEX"
        for item in summary["failed_expectations"]
    )


def test_invalid_target_fails_validation() -> None:
    """The target must stay binary."""
    frame = make_synthetic_raw(rows=1200)
    frame.loc[frame.index[0], "default.payment.next.month"] = 2

    summary = validate_dataframe(frame)

    assert summary["success"] is False


def test_wrong_dtype_fails_validation() -> None:
    """A float where an int is expected is a schema break."""
    frame = make_synthetic_raw(rows=1200).astype({"AGE": "float64"})
    frame["AGE"] = frame["AGE"] + 0.5

    summary = validate_dataframe(frame)

    assert summary["success"] is False
    assert any(
        item["expectation"] == "expect_column_values_to_be_of_type" and item["column"] == "AGE"
        for item in summary["failed_expectations"]
    )


def test_duplicate_ids_fail_validation() -> None:
    """Client ids must be unique (otherwise the split leaks)."""
    frame = make_synthetic_raw(rows=1200)
    frame.loc[frame.index[1], "ID"] = frame.loc[frame.index[0], "ID"]

    summary = validate_dataframe(frame)

    assert summary["success"] is False
    assert any(item["expectation"] == "expect_column_values_to_be_unique" for item in summary["failed_expectations"])


def test_too_few_rows_fail_validation() -> None:
    """A 10-row delivery cannot be a valid monthly batch."""
    frame = make_synthetic_raw(rows=10)

    summary = validate_dataframe(frame)

    assert summary["success"] is False
    assert any(item["expectation"] == "expect_table_row_count_to_be_between" for item in summary["failed_expectations"])


def test_missing_column_fails_validation() -> None:
    """Dropping a mandatory column is caught before feature engineering."""
    frame = make_synthetic_raw(rows=1200).drop(columns=["PAY_AMT6"])

    summary = validate_dataframe(frame)

    assert summary["success"] is False
    assert any(item["expectation"] == "expect_column_to_exist" for item in summary["failed_expectations"])


def test_strict_mode_raises() -> None:
    """``raise_on_error`` turns a failed suite into an exception for CI."""
    frame = make_synthetic_raw(rows=1200)
    frame.loc[frame.index[0], "AGE"] = 150

    with pytest.raises(DataValidationError):
        validate_dataframe(frame, raise_on_error=True)


def test_suite_export_creates_json(tmp_path) -> None:
    """The suite definition is persisted as an auditable JSON artifact."""
    path = export_suite(tmp_path / "suite.json")

    assert path.exists()
    assert path.stat().st_size > 1000


def test_column_specs_match_the_dataset_schema() -> None:
    """Every raw column is covered by the contract (no silent gaps)."""
    expected = {spec["column"] for spec in COLUMN_SPECS}
    assert len(expected) == 25
    assert "default.payment.next.month" in expected
    assert pd.api.types.is_integer_dtype(make_synthetic_raw(rows=5)["ID"])
