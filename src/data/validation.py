"""Data validation with Great Expectations.

The suite below is the contract for the *raw* UCI credit card dataset: it
checks column presence, nullability, dtypes, value ranges/sets, uniqueness of
the client id and the overall row count. It is deliberately strict – its job is
to fail loudly when a new data delivery breaks the schema, because every
downstream stage (feature engineering, training, scoring) assumes it.

Usage::

    python -m src.data.validation                       # validate data/raw/*
    python -m src.data.validation --input new_delivery.csv --strict
    python -m src.data.validation --export-suite        # write the suite JSON

Exit code ``1`` means the suite failed, which is exactly what the CI job
expects on bad data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# tqdm (used by Great Expectations for metric progress) honours this flag.
os.environ.setdefault("TQDM_DISABLE", "1")

from src.config import EXPECTATIONS_DIR, RAW_DATA_FILE, TARGET_COLUMN, write_json
from src.data.io import load_raw_csv
from src.features.build_features import FeatureEngineeringError

LOGGER = logging.getLogger(__name__)

# Great Expectations is extremely chatty on import (one INFO line per registry
# hook). Keep the project logs readable by silencing its own loggers.
for _noisy_logger in ("great_expectations", "great_expectations._docs_decorators"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

SUITE_NAME = "credit_raw_data_suite"

#: Expected dtype, range and allowed values per raw column.
#: ``(column, dtype, min_value, max_value, allowed_values)``
COLUMN_SPECS: list[dict] = [
    {"column": "ID", "dtype": "int64", "unique": True},
    {"column": "LIMIT_BAL", "dtype": "int64", "min": 10_000, "max": 1_000_000},
    {"column": "SEX", "dtype": "int64", "values": [1, 2]},
    {"column": "EDUCATION", "dtype": "int64", "values": [0, 1, 2, 3, 4, 5, 6]},
    {"column": "MARRIAGE", "dtype": "int64", "values": [0, 1, 2, 3]},
    {"column": "AGE", "dtype": "int64", "min": 18, "max": 100},
    {"column": "PAY_0", "dtype": "int64", "min": -2, "max": 9},
    {"column": "PAY_2", "dtype": "int64", "min": -2, "max": 9},
    {"column": "PAY_3", "dtype": "int64", "min": -2, "max": 9},
    {"column": "PAY_4", "dtype": "int64", "min": -2, "max": 9},
    {"column": "PAY_5", "dtype": "int64", "min": -2, "max": 9},
    {"column": "PAY_6", "dtype": "int64", "min": -2, "max": 9},
    *[{"column": f"BILL_AMT{i}", "dtype": "int64", "min": -1_000_000, "max": 10_000_000} for i in range(1, 7)],
    *[{"column": f"PAY_AMT{i}", "dtype": "int64", "min": 0, "max": 10_000_000} for i in range(1, 7)],
    {"column": TARGET_COLUMN, "dtype": "int64", "values": [0, 1]},
]

#: Row-count contract of a full dataset delivery (a 1 000-row sample still passes).
MIN_ROWS = 1_000
MAX_ROWS = 1_000_000


def _import_ge():
    """Import Great Expectations lazily so the rest of the app stays light."""
    try:
        import great_expectations as gx
    except ImportError as error:  # pragma: no cover - dependency is pinned
        raise RuntimeError(
            "Great Expectations is required for data validation. "
            "Install it with `pip install -r requirements-train.txt`."
        ) from error
    return gx


def build_expectation_suite(gx=None):
    """Create the Great Expectations suite for the raw credit-card dataset."""
    if gx is None:
        gx = _import_ge()

    context = gx.get_context(mode="ephemeral")

    suite = context.suites.add(gx.ExpectationSuite(name=SUITE_NAME))
    expectations = gx.expectations

    suite.add_expectation(expectations.ExpectTableRowCountToBeBetween(min_value=MIN_ROWS, max_value=MAX_ROWS))
    suite.add_expectation(
        expectations.ExpectTableColumnsToMatchSet(
            column_set=[spec["column"] for spec in COLUMN_SPECS], exact_match=True
        )
    )

    for spec in COLUMN_SPECS:
        column = spec["column"]
        suite.add_expectation(expectations.ExpectColumnToExist(column=column))
        suite.add_expectation(expectations.ExpectColumnValuesToNotBeNull(column=column, mostly=1.0))
        suite.add_expectation(expectations.ExpectColumnValuesToBeOfType(column=column, type_=spec["dtype"]))
        if "min" in spec or "max" in spec:
            suite.add_expectation(
                expectations.ExpectColumnValuesToBeBetween(
                    column=column,
                    min_value=spec.get("min"),
                    max_value=spec.get("max"),
                )
            )
        if "values" in spec:
            suite.add_expectation(expectations.ExpectColumnValuesToBeInSet(column=column, value_set=spec["values"]))
        if spec.get("unique"):
            suite.add_expectation(expectations.ExpectColumnValuesToBeUnique(column=column))

    # Distribution sanity checks – catch silent shifts in a new data delivery.
    suite.add_expectation(expectations.ExpectColumnMeanToBeBetween(column="AGE", min_value=20, max_value=45))
    suite.add_expectation(
        expectations.ExpectColumnMeanToBeBetween(column=TARGET_COLUMN, min_value=0.10, max_value=0.35)
    )
    return context, suite


def validate_dataframe(df: pd.DataFrame, raise_on_error: bool = False) -> dict:
    """Run the suite against ``df`` and return a JSON-friendly result dict."""
    gx = _import_ge()
    context, suite = build_expectation_suite(gx)

    data_source = context.data_sources.add_pandas("credit_pandas")
    data_asset = data_source.add_dataframe_asset(name="raw_credit_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("whole_dataframe")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(suite)

    failed = []
    for expectation_result in result.results:
        if expectation_result.success:
            continue
        config = expectation_result.expectation_config
        failed.append(
            {
                "expectation": config.type,
                "column": (config.kwargs or {}).get("column"),
                "kwargs": {key: value for key, value in (config.kwargs or {}).items() if key != "column"},
                "unexpected_count": (expectation_result.result or {}).get("unexpected_count"),
                "unexpected_percent": (expectation_result.result or {}).get("unexpected_percent"),
            }
        )

    summary = {
        "success": bool(result.success),
        "suite": SUITE_NAME,
        "rows": int(len(df)),
        "evaluated_expectations": int(result.statistics.get("evaluated_expectations", 0)),
        "successful_expectations": int(result.statistics.get("successful_expectations", 0)),
        "unsuccessful_expectations": int(result.statistics.get("unsuccessful_expectations", 0)),
        "success_percent": float(result.statistics.get("success_percent", 0.0)),
        "failed_expectations": failed,
    }

    if raise_on_error and not summary["success"]:
        raise DataValidationError(summary)
    return summary


class DataValidationError(RuntimeError):
    """Raised when the Great Expectations suite fails in strict mode."""

    def __init__(self, summary: dict):
        self.summary = summary
        failures = ", ".join(
            sorted({f"{item['expectation']}({item['column']})" for item in summary["failed_expectations"]})
        )
        super().__init__(
            f"data validation failed: {summary['successful_expectations']}/"
            f"{summary['evaluated_expectations']} expectations passed. Failures: {failures}"
        )


def validate_file(path: Path | str = RAW_DATA_FILE, raise_on_error: bool = False) -> dict:
    """Validate a CSV file on disk (dtypes are normalised by the ingestion layer)."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"dataset not found: {source}")
    try:
        df = load_raw_csv(source)
    except Exception as error:  # pragma: no cover - unreadable file
        raise FeatureEngineeringError(f"cannot read {source}: {error}") from error
    LOGGER.info("validating %s (%s rows x %s columns)", source, *df.shape)
    return validate_dataframe(df, raise_on_error=raise_on_error)


def export_suite(path: Path | str | None = None) -> Path:
    """Persist the suite definition as JSON (audit artifact for reviewers)."""
    _, suite = build_expectation_suite()
    destination = Path(path) if path else EXPECTATIONS_DIR / f"{SUITE_NAME}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = suite.to_json_dict()
    payload.pop("id", None)
    for expectation in payload.get("expectations", []):
        expectation.pop("id", None)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    LOGGER.info("suite exported to %s", destination)
    return destination


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(RAW_DATA_FILE), help="CSV file to validate")
    parser.add_argument("--output", default=None, help="write the validation result as JSON")
    parser.add_argument("--export-suite", action="store_true", help="only export the suite JSON")
    parser.add_argument("--strict", action="store_true", help="raise/exit 1 on failure (default for the CLI)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)

    if args.export_suite:
        export_suite()
        return 0

    summary = validate_file(args.input)
    if args.output:
        write_json(summary, args.output)

    print(f"suite: {summary['suite']}")
    print(
        f"expectations: {summary['successful_expectations']}/{summary['evaluated_expectations']} passed "
        f"({summary['success_percent']:.1f}%) on {summary['rows']} rows"
    )
    if summary["failed_expectations"]:
        print("\nfailed expectations:")
        for item in summary["failed_expectations"]:
            column = item["column"] or "-"
            print(f"  - {item['expectation']} [{column}] unexpected={item['unexpected_count']}")
        return 1
    print("validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
