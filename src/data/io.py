"""Dataset ingestion helpers.

The Kaggle/UCI export stores large amounts in scientific notation
(``5e+05``), so ``pandas.read_csv`` infers ``float64`` for ``LIMIT_BAL``,
``BILL_AMT*`` and ``PAY_AMT*`` even though the values are integers. The
ingestion layer normalises those columns back to ``int64`` so that:

* the Great Expectations dtype contract is meaningful (a genuine float in an
  amount column is still detected because the cast only happens when every
  value is integral);
* downstream feature engineering and the API work with consistent dtypes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import INTEGER_RAW_COLUMNS

LOGGER = logging.getLogger(__name__)


def coerce_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast integral numeric columns to ``int64`` without hiding real problems.

    A column is cast only when it is numeric (or numeric-castable) *and* every
    present value is integral. Columns with nulls or fractional values keep
    their float dtype, which lets the validation suite raise a proper alert.
    """
    coerced = df.copy()
    for column in coerced.columns:
        if column not in INTEGER_RAW_COLUMNS:
            continue
        values = pd.to_numeric(coerced[column], errors="coerce")
        if values.isna().any():
            coerced[column] = values
            continue
        if np.allclose(np.mod(values.to_numpy(dtype=float), 1.0), 0.0):
            coerced[column] = values.astype("int64")
        else:
            coerced[column] = values
    return coerced


def load_raw_csv(path: Path | str, **read_csv_kwargs) -> pd.DataFrame:
    """Read the raw dataset and normalise its numeric dtypes."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"raw dataset not found: {source}")
    df = pd.read_csv(source, **read_csv_kwargs)
    normalised = coerce_numeric_dtypes(df)
    changed = [column for column in df.columns if str(df[column].dtype) != str(normalised[column].dtype)]
    if changed:
        LOGGER.info("normalised dtypes for %s column(s): %s", len(changed), ", ".join(changed))
    return normalised
