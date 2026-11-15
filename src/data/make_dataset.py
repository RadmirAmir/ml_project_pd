"""Build the processed train/test datasets used by the training stage.

Usage::

    python -m src.data.make_dataset \
        --input data/raw/UCI_Credit_Card.csv \
        --output-dir data/processed \
        --test-size 0.2 \
        --random-state 42

The stage is deterministic: the same input file, split ratio and seed always
produce byte-identical CSV files, which is what makes ``dvc repro`` cacheable.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_METADATA_FILE,
    ID_COLUMN,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
    read_json,
    write_json,
)
from src.data.io import load_raw_csv
from src.features.build_features import (
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    FeatureEngineeringError,
    build_features,
    get_feature_lists,
    validate_raw_schema,
)

LOGGER = logging.getLogger(__name__)


def load_raw_data(path: Path | str) -> pd.DataFrame:
    """Read the raw CSV, normalise dtypes and validate the expected columns."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"raw dataset not found: {source}. " "Run `dvc pull` or place UCI_Credit_Card.csv into data/raw/."
        )
    df = load_raw_csv(source)
    validate_raw_schema(df)
    LOGGER.info("loaded raw data: %s rows x %s columns", *df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a featurised dataframe into stratified train/test parts."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be inside the (0, 1) interval")
    if TARGET_COLUMN not in df.columns:
        raise FeatureEngineeringError(f"target column {TARGET_COLUMN!r} is missing")

    stratify_column = df[TARGET_COLUMN] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_column,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_dataset(
    input_path: Path | str = RAW_DATA_FILE,
    output_dir: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    metadata_path: Path | str | None = None,
    write_metadata: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full preparation step and persist ``train.csv``/``test.csv``.

    Returns the two dataframes so tests and notebooks can reuse them without
    touching the disk.
    """
    raw_df = load_raw_data(input_path)
    features_df = build_features(raw_df)

    train_df, test_df = split_data(features_df, test_size=test_size, random_state=random_state)

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        train_path = destination / TRAIN_FILE.name
        test_path = destination / TEST_FILE.name
        metadata_file = Path(metadata_path) if metadata_path else destination / FEATURE_METADATA_FILE.name
    else:
        train_path, test_path = TRAIN_FILE, TEST_FILE
        metadata_file = Path(metadata_path) if metadata_path else FEATURE_METADATA_FILE

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    LOGGER.info("saved %s (%s rows) and %s (%s rows)", train_path, len(train_df), test_path, len(test_df))

    if write_metadata:
        numeric_features, categorical_features = get_feature_lists()
        metadata = {
            "target": TARGET_COLUMN,
            "id_column": ID_COLUMN,
            "raw_features": [c for c in raw_df.columns if c not in (ID_COLUMN, TARGET_COLUMN)],
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
            "engineered_categorical_features": ENGINEERED_CATEGORICAL_FEATURES,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_size": test_size,
            "random_state": random_state,
            "default_rate_train": float(train_df[TARGET_COLUMN].mean()),
            "default_rate_test": float(test_df[TARGET_COLUMN].mean()),
            "raw_dataset": Path(input_path).name,
        }
        write_json(metadata, metadata_file)
        LOGGER.info("saved feature metadata to %s", metadata_file)

    return train_df, test_df


def load_processed_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load a processed split (defaults to the training split)."""
    target = Path(path) if path is not None else TRAIN_FILE
    if not target.exists():
        raise FileNotFoundError(f"processed dataset not found: {target}. Run the `prepare` stage first.")
    return pd.read_csv(target)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a processed dataframe into ``X`` (model features) and ``y``."""
    numeric_features, categorical_features = get_feature_lists()
    columns = numeric_features + categorical_features
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise FeatureEngineeringError("processed dataset is missing columns: " + ", ".join(missing))
    return df[columns].copy(), df[TARGET_COLUMN].astype(int).copy()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(RAW_DATA_FILE), help="path to the raw UCI CSV")
    parser.add_argument("--output-dir", default=str(TRAIN_FILE.parent), help="where to write train/test CSVs")
    parser.add_argument("--test-size", type=float, default=0.2, help="fraction of the data used for testing")
    parser.add_argument("--random-state", type=int, default=42, help="random seed")
    parser.add_argument("--params", default=None, help="optional params.yaml (overrides the flags)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)
    test_size = args.test_size
    random_state = args.random_state
    if args.params and Path(args.params).exists():
        import yaml

        with Path(args.params).open("r", encoding="utf-8") as handle:
            params = yaml.safe_load(handle) or {}
        data_params = params.get("data", {})
        test_size = data_params.get("test_size", test_size)
        random_state = data_params.get("random_state", random_state)

    prepare_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    main()
