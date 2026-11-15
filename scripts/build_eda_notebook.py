"""Build (and optionally execute) the exploratory data analysis notebook.

Notebooks are JSON, which is painful to maintain by hand, so the source of
truth for ``notebooks/01_eda.ipynb`` is this script. It keeps the EDA
reproducible::

    python scripts/build_eda_notebook.py --execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_eda.ipynb"


def build_notebook() -> nbf.NotebookNode:
    """Create the notebook object with the EDA narrative and code cells."""
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }

    cells = []

    def markdown(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(source: str) -> None:
        cells.append(nbf.v4.new_code_cell(source.strip()))

    markdown("""
# EDA: Default of Credit Card Clients (PD model)

**Goal.** Explore the UCI credit card dataset before building the probability-of-default
(PD) model, and document the decisions that feed `src/features/build_features.py`
and `src/data/validation.py`.

**Dataset.** 30 000 clients of a Taiwanese credit card issuer (April–September 2005):
demographics, six months of repayment status, bill statements and previous payments.
The target `default.payment.next.month` is `1` when the client defaults next month.

**Sections**

1. Data quality (dtypes, duplicates, missing values, undocumented category codes)
2. Target distribution (class imbalance)
3. Numeric and categorical drivers
4. Payment history — the strongest signal
5. Engineered features
6. Conclusions and modelling decisions
""")

    code("""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make the project importable when the notebook is executed from notebooks/
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_FILE, TARGET_COLUMN, BILL_COLUMNS, PAYMENT_COLUMNS, PAY_STATUS_COLUMNS
from src.data.io import load_raw_csv
from src.features.build_features import build_features, get_feature_lists

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

df = load_raw_csv(RAW_DATA_FILE)
print(f"rows: {len(df):,} | columns: {df.shape[1]}")
df.head()
""")

    markdown("""
## 1. Data quality

The dataset has no missing values, but it is not spotless: the UCI documentation
defines `EDUCATION` as 1–4 and `MARRIAGE` as 1–3, yet the file contains codes
`0`, `5` and `6` (undocumented) plus payment statuses outside the documented
`[-2, 9]` range. All of this is handled by `clean_data`.
""")

    code("""
quality = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "unique": df.nunique(),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
    }
)
display(quality)

print("duplicated client IDs:", int(df["ID"].duplicated().sum()))
print("EDUCATION value counts:")
display(df["EDUCATION"].value_counts().sort_index().to_frame("count"))
print("MARRIAGE value counts:")
display(df["MARRIAGE"].value_counts().sort_index().to_frame("count"))
print("PAY_0 range:", df["PAY_0"].min(), "-", df["PAY_0"].max())
""")

    markdown("""
**Finding.** `EDUCATION` codes `0/5/6` (345 clients) and `MARRIAGE` code `0`
(54 clients) are undocumented and unknown-bucket material; they are mapped to
the "others" category (`4` and `3`) in `clean_data`. Amounts in the CSV use
scientific notation (`5e+05`), so they are re-typed to `int64` on ingestion.
""")

    code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for axis, column in zip(axes, ["AGE", "LIMIT_BAL", "BILL_AMT1"]):
    sns.histplot(df[column], bins=50, ax=axis, color="#4c72b0")
    axis.set_title(f"{column} distribution")
    axis.set_xlabel(column)
fig.tight_layout()
plt.show()
""")

    markdown("""
## 2. Target distribution

The classes are imbalanced — roughly **22 % default rate**. That is why the
project reports ROC-AUC, PR-AUC, KS and Precision/Recall/F1 instead of accuracy,
tunes the decision threshold, and evaluates `class_weight="balanced"` variants.
""")

    code("""
target_counts = df[TARGET_COLUMN].value_counts().sort_index()
default_rate = df[TARGET_COLUMN].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(["No default (0)", "Default (1)"], target_counts.values, color=["#4c72b0", "#c44e52"])
axes[0].set_title(f"Class balance — default rate {default_rate:.2%}")
axes[0].set_ylabel("clients")
for index, value in enumerate(target_counts.values):
    axes[0].text(index, value, f"{value:,}", ha="center", va="bottom")

# Default rate across the age bins used by the feature builder.
age_bins = pd.cut(df["AGE"], bins=[0, 25, 30, 35, 40, 45, 50, 55, 200], right=False,
                  labels=["<25", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55+"])
by_age = df.groupby(age_bins, observed=True)[TARGET_COLUMN].mean()
by_age.plot(kind="bar", ax=axes[1], color="#55a868")
axes[1].axhline(default_rate, color="red", linestyle="--", label=f"overall {default_rate:.2%}")
axes[1].set_title("Default rate by age bin")
axes[1].set_ylabel("default rate")
axes[1].legend()
fig.tight_layout()
plt.show()

print(f"overall default rate: {default_rate:.2%}")
""")

    markdown("""
## 3. Categorical drivers

Sex, education and marital status show only weak differences, while the
repayment status carries most of the signal. Categorical features are therefore
one-hot encoded (low cardinality), and the numeric credit variables are
standardised inside the pipeline.
""")

    code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for axis, column in zip(axes, ["SEX", "EDUCATION", "MARRIAGE"]):
    rates = df.groupby(column)[TARGET_COLUMN].agg(["mean", "count"])
    axis.bar(rates.index.astype(str), rates["mean"], color="#4c72b0")
    axis.axhline(default_rate, color="red", linestyle="--", lw=1)
    axis.set_title(f"Default rate by {column}")
    axis.set_ylabel("default rate")
    for position, (rate, count) in enumerate(zip(rates["mean"], rates["count"])):
        axis.text(position, rate, f"{rate:.2%}\\n(n={count:,})", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
plt.show()
""")

    markdown("""
## 4. Payment history is the dominant signal

`PAY_0` (September repayment status) alone separates the classes: clients in
good standing default at ~12 %, while clients three or more months late default
at more than 60 %. This justifies the engineered features `DELINQ_COUNT`,
`MAX_DELINQ`, `SEVERE_DELINQ_COUNT` and `PAY_0_STATUS`.
""")

    code("""
pay_stats = df.groupby("PAY_0")[TARGET_COLUMN].agg(["mean", "count"]).rename(columns={"mean": "default_rate"})
display(pay_stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
axes[0].bar(pay_stats.index.astype(str), pay_stats["default_rate"], color="#c44e52")
axes[0].set_title("Default rate by PAY_0 (repayment status)")
axes[0].set_xlabel("PAY_0 (-2 = no consumption, -1 = paid, 0 = revolving, n = n months late)")
axes[0].set_ylabel("default rate")
for position, (rate, count) in enumerate(zip(pay_stats["default_rate"], pay_stats["count"])):
    axes[0].text(position, rate, f"{rate:.0%}", ha="center", va="bottom", fontsize=8)

sns.kdeplot(data=df, x="PAY_AMT1", hue=TARGET_COLUMN, clip=(0, 20000), common_norm=False, ax=axes[1])
axes[1].set_title("PAY_AMT1 distribution by outcome")
fig.tight_layout()
plt.show()
""")

    markdown("""
## 5. Correlations and engineered features

The raw numeric block is highly collinear (six monthly bill statements), which is
exactly why the project adds aggregate and trend features
(`BILL_AMT_MEAN/STD/TREND`, `CREDIT_UTILIZATION`, `PAYMENT_RATIO`).
""")

    code("""
engineered = build_features(df)
numeric_features, categorical_features = get_feature_lists()

corr_columns = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "PAY_0"] + [
    "CREDIT_UTILIZATION", "BILL_AMT_TREND", "PAYMENT_RATIO", "DELINQ_COUNT", "MAX_DELINQ"
] + [TARGET_COLUMN]

fig, axis = plt.subplots(figsize=(9, 7))
sns.heatmap(
    engineered[corr_columns].corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    ax=axis,
)
axis.set_title("Correlation matrix: raw vs. engineered features")
fig.tight_layout()
plt.show()

print(f"engineered feature matrix: {len(engineered):,} rows x {len(numeric_features + categorical_features)} model features")
print("numeric:", len(numeric_features), "| categorical:", len(categorical_features))
""")

    code("""
comparison = (
    engineered.groupby(TARGET_COLUMN)[["CREDIT_UTILIZATION", "PAYMENT_RATIO", "DELINQ_COUNT", "BILL_AMT_TREND"]]
    .mean()
    .T
    .rename(columns={0: "no default", 1: "default"})
)
comparison["difference"] = comparison["default"] - comparison["no default"]
display(comparison.round(4))
""")

    markdown("""
## 6. Conclusions and modelling decisions

1. **Class imbalance (~22 %)** → optimise ROC-AUC/PR-AUC, tune the decision
   threshold, report Precision/Recall/F1 at that threshold, and test
   `class_weight="balanced"` for the linear models.
2. **No missing values in the raw data**, but the pipeline still contains
   imputers: they make the service robust to partial API payloads and to future
   data deliveries.
3. **Dirty categories** (`EDUCATION` 0/5/6, `MARRIAGE` 0) are folded into
   "others" by `clean_data`; Great Expectations allows them in the raw contract
   but the model never sees undocumented groups.
4. **Payment history dominates**: `PAY_0` and the delinquency counters are the
   strongest predictors. Aggregating the six months of bill/payment history into
   means, standard deviations, trends and utilisation adds signal that a single
   month cannot provide.
5. **Feature engineering is shared** between training and serving
   (`build_features`), so the API scores raw client records without train/serve
   skew.
6. **Scaling matters for the linear model**, not for the trees; keeping
   `StandardScaler` inside the pipeline is harmless for trees and required for
   `LogisticRegression`.
""")

    notebook["cells"] = cells
    return notebook


def execute_notebook(notebook, timeout: int = 600):
    """Run every cell with the project virtualenv kernel and keep the outputs."""
    from nbconvert.preprocessors import ExecutePreprocessor

    executor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    return executor.preprocess(notebook, {"metadata": {"path": str(PROJECT_ROOT)}})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute the notebook after building it")
    parser.add_argument("--output", default=str(NOTEBOOK_PATH), help="notebook destination")
    args = parser.parse_args(argv)

    notebook = build_notebook()
    if args.execute:
        print("executing the notebook (this may take a minute)...")
        notebook, _ = execute_notebook(notebook)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    print(f"notebook written to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
