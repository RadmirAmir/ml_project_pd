"""Metric computation and diagnostic plots for the PD model.

The evaluation focuses on what a credit-risk team actually looks at:

* discrimination – ROC-AUC, PR-AUC (average precision) and the KS statistic;
* calibration     – Brier score and log loss;
* business view   – Precision/Recall/F1 at an explicit decision threshold,
  including a search for the F1-optimal threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _threshold_grid(thresholds: np.ndarray | None = None) -> np.ndarray:
    return np.linspace(0.05, 0.95, 91) if thresholds is None else thresholds


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between the default and non-default score CDFs."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def find_best_threshold(y_true, y_proba, metric: str = "f1") -> tuple[float, float]:
    """Scan thresholds and return ``(best_threshold, best_score)``.

    ``metric`` is either ``"f1"`` or ``"youden"`` (sensitivity + specificity - 1).
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    best_threshold, best_score = 0.5, -np.inf
    for threshold in _threshold_grid():
        y_pred = (y_proba >= threshold).astype(int)
        if metric == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            score = sensitivity + specificity - 1
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def compute_metrics(
    y_true,
    y_proba,
    threshold: float = 0.5,
    include_threshold_search: bool = True,
) -> dict:
    """Compute the full metric dictionary used in MLflow and ``metrics.json``."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics on an empty sample")

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "ks": ks_statistic(y_true, y_proba),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "n_samples": int(len(y_true)),
        "default_rate": float(y_true.mean()),
        "mean_predicted_pd": float(y_proba.mean()),
    }

    if include_threshold_search:
        best_threshold, best_f1 = find_best_threshold(y_true, y_proba, metric="f1")
        youden_threshold, youden_score = find_best_threshold(y_true, y_proba, metric="youden")
        metrics.update(
            {
                "best_f1_threshold": best_threshold,
                "best_f1": float(best_f1),
                "youden_threshold": youden_threshold,
                "youden_score": float(youden_score),
            }
        )

    return metrics


def top_k_metrics(y_true, y_proba, fractions=(0.05, 0.1, 0.2)) -> dict:
    """Business metrics for the riskiest ``fraction`` of the portfolio.

    ``capture_rate`` is the share of all defaults that land in the selected
    (highest PD) segment – the number a collections team cares about.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    order = np.argsort(-y_proba)
    result = {}
    for fraction in fractions:
        k = max(1, int(round(len(y_true) * fraction)))
        selected = order[:k]
        captured = int(y_true[selected].sum())
        result[f"top_{int(fraction * 100)}pct_capture_rate"] = float(captured / max(1, y_true.sum()))
        result[f"top_{int(fraction * 100)}pct_precision"] = float(captured / k)
    return result


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )
    return plt


def plot_roc_curve(y_true, y_proba, path: Path | str, auc: float | None = None) -> Path:
    """Save the ROC curve with the AUC in the legend."""
    plt = _prepare_matplotlib()
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), y_proba)
    auc = roc_auc_score(np.asarray(y_true).astype(int), y_proba) if auc is None else auc

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"PD model (ROC-AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve – credit default model")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_precision_recall_curve(y_true, y_proba, path: Path | str) -> Path:
    """Save the precision-recall curve."""
    from sklearn.metrics import precision_recall_curve

    plt = _prepare_matplotlib()
    y_true = np.asarray(y_true).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(recall, precision, color="#2ca02c", lw=2, label=f"PR-AUC = {ap:.4f}")
    ax.axhline(y_true.mean(), color="grey", linestyle="--", lw=1, label=f"Baseline = {y_true.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve – credit default model")
    ax.legend(loc="upper right")
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_confusion_matrix(y_true, y_proba, path: Path | str, threshold: float = 0.5) -> Path:
    """Save an annotated confusion matrix for the chosen threshold."""
    plt = _prepare_matplotlib()
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            colour = "white" if value > matrix.max() / 2 else "black"
            ax.text(column, row, f"{value:,}", ha="center", va="center", color=colour, fontsize=12)
    ax.set_xticks([0, 1], labels=["Predicted: no default", "Predicted: default"])
    ax.set_yticks([0, 1], labels=["Actual: no default", "Actual: default"])
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_threshold_sweep(y_true, y_proba, path: Path | str) -> Path:
    """Save precision/recall/F1 as a function of the decision threshold."""
    plt = _prepare_matplotlib()
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    thresholds = _threshold_grid()
    f1_scores, precisions, recalls = [], [], []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(thresholds, precisions, label="Precision", color="#1f77b4")
    ax.plot(thresholds, recalls, label="Recall", color="#d62728")
    ax.plot(thresholds, f1_scores, label="F1", color="#2ca02c", lw=2)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / recall / F1 vs. threshold")
    ax.legend(loc="best")
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_feature_importance(pipeline, path: Path | str, top_n: int = 20) -> Path | None:
    """Save the top-``top_n`` feature importances (or coefficients)."""
    plt = _prepare_matplotlib()
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover
        return None

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_).ravel())
    else:  # pragma: no cover - ensembles without native importances
        return None

    if len(values) != len(names):
        return None

    order = np.argsort(values)[::-1][:top_n]
    labels = [names[i].replace("num__", "").replace("cat__", "") for i in order]
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(order) + 1.6))
    ax.barh(range(len(order))[::-1], values[order], color="#4c72b0")
    ax.set_yticks(range(len(order))[::-1], labels=labels)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{len(order)} features")
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_score_distribution(y_true, y_proba, path: Path | str) -> Path:
    """Save the predicted-PD distribution split by the actual outcome."""
    plt = _prepare_matplotlib()
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.hist(y_proba[y_true == 0], bins=40, alpha=0.65, label="No default (0)", color="#4c72b0", density=True)
    ax.hist(y_proba[y_true == 1], bins=40, alpha=0.65, label="Default (1)", color="#c44e52", density=True)
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by actual outcome")
    ax.legend()
    fig.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return destination
