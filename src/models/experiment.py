"""Запуск матрицы экспериментов с логированием каждого прогона в MLflow.

Использование::

    python -m src.models.experiment                # все эксперименты (6 прогонов)
    python -m src.models.experiment --only logreg  # фильтр по подстроке имени
    python -m src.models.experiment --comparison reports/experiments.md

Каждый эксперимент обучает один алгоритм (при необходимости — с подбором
гиперпараметров), пишет параметры, метрики и артефакты в MLflow и добавляет
строку в сравнительную таблицу. Лучший прогон копируется в ``models/`` и
``metrics/metrics.json``, поэтому DVC-стадия и API всегда используют модель-победителя.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import REPORTS_DIR, write_json
from src.models.train import fit_and_evaluate

LOGGER = logging.getLogger(__name__)

#: Матрица экспериментов (не менее пяти прогонов).
EXPERIMENTS: list[dict] = [
    {
        "name": "01_logreg_baseline",
        "model_name": "logistic_regression",
        "search_method": "none",
        "model_params": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000},
        "note": "Базовый вариант логистической регрессии — интерпретируемый бенчмарк",
    },
    {
        "name": "02_logreg_tuned",
        "model_name": "logistic_regression",
        "search_method": "randomized",
        "n_iter": 10,
        "note": "Логистическая регрессия с RandomizedSearchCV (C, веса классов, решатель)",
    },
    {
        "name": "03_random_forest",
        "model_name": "random_forest",
        "search_method": "randomized",
        "n_iter": 8,
        "note": "Случайный лес — бэггинг декоррелированных деревьев",
    },
    {
        "name": "04_extra_trees",
        "model_name": "extra_trees",
        "search_method": "randomized",
        "n_iter": 6,
        "note": "Extra Trees — усиленная регуляризация",
    },
    {
        "name": "05_gradient_boosting",
        "model_name": "gradient_boosting",
        "search_method": "none",
        # Классический (негистограммный) бустинг медленный, поэтому обучается
        # с компактной фиксированной конфигурацией вместо полного перебора.
        "model_params": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.9},
        "note": "Классический градиентный бустинг (компактная фиксированная конфигурация)",
    },
    {
        "name": "06_hist_gradient_boosting",
        "model_name": "hist_gradient_boosting",
        "search_method": "randomized",
        "n_iter": 15,
        "note": "Гистограммный градиентный бустинг (быстрый, кандидат в продакшн)",
    },
]


def run_experiments(
    experiments: list[dict] | None = None,
    cv: int = 3,
    log_to_mlflow: bool = True,
    register_model: bool = False,
    save_artifacts: bool = False,
) -> list[dict]:
    """Обучить все эксперименты и вернуть список сводок результатов."""
    selected = experiments if experiments is not None else EXPERIMENTS
    results = []
    for experiment in selected:
        LOGGER.info("=" * 70)
        LOGGER.info("эксперимент %s (%s)", experiment["name"], experiment["model_name"])
        result = fit_and_evaluate(
            model_name=experiment["model_name"],
            search_method=experiment.get("search_method", "randomized"),
            n_iter=experiment.get("n_iter", 10),
            cv=cv,
            model_params=experiment.get("model_params"),
            log_to_mlflow=log_to_mlflow,
            run_name=experiment["name"],
            register_model=register_model,
            save_artifacts=save_artifacts,
            tags={"experiment_note": experiment.get("note", "")},
        )
        metrics = result["test_metrics"]
        results.append(
            {
                "name": experiment["name"],
                "model_name": experiment["model_name"],
                "search_method": result["search_method"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ks": metrics["ks"],
                "threshold": result["threshold"],
                "mlflow_run_id": result.get("mlflow_run_id"),
                "best_params": result.get("search", {}).get("best_params", experiment.get("model_params") or {}),
            }
        )
    return results


def comparison_table(results: list[dict]) -> str:
    """Оформить сравнение экспериментов как Markdown-таблицу."""
    header = (
        "| # | Эксперимент | Модель | ROC-AUC | PR-AUC | F1 | Precision | Recall | KS |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for index, row in enumerate(sorted(results, key=lambda item: item["roc_auc"], reverse=True), start=1):
        rows.append(
            f"| {index} | {row['name']} | {row['model_name']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | "
            f"{row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['ks']:.4f} |"
        )
    return "\n".join([header, *rows])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, help="фильтр по подстроке имени эксперимента")
    parser.add_argument("--cv", type=int, default=3, help="число фолдов кросс-валидации")
    parser.add_argument("--no-mlflow", action="store_true", help="не логировать прогоны в MLflow")
    parser.add_argument("--comparison", default=str(REPORTS_DIR / "experiments.md"), help="путь к Markdown-отчёту")
    parser.add_argument("--json", default=str(REPORTS_DIR / "experiments.json"), help="путь к JSON-отчёту")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _parse_args(argv)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [item for item in EXPERIMENTS if args.only.lower() in item["name"].lower()]
        if not experiments:
            LOGGER.error("не найден ни один эксперимент по фильтру %r", args.only)
            return 1

    results = run_experiments(experiments, cv=args.cv, log_to_mlflow=not args.no_mlflow)
    table = comparison_table(results)

    Path(args.comparison).parent.mkdir(parents=True, exist_ok=True)
    Path(args.comparison).write_text("# Сравнение экспериментов\n\n" + table + "\n", encoding="utf-8")
    write_json({"experiments": results}, Path(args.json))
    print("\n" + table + "\n")
    print(f"сравнение сохранено в {args.comparison}")

    best = max(results, key=lambda item: item["roc_auc"])
    print(f"лучший эксперимент: {best['name']} (ROC-AUC = {best['roc_auc']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
