# Credit Scoring Model — end-to-end PD pipeline

Сквозной (end-to-end) MLOps-пайплайн для скоринговой модели предсказания вероятности
дефолта (PD-модели) по датасету
[Default of Credit Card Clients (UCI)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).

Проект покрывает весь жизненный цикл модели: контроль версий кода и данных → валидация
входящих данных → feature engineering → обучение и подбор гиперпараметров → трекинг
экспериментов → unit-тесты и CI → контейнеризация и REST API → мониторинг дрифта.

---

## Содержание

- [Результаты](#результаты)
- [Структура репозитория](#структура-репозитория)
- [Быстрый старт](#быстрый-старт)
- [1. Данные и валидация](#1-данные-и-валидация)
- [2. Feature engineering](#2-feature-engineering)
- [3. Модель и подбор гиперпараметров](#3-модель-и-подбор-гиперпараметров)
- [4. MLflow: эксперименты](#4-mlflow-эксперименты)
- [5. DVC: версионирование данных и пайплайн](#5-dvc-версионирование-данных-и-пайплайн)
- [6. Тесты и CI](#6-тесты-и-ci)
- [7. Docker и FastAPI](#7-docker-и-fastapi)
- [8. Мониторинг дрифта (PSI)](#8-мониторинг-дрифта-psi)
- [Соответствие критериям оценки](#соответствие-критериям-оценки)
- [Ограничения и дальнейшие шаги](#ограничения-и-дальнейшие-шаги)

---

## Результаты

| Метрика | Значение (hold-out, 6 000 клиентов) |
|---|---|
| **ROC-AUC** | **0.781** |
| **PR-AUC (average precision)** | **0.557** |
| **KS** | **0.431** |
| F1 (порог 0.29) | 0.552 |
| Precision | 0.537 |
| Recall | 0.569 |
| Brier score | 0.134 |
| Доля дефолтов в выборке | 22.1 % |

Бизнес-интерпретация: 20 % клиентов с наибольшим PD содержат **51.2 %** всех дефолтов
(`top_20pct_capture_rate`), 10 % — 31.5 %.

Модель: `HistGradientBoostingClassifier` (sklearn) внутри `Pipeline` с `Imputer`,
`StandardScaler` и `OneHotEncoder`; гиперпараметры подобраны `RandomizedSearchCV`
(5-fold CV, 25 кандидатов, оптимизация ROC-AUC); порог решения 0.29 подобран на
тренировочной выборке по F1, поэтому тестовая выборка не участвует в выборе модели.
Обучение детерминировано: `random_state` передаётся во все estimators, поэтому два
прогона `dvc repro` дают идентичные метрики.

---

## Структура репозитория

```
credit-scoring-model/
├── data/
│   ├── raw/            # исходный CSV (версионируется DVC)
│   ├── processed/      # train/test после feature engineering (DVC)
│   └── expectations/   # JSON-набор правил Great Expectations + результат валидации
├── models/             # сериализованная модель, model card (DVC)
├── metrics/            # metrics.json — DVC-метрика стадии train
├── reports/
│   ├── figures/        # ROC, PR, confusion matrix, feature importance, drift
│   ├── drift_report.json
│   └── experiments.md  # сравнение экспериментов
├── notebooks/
│   └── 01_eda.ipynb    # EDA с выводами и графиками
├── src/
│   ├── config.py           # пути, схема данных, MLflow/DVC-настройки
│   ├── data/
│   │   ├── io.py           # загрузка CSV и восстановление типов
│   │   ├── make_dataset.py # prepare: очистка, FE, train/test split
│   │   └── validation.py   # Great Expectations suite + CLI
│   ├── features/
│   │   └── build_features.py # очистка и feature engineering (общие для train и serve)
│   ├── models/
│   │   ├── pipeline.py     # sklearn Pipeline + пространства поиска
│   │   ├── train.py        # обучение, метрики, графики, MLflow, артефакты
│   │   ├── experiment.py   # матрица из 6 экспериментов
│   │   ├── evaluate.py     # метрики и визуализации
│   │   └── predict.py      # загрузка бандла, скоринг, risk bands
│   ├── api/
│   │   ├── app.py          # FastAPI: /predict, /predict/batch, /health, /model/info
│   │   └── schemas.py      # Pydantic-схемы запросов и ответов
│   └── monitoring/
│       └── drift.py        # PSI/KS-мониторинг входящего потока
├── scripts/
│   ├── docker_build_run.ps1   # сборка образа, запуск контейнера, smoke-тест
│   ├── smoke_test_api.py      # проверка работающего сервиса
│   ├── build_eda_notebook.py  # генерация и исполнение ноутбука EDA
│   ├── build_presentation.py  # сборка docs/presentation.pptx из артефактов
├── tests/                  # unit- и интеграционные тесты (pytest)
├── .github/workflows/ci-cd.yml
├── Dockerfile
├── LICENSE
├── dvc.yaml
├── params.yaml
├── requirements*.txt
└── pyproject.toml
```

---

## Быстрый старт

```powershell
# 1. Окружение
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt

# 2. Данные: либо dvc pull (если настроен remote), либо положить CSV вручную
.\.venv\Scripts\python -m dvc pull
#   .\data\raw\UCI_Credit_Card.csv  <- исходный файл датасета

# 3. Воспроизвести пайплайн (валидация -> подготовка -> обучение)
.\.venv\Scripts\python -m dvc repro

# 4. Посмотреть эксперименты в UI MLflow
.\.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
#    http://127.0.0.1:5000

# 5. Поднять API
.\.venv\Scripts\python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
#    Swagger: http://127.0.0.1:8000/docs

# 6. Проверить сервис и посчитать дрифт
.\.venv\Scripts\python scripts\smoke_test_api.py --url http://127.0.0.1:8000
.\.venv\Scripts\python -m src.monitoring.drift --api-url http://127.0.0.1:8000 --sample-size 1000

# 7. Тесты и линтеры
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m flake8 src tests
.\.venv\Scripts\python -m black --check src tests

# 8. Docker
powershell -ExecutionPolicy Bypass -File scripts\docker_build_run.ps1
```

---

## 1. Данные и валидация

**Датасет.** 30 000 клиентов банка Тайваня, 23 признака: лимит, пол, образование,
семейное положение, возраст, 6 месяцев статуса погашения (`PAY_0`…`PAY_6`), 6 сумм
выписок (`BILL_AMT1…6`) и 6 сумм платежей (`PAY_AMT1…6`). Целевая переменная
`default.payment.next.month ∈ {0, 1}`, доля дефолтов **22.12 %**.

**Что найдено при аудите данных.**

| Особенность | Решение |
|---|---|
| `EDUCATION` содержит недокументированные коды `0`, `5`, `6` | маппинг в категорию «others» = `4` (`clean_data`) |
| `MARRIAGE` содержит код `0` | маппинг в `3` |
| `PAY_*` выходят за документированный диапазон | клиппинг в `[-2, 8]` |
| Суммы записаны в экспоненциальной нотации (`5e+05`) → pandas читает их как `float64` | слой ingestion `src/data/io.py` приводит их к `int64` |
| Дубликатов `ID` нет, пропусков нет | — |

**Great Expectations.** Набор правил (`src/data/validation.py`, suite
`credit_raw_data_suite`) описывает контракт на сырые данные:

- наличие всех 25 колонок (`expect_table_columns_to_match_set`);
- отсутствие null (`expect_column_values_to_not_be_null`);
- типы данных (`expect_column_values_to_be_of_type`, `int64`);
- диапазоны (`AGE` 18–100, `LIMIT_BAL` 10 000–1 000 000, суммы платежей ≥ 0);
- допустимые значения (`SEX ∈ {1,2}`, `EDUCATION ∈ {0…6}`, `TARGET ∈ {0,1}`);
- уникальность `ID`;
- количество строк (`expect_table_row_count_to_be_between`, 1 000–1 000 000);
- распределение (`expect_column_mean_to_be_between` для `AGE` и доли дефолтов).

Всего **104 проверки**, на исходном файле проходят 104/104. Набор правил экспортируется
в `data/expectations/credit_raw_data_suite.json`, результат прогона — в
`data/expectations/validation_result.json`.

```powershell
.\.venv\Scripts\python -m src.data.validation --input data/raw/UCI_Credit_Card.csv
# suite: credit_raw_data_suite
# expectations: 104/104 passed (100.0%) on 30000 rows
```

Если в новом поступлении данных появится аномалия, команда завершится с кодом `1` и
перечислит упавшие проверки — это же поведение используется в CI и в DVC-стадии
`validate`.

---

## 2. Feature engineering

`src/features/build_features.py` — единственная точка правды для преобразований,
она используется и при обучении, и в API (нет train/serve skew).

Добавленные признаки (20 числовых + 3 категориальных):

| Группа | Признаки |
|---|---|
| Утилизация кредита | `CREDIT_UTILIZATION` = средний bill / лимит, `LIMIT_PER_AGE` |
| Агрегаты по выпискам | `BILL_AMT_MEAN`, `BILL_AMT_STD`, `BILL_AMT_MAX`, `BILL_AMT_MIN`, `BILL_AMT_CHANGE` |
| Тренд по выпискам | `BILL_AMT_TREND` (МНК-наклон за 6 месяцев) |
| Агрегаты по платежам | `PAY_AMT_MEAN`, `PAY_AMT_STD`, `PAY_AMT_MAX`, `PAY_AMT_SUM`, `PAY_AMT_TREND` |
| Платёжная дисциплина | `PAYMENT_RATIO` = сумма платежей / сумма выписок, `BILL_PAYMENT_GAP` |
| Просрочки | `DELINQ_COUNT`, `SEVERE_DELINQ_COUNT`, `MAX_DELINQ`, `DELINQ_TREND` |
| Поведение | `MONTHS_WITHOUT_PAYMENT` |
| Биннинг | `AGE_BIN` (8 интервалов), `LIMIT_BAL_BIN`, `PAY_0_STATUS` |

Итого **46 признаков** (23 исходных + 23 производных). Все деления защищены от
`ZeroDivision` (`_safe_divide`), поэтому признаки остаются конечными при нулевых
выписках. EDA с обоснованием выбора признаков — в
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

---

## 3. Модель и подбор гиперпараметров

```python
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005))]),
         categorical_features),
    ])),
    ("classifier", HistGradientBoostingClassifier()),
])
```

- **Разделение**: стратифицированный train/test 80/20 (`random_state=42`) на стадии
  `prepare`; тест не используется ни для подбора гиперпараметров, ни для выбора порога.
- **Поиск**: `RandomizedSearchCV` (в `params.yaml` можно переключить на `grid`),
  5-fold `StratifiedKFold`, `scoring="roc_auc"`, 25 кандидатов.
- **Порог решения**: F1-оптимальный порог считается на предсказаниях *тренировочной*
  выборки и сохраняется в бандл модели (`threshold = 0.29`).
- **Метрики**: ROC-AUC, PR-AUC, KS, Brier, log-loss, Precision, Recall, F1, матрица
  ошибок, а также бизнес-метрики `top_5/10/20pct_capture_rate`.
- **Артефакты**: `models/credit_default_model.joblib` (pipeline + метаданные),
  `models/model_card.json`, `metrics/metrics.json`, PNG-графики в `reports/figures/`.

Запуск обучения:

```powershell
.\.venv\Scripts\python -m src.models.train --params params.yaml
# или через DVC (кешируется по зависимостям и params.yaml)
.\.venv\Scripts\python -m dvc repro train
```

---

## 4. MLflow: эксперименты

Трекинг включён в `src/models/train.py`: логируются параметры (модель, гиперпараметры,
размер выборки, порог), метрики (train/test), артефакты (ROC, PR, confusion matrix,
threshold sweep, feature importance, score distribution), `model_card.json`,
`test_metrics.json` и сама модель (`mlflow.sklearn.log_model` с регистрацией в
`CreditDefaultModel`).

```powershell
# UI
.\.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# программный просмотр
.\.venv\Scripts\python -c "import mlflow; print(mlflow.search_runs(experiment_names=['credit_default_pd_model'])[['tags.mlflow.runName','metrics.test_roc_auc']])"
```

Матрица экспериментов (`python -m src.models.experiment`) — 6 запусков:

<!-- EXPERIMENTS_TABLE_START -->
| # | Эксперимент | Алгоритм | ROC-AUC | PR-AUC | F1 | Precision | Recall | KS |
|---|---|---|---|---|---|---|---|---|
| 1 | `05_gradient_boosting` | GradientBoosting | **0.7812** | **0.5588** | 0.5485 | 0.5355 | 0.5622 | 0.4335 |
| 2 | `06_hist_gradient_boosting` | HistGradientBoosting (+ поиск) | 0.7804 | 0.5560 | 0.5490 | 0.5145 | 0.5885 | 0.4347 |
| 3 | `03_random_forest` | RandomForest (+ поиск) | 0.7792 | 0.5565 | 0.5472 | 0.5171 | 0.5810 | 0.4288 |
| 4 | `04_extra_trees` | ExtraTrees (+ поиск) | 0.7713 | 0.5534 | 0.5217 | 0.4517 | 0.6172 | 0.4148 |
| 5 | `01_logreg_baseline` | LogisticRegression | 0.7667 | 0.5338 | 0.5338 | 0.5125 | 0.5569 | 0.4104 |
| 6 | `02_logreg_tuned` | LogisticRegression (+ поиск) | 0.7665 | 0.5299 | 0.5309 | 0.5218 | 0.5403 | 0.4095 |

**Выводы по экспериментам.**

- Все градиентные бустинги дают ≈ 0.78 ROC-AUC; разница между HGB и классическим GB
  в пределах шума (0.780–0.781), но HGB обучается в ~25 раз быстрее (1.9 с против
  51 с на fit), поэтому именно он выбран для финального пайплайна.
- Логистическая регрессия даёт 0.767 — хороший интерпретируемый бенчмарк, но уступает
  бустингу ~1.5 п.п. ROC-AUC и 2.5 п.п. PR-AUC.
- Подбор гиперпараметров для логистической регрессии на этой задаче ничего не даёт
  (0.7665 против 0.7667): сигнал в данных в основном нелинейный (просрочки), поэтому
  деревья выигрывают.
- У HGB выше Recall (0.59) при умеренной Precision — следствие выбранного
  F1-оптимального порога 0.29; при пороге 0.5 модель консервативнее.

Полная таблица: [`reports/experiments.md`](reports/experiments.md),
машиночитаемая версия — `reports/experiments.json`.
<!-- EXPERIMENTS_TABLE_END -->

---

## 5. DVC: версионирование данных и пайплайн

```powershell
.\.venv\Scripts\python -m dvc init
.\.venv\Scripts\python -m dvc add data/raw/UCI_Credit_Card.csv   # .dvc-файл в git
.\.venv\Scripts\python -m dvc repro                              # prepare -> validate -> train
.\.venv\Scripts\python -m dvc metrics show
.\.venv\Scripts\python -m dvc plots show
```

`dvc.yaml` описывает три стадии:

| Стадия | Команда | Входы | Выходы |
|---|---|---|---|
| `prepare` | `python -m src.data.make_dataset` | raw CSV, код FE | `data/processed/{train,test}.csv`, `feature_metadata.json` |
| `validate` | `python -m src.data.validation` | raw CSV, suite | `data/expectations/validation_result.json` |
| `train` | `python -m src.models.train` | processed данные, код, `params.yaml` | модель (DVC), `model_card.json`, `metrics/metrics.json`, ROC/PR/CM-графики |

Датасет и модель версионируются DVC (`.dvc`-указатели в git), параметры — в
`params.yaml`, поэтому изменение `params.yaml` инвалидирует кеш стадии `train`.

Настройка удалённого хранилища (опционально):

```powershell
.\.venv\Scripts\python -m dvc remote add -d storage s3://my-bucket/credit-scoring
.\.venv\Scripts\python -m dvc push
```

---

## 6. Тесты и CI

```powershell
.\.venv\Scripts\python -m pytest -q                    # unit-тесты
.\.venv\Scripts\python -m pytest -m "not slow" -q     # быстрый прогон
.\.venv\Scripts\python -m flake8 src tests
.\.venv\Scripts\python -m black --check src tests
```

Покрыто тестами:

| Файл | Что проверяет |
|---|---|
| `tests/test_build_features.py` | очистка категорий, агрегаты, тренды, биннинг, обработка ошибок |
| `tests/test_io.py` | восстановление типов из экспоненциальной нотации, пропуски |
| `tests/test_validation.py` | Great Expectations падает на null/выбросах/типах/дубликатах ID/малом числе строк |
| `tests/test_pipeline.py` | состав пайплайна, обучение всех 5 алгоритмов, imputer, unknown-категории, поиск гиперпараметров |
| `tests/test_metrics.py` | корректность метрик, порог, KS, бизнес-метрики, графики |
| `tests/test_predict.py` | загрузка бандла, скоринг raw/featurised данных, risk bands |
| `tests/test_api.py` | все эндпоинты, валидация payload, 503 без модели |
| `tests/test_drift.py` | PSI/KS, алерты, отправка в API, график |
| `tests/test_train.py` | сквозной `prepare → train`, артефакты, model card |

CI (`.github/workflows/ci-cd.yml`) — 4 джоба:

1. **lint** — `black --check`, `flake8`;
2. **test** — `pytest -m "not slow"` с покрытием, артефакт `coverage.xml`;
3. **data-validation** — Great Expectations на сэмпле `tests/fixtures/sample_credit.csv`
   (1 400 строк, закоммичен, чтобы CI не зависел от DVC-хранилища) + экспорт suite;
4. **docker** — обучение на сэмпле, `docker build`, запуск контейнера и smoke-тест
   `/health` и `/predict`.

---

## 7. Docker и FastAPI

**API** (`src/api/app.py`):

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/` | баннер/проверка живости |
| GET | `/health` | liveness + статус загруженной модели |
| GET | `/model/info` | метаданные модели, схема признаков, метрики |
| POST | `/predict` | скоринг одного клиента |
| POST | `/predict/batch` | скоринг до 5 000 клиентов |

Пример запроса (совместим с примером из задания — обязательны 8 полей, остальные
опциональны):

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"BILL_AMT1":3913,"PAY_AMT1":0}'
```

```json
{
  "default_prediction": 1,
  "default_probability": 0.703715,
  "risk_band": "critical",
  "threshold": 0.29,
  "model_name": "hist_gradient_boosting"
}
```

**Docker.**

```powershell
docker build -t credit-default-api:latest .
docker run -d --name credit-default-api -p 8000:8000 credit-default-api:latest
# либо одной командой: сборка + запуск + smoke-тест
powershell -ExecutionPolicy Bypass -File scripts\docker_build_run.ps1
```

Образ основан на `python:3.12-slim`, работает под непривилегированным пользователем,
содержит `HEALTHCHECK` и запускает `uvicorn`. Для образа, способного также обучать
модель (`mlflow`, `great-expectations`):

```powershell
docker build --build-arg REQUIREMENTS=requirements-train.txt -t credit-default-api:train .
```

---

## 8. Мониторинг дрифта (PSI)

`src/monitoring/drift.py` имитирует поступление новых данных: берёт срез тестовой
выборки, отправляет его в работающий API (`POST /predict/batch`) и сравнивает
распределение PD и ключевых признаков с тренировочной выборкой.

```powershell
.\.venv\Scripts\python -m src.monitoring.drift --api-url http://127.0.0.1:8000 --sample-size 1000
# offline-режим (без запущенного API)
.\.venv\Scripts\python -m src.monitoring.drift --offline
# демонстрация срабатывания алерта
.\.venv\Scripts\python -m src.monitoring.drift --offline --inject-drift
```

- **PSI** считается по квинтильным бинам референса для PD и для признаков
  `LIMIT_BAL`, `AGE`, `PAY_0`, `BILL_AMT1`, `PAY_AMT1`, `CREDIT_UTILIZATION`,
  `DELINQ_COUNT`.
- **Интерпретация**: `PSI < 0.10` — стабильно, `0.10–0.25` — умеренный сдвиг,
  `> 0.25` — значимый сдвиг (алерт).
- Дополнительно: two-sample **KS-тест** и «живые» метрики качества, если в данных есть
  таргет.
- Результаты: `reports/drift_report.json` и `reports/figures/drift_psi.png`;
  код возврата `2` при срабатывании алерта (удобно для cron/airflow).

Фактический прогон на 1 000 клиентов из тестовой выборки (тот же период, что и train):

| Метрика | Реальные данные | `--inject-drift` |
|---|---|---|
| PSI (PD) | 0.029 — stable | **0.533 — significant** |
| KS (PD) | 0.038 (p = 0.47) | 0.301 (p ≈ 2e-40) |
| Средний PD | 0.219 | 0.360 |
| Live ROC-AUC | 0.768 | 0.662 |

То есть на «нормальном» потоке алертов нет, а на искусственно ухудшенной когорте
(больше просрочек, меньше платежей) срабатывают два алерта — по score и по `PAY_0`;
это и есть сценарий, ради которого существует мониторинг.

---

## Соответствие критериям оценки

| № | Критерий | Где реализовано |
|---|---|---|
| 1 | Организация кода и Git | структура `src/`, `tests/`, `notebooks/`, `data/`, `models/`; осмысленные коммиты; README |
| 2 | Подготовка и валидация данных | `src/data/io.py`, `make_dataset.py`, `src/features/build_features.py`, Great Expectations suite (104 проверки), `notebooks/01_eda.ipynb` |
| 3 | Построение и настройка модели | `src/models/pipeline.py` (Imputer/Scaler/OneHotEncoder + модель), `RandomizedSearchCV`, метрики ROC-AUC/PR-AUC/KS/Precision/Recall/F1, ROC-кривая |
| 4 | MLflow Tracking | `src/models/train.py` (параметры, метрики, артефакты, модель + registry), `experiment.py` (6 экспериментов), `reports/experiments.md` |
| 5 | DVC | `dvc.yaml` (prepare/validate/train), `params.yaml`, `.dvc`-файлы, `dvc metrics/plots` |
| 6 | Тестирование и CI | 9 файлов тестов, `pytest`, `black`, `flake8`, GH Actions (lint, test, data-validation, docker) |
| 7 | Docker и FastAPI | `Dockerfile`, `src/api/app.py` (`/predict`), `scripts/docker_build_run.ps1`, smoke-тест |
| 8 | Мониторинг (дрифт) | `src/monitoring/drift.py` (PSI + KS + алерты), `reports/drift_report.json` |
| 9 | Демонстрация | `docs/presentation.pptx`, `reports/experiments.md`, MLflow UI |

---

## Ограничения и дальнейшие шаги

- **Дисбаланс классов** (~22 % дефолтов) обрабатывается порогом и метриками, но не
  oversampling'ом: SMOTE не применялся, чтобы не усложнять пайплайн и не ломать
  калибровку PD.
- **Калибровка**: Brier score и log-loss логируются, но отдельная калибровка
  (`CalibratedClassifierCV`) не проводилась — это следующий шаг для продакшн-PD, где
  важна точность вероятностей, а не только ранжирование.
- **Дрифт** оценивается по PSI/KS; автоматического переобучения по триггеру нет —
  скрипт предназначен для запуска по расписанию (cron/Airflow) с кодом возврата 2.
- **Данные** 2005 года, поэтому метрики не переносятся напрямую на текущий портфель;
  пайплайн воспроизводим и заменяем на актуальные выгрузки без изменения кода.
- **DVC remote** не настроен (локальное хранилище); для командной работы достаточно
  добавить S3/GCS/Azure remote и выполнить `dvc push`.

---

## Лицензия и данные

Датасет распространяется UCI Machine Learning Repository для исследовательских целей
(I-Cheng Yeh, *Default of Credit Card Clients*, 2016). Код проекта — MIT (см. [LICENSE](LICENSE)).
