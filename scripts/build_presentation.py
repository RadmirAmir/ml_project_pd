"""Build the defence presentation (docs/presentation.pptx).

The deck is generated from the real artifacts of the project (metrics, figures,
experiment table), so re-running the pipeline and this script keeps the slides
in sync with the code.

Usage::

    python scripts/build_presentation.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT_ROOT / "reports" / "figures"
OUTPUT = PROJECT_ROOT / "docs" / "presentation.pptx"

NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x75, 0xB6)
LIGHT_BLUE = RGBColor(0xDE, 0xEA, 0xF6)
PALE = RGBColor(0xF5, 0xF7, 0xFA)
GREY = RGBColor(0x59, 0x59, 0x59)
DARK = RGBColor(0x26, 0x26, 0x26)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x37, 0x7D, 0x22)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# --------------------------------------------------------------------------- #
# data (read once, so the slides always match the artifacts)
# --------------------------------------------------------------------------- #
METRICS = json.loads((PROJECT_ROOT / "metrics" / "metrics.json").read_text(encoding="utf-8"))
MODEL_CARD = json.loads((PROJECT_ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
DRIFT_STABLE = json.loads((PROJECT_ROOT / "reports" / "drift_report.json").read_text(encoding="utf-8"))
DRIFT_ALERT = json.loads((PROJECT_ROOT / "reports" / "drift_report_alert.json").read_text(encoding="utf-8"))
EXPERIMENTS = json.loads((PROJECT_ROOT / "reports" / "experiments.json").read_text(encoding="utf-8"))["experiments"]

#: Python files covered by black/flake8 (used on the CI slide).
N_PY_FILES = len(
    list(PROJECT_ROOT.glob("src/**/*.py"))
    + list(PROJECT_ROOT.glob("tests/**/*.py"))
    + list(PROJECT_ROOT.glob("scripts/*.py"))
)

#: Demo request/response captured from the running service (see README).
_EXAMPLE_FILE = PROJECT_ROOT / "reports" / "example_prediction.json"
EXAMPLE = json.loads(_EXAMPLE_FILE.read_text(encoding="utf-8")) if _EXAMPLE_FILE.exists() else {}
EXAMPLE_PD = round(EXAMPLE.get("response", {}).get("default_probability", 0.7037), 4)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _textbox(
    slide,
    left,
    top,
    width,
    height,
    text="",
    size=18,
    bold=False,
    colour=DARK,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    font="Calibri",
    italic=False,
):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = colour
    run.font.name = font
    return box


def _bullets(slide, left, top, width, height, items, size=16, spacing=9, bullet_colour=BLUE):
    """Render a bullet list; an item may be ``("Bold lead-in", "rest of text")``."""
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(spacing)
        paragraph.line_spacing = 1.0
        marker = paragraph.add_run()
        marker.text = "▪  "
        marker.font.size = Pt(size)
        marker.font.color.rgb = bullet_colour
        marker.font.name = "Calibri"
        if isinstance(item, tuple):
            lead, rest = item
            head = paragraph.add_run()
            head.text = f"{lead} "
            head.font.size = Pt(size)
            head.font.bold = True
            head.font.color.rgb = NAVY
            head.font.name = "Calibri"
            body = paragraph.add_run()
            body.text = rest
            body.font.size = Pt(size)
            body.font.color.rgb = DARK
            body.font.name = "Calibri"
        else:
            run = paragraph.add_run()
            run.text = item
            run.font.size = Pt(size)
            run.font.color.rgb = DARK
            run.font.name = "Calibri"
    return box


def _title_bar(slide, title, subtitle=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(1.02))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False
    _textbox(
        slide,
        Inches(0.5),
        Inches(0.12),
        Inches(12.3),
        Inches(0.8),
        title,
        size=27,
        bold=True,
        colour=WHITE,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    if subtitle:
        _textbox(slide, Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.4), subtitle, size=13, colour=GREY)


def _footer(slide, text, page):
    _textbox(slide, Inches(0.5), Inches(7.05), Inches(11.0), Inches(0.3), text, size=9, colour=GREY)
    _textbox(
        slide,
        Inches(12.1),
        Inches(7.05),
        Inches(0.7),
        Inches(0.3),
        str(page),
        size=9,
        colour=GREY,
        align=PP_ALIGN.RIGHT,
    )


def _table(
    slide,
    left,
    top,
    width,
    height,
    headers,
    rows,
    col_widths=None,
    font_size=12,
    header_size=12,
    align_first_left=True,
):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width, height)
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for index, share in enumerate(col_widths):
            table.columns[index].width = Emu(int(width * share / total))

    for column, header in enumerate(headers):
        cell = table.cell(0, column)
        cell.text = ""
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.LEFT if (column == 0 and align_first_left) else PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = str(header)
        run.font.size = Pt(header_size)
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.name = "Calibri"
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY

    for row_index, row in enumerate(rows, start=1):
        for column, value in enumerate(row):
            cell = table.cell(row_index, column)
            cell.text = ""
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if (column == 0 and align_first_left) else PP_ALIGN.CENTER
            run = paragraph.add_run()
            run.text = str(value)
            run.font.size = Pt(font_size)
            run.font.name = "Calibri"
            run.font.color.rgb = DARK
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BLUE if row_index % 2 == 0 else WHITE
    return table


def _code_panel(slide, left, top, width, height, lines, font_size=12):
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    panel.fill.solid()
    panel.fill.fore_color.rgb = PALE
    panel.line.color.rgb = RGBColor(0xD9, 0xE2, 0xF3)
    panel.shadow.inherit = False
    frame = panel.text_frame
    frame.word_wrap = True
    frame.margin_left = Inches(0.18)
    frame.margin_top = Inches(0.1)
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.LEFT
        run = paragraph.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.name = "Consolas"
        run.font.color.rgb = GREEN if "passed" in line else (NAVY if line.startswith("$") else DARK)
    return panel


def _add_image(slide, path: Path, left, top, width=None, height=None):
    if not path.exists():
        _textbox(
            slide,
            left,
            top,
            Inches(5),
            Inches(0.4),
            f"[missing figure: {path.name}]",
            size=11,
            colour=RGBColor(0xC0, 0x00, 0x00),
        )
        return None
    return slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def _new_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def build() -> Presentation:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # ------------------------------------------------------------------ #
    # 1. Title
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    background.fill.solid()
    background.fill.fore_color.rgb = NAVY
    background.line.fill.background()
    background.shadow.inherit = False

    _textbox(
        slide,
        Inches(0.8),
        Inches(1.5),
        Inches(11.8),
        Inches(1.0),
        "PD-модель кредитного дефолта",
        size=44,
        bold=True,
        colour=WHITE,
    )
    _textbox(
        slide,
        Inches(0.8),
        Inches(2.6),
        Inches(11.8),
        Inches(0.7),
        "сквозной MLOps-пайплайн: данные → валидация → модель → API → мониторинг",
        size=20,
        colour=RGBColor(0xBD, 0xD7, 0xEE),
    )
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.5), Inches(3.2), Inches(0.05))
    line.fill.solid()
    line.fill.fore_color.rgb = BLUE
    line.line.fill.background()

    _textbox(
        slide,
        Inches(0.8),
        Inches(3.9),
        Inches(11.8),
        Inches(0.4),
        "Итоговый проект по дисциплине",
        size=15,
        colour=WHITE,
    )
    _textbox(
        slide,
        Inches(0.8),
        Inches(4.3),
        Inches(11.8),
        Inches(0.6),
        "«Автоматизация процессов разработки и тестирования моделей машинного обучения»",
        size=17,
        bold=True,
        colour=WHITE,
    )
    _textbox(
        slide,
        Inches(0.8),
        Inches(5.35),
        Inches(11.8),
        Inches(0.5),
        f"Датасет: Default of Credit Card Clients (UCI) · 30 000 клиентов · ROC-AUC {METRICS['roc_auc']:.3f}",
        size=15,
        colour=RGBColor(0xBD, 0xD7, 0xEE),
    )
    _textbox(
        slide,
        Inches(0.8),
        Inches(6.55),
        Inches(11.8),
        Inches(0.4),
        "ML-инженер кредитного отдела · GitHub: credit-scoring-model",
        size=11,
        colour=RGBColor(0x8E, 0xA9, 0xC9),
    )
    _notes(
        slide, "Проект закрывает весь жизненный цикл PD-модели: от валидации данных до мониторинга дрифта в продакшне."
    )

    # ------------------------------------------------------------------ #
    # 2. Task and data
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "Задача и данные")
    _bullets(
        slide,
        Inches(0.5),
        Inches(1.4),
        Inches(6.3),
        Inches(5.3),
        [
            ("Роль:", "ML-инженер кредитного отдела банка — автоматизировать PD-модель."),
            (
                "Данные:",
                "30 000 клиентов банка Тайваня, 23 признака: лимит, демография, 6 месяцев статуса погашения, выписки, платежи.",
            ),
            ("Таргет:", "default.payment.next.month ∈ {0, 1} — дефолт в следующем месяце."),
            ("Дисбаланс:", "22.1 % дефолтов → ROC-AUC, PR-AUC, KS, Precision/Recall/F1 вместо accuracy."),
            (
                "Требования:",
                "воспроизводимость (DVC), трекинг (MLflow), тесты и CI, Docker + FastAPI, мониторинг дрифта.",
            ),
        ],
        size=15,
        spacing=10,
    )
    _table(
        slide,
        Inches(7.0),
        Inches(1.45),
        Inches(5.8),
        Inches(2.35),
        ["Характеристика", "Значение"],
        [
            ["Клиентов", "30 000"],
            ["Признаков (raw)", "23"],
            ["Доля дефолтов", "22.1 %"],
            ["Train / test", "24 000 / 6 000"],
            ["Пропусков", "0"],
        ],
        col_widths=[3, 2],
        font_size=12,
    )
    _add_image(slide, FIGURES / "score_distribution.png", Inches(7.15), Inches(3.9), width=Inches(4.6))
    _footer(slide, "Источник: UCI ML Repository, Default of Credit Card Clients (Yeh, 2016)", 2)
    _notes(slide, "Датасет классический, но с грязными категориями — это на следующем слайде.")

    # ------------------------------------------------------------------ #
    # 3. Architecture
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "Архитектура пайплайна")

    stages = [
        ("Raw CSV", "30 000 × 25"),
        ("Валидация", "Great Expectations"),
        ("Подготовка", "FE + split"),
        ("Обучение", "sklearn + поиск"),
        ("Артефакты", "DVC + registry"),
        ("Сервис", "FastAPI + Docker"),
        ("Мониторинг", "PSI / KS"),
    ]
    box_width = Inches(1.66)
    gap = Inches(0.14)
    left = Inches(0.42)
    top = Inches(1.5)
    for index, (label, caption) in enumerate(stages):
        x = left + index * (box_width + gap)
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, top, box_width, Inches(1.05))
        shape.fill.solid()
        shape.fill.fore_color.rgb = NAVY if index % 2 == 0 else BLUE
        shape.line.fill.background()
        shape.shadow.inherit = False
        frame = shape.text_frame
        frame.word_wrap = True
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = label
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.name = "Calibri"
        _textbox(
            slide, x, top + Inches(1.12), box_width, Inches(0.3), caption, size=9, colour=GREY, align=PP_ALIGN.CENTER
        )

    _bullets(
        slide,
        Inches(0.5),
        Inches(3.05),
        Inches(6.2),
        Inches(3.6),
        [
            ("Код:", "модули src/data, src/features, src/models, src/api, src/monitoring + tests + notebooks."),
            ("Версионирование:", "git для кода и конфигов, DVC для датасета, моделей и метрик."),
            ("Воспроизводимость:", "dvc repro пересобирает все три стадии из params.yaml."),
            ("Единая точка FE:", "build_features используется и при обучении, и в API — нет train/serve skew."),
        ],
        size=14,
        spacing=10,
    )

    cards = [
        ("CI", "black + flake8 + pytest + Great Expectations + docker build"),
        ("CD", "образ с моделью, /health, smoke-тест /predict, PSI-мониторинг по расписанию"),
    ]
    for index, (label, text) in enumerate(cards):
        y = Inches(3.1 + index * 1.65)
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.95), y, Inches(5.95), Inches(1.35))
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_BLUE
        shape.line.color.rgb = BLUE
        shape.shadow.inherit = False
        frame = shape.text_frame
        frame.word_wrap = True
        frame.margin_left = Inches(0.15)
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.LEFT
        run = paragraph.add_run()
        run.text = f"{label}  ·  "
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = NAVY
        run.font.name = "Calibri"
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(12)
        run.font.color.rgb = DARK
        run.font.name = "Calibri"
    _footer(slide, "Все стадии описаны декларативно: dvc.yaml, params.yaml, .github/workflows/ci-cd.yml", 3)
    _notes(
        slide,
        "Структура соответствует примеру из задания: data → src/{data,features,models,api} → tests → dvc.yaml → Dockerfile.",
    )

    # ------------------------------------------------------------------ #
    # 4. Data quality + validation
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "Качество данных и валидация", "Great Expectations: 104 проверки, 104/104 на исходном файле")
    _table(
        slide,
        Inches(0.5),
        Inches(1.55),
        Inches(7.0),
        Inches(3.3),
        ["Найденная особенность", "Решение"],
        [
            ["EDUCATION: коды 0, 5, 6 вне документации", "→ категория «others» = 4"],
            ["MARRIAGE: код 0 вне документации", "→ категория «others» = 3"],
            ["PAY_* вне диапазона [-2, 9]", "клиппинг в [-2, 8]"],
            ["Суммы в нотации 5e+05 → pandas даёт float", "ingestion приводит к int64"],
            ["Дубликатов ID и пропусков нет", "—"],
        ],
        col_widths=[4.4, 2.6],
        font_size=11,
    )
    _bullets(
        slide,
        Inches(7.75),
        Inches(1.55),
        Inches(5.1),
        Inches(3.3),
        [
            (
                "Что проверяем:",
                "колонки, null, типы, диапазоны, допустимые значения, уникальность ID, число строк, средние.",
            ),
            ("Контракт как код:", "suite строится программно → ревью в git; JSON экспортируется в data/expectations."),
            ("Провал = остановка:", "exit code 1 в CLI, падение CI-джоба и DVC-стадии validate."),
        ],
        size=13,
        spacing=10,
    )
    _code_panel(
        slide,
        Inches(0.5),
        Inches(5.1),
        Inches(12.35),
        Inches(1.5),
        [
            "$ python -m src.data.validation --input data/raw/UCI_Credit_Card.csv",
            "suite: credit_raw_data_suite",
            "expectations: 104/104 passed (100.0%) on 30000 rows",
        ],
        font_size=12,
    )
    _footer(slide, "src/data/validation.py · data/expectations/credit_raw_data_suite.json", 4)
    _notes(
        slide,
        "Тесты валидации специально ломают данные: null, AGE=150, SEX=5, дубли ID, 10 строк — suite обязана упасть.",
    )

    # ------------------------------------------------------------------ #
    # 5. Feature engineering
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "Feature engineering", "23 исходных + 23 производных признака = 46")
    _table(
        slide,
        Inches(0.5),
        Inches(1.55),
        Inches(7.2),
        Inches(4.0),
        ["Группа", "Признаки"],
        [
            ["Утилизация", "CREDIT_UTILIZATION, LIMIT_PER_AGE"],
            ["Выписки", "BILL_AMT_MEAN/STD/MAX/MIN/CHANGE, BILL_AMT_TREND"],
            ["Платежи", "PAY_AMT_MEAN/STD/MAX/SUM/TREND"],
            ["Дисциплина", "PAYMENT_RATIO, BILL_PAYMENT_GAP"],
            ["Просрочки", "DELINQ_COUNT, SEVERE_DELINQ_COUNT, MAX_DELINQ, DELINQ_TREND"],
            ["Поведение", "MONTHS_WITHOUT_PAYMENT"],
            ["Биннинг", "AGE_BIN (8), LIMIT_BAL_BIN, PAY_0_STATUS"],
        ],
        col_widths=[1.7, 5.5],
        font_size=11,
    )
    _add_image(slide, FIGURES / "feature_importance.png", Inches(7.95), Inches(1.45), width=Inches(4.9))
    _bullets(
        slide,
        Inches(0.5),
        Inches(5.75),
        Inches(7.1),
        Inches(1.1),
        [
            (
                "Защита от деления на ноль:",
                "_safe_divide → нулевые выписки дают 0, а не inf/NaN; наклон за 6 месяцев считается МНК-векторизованно.",
            ),
            ("Почему агрегаты:", "один месяц не показывает динамику; тренды и утилизация дают модели контекст."),
        ],
        size=12,
        spacing=6,
    )
    _footer(slide, "src/features/build_features.py · обоснование признаков — notebooks/01_eda.ipynb", 5)
    _notes(slide, "Важность признаков подтверждает выводы EDA: PAY_0 и просрочки доминируют.")

    # ------------------------------------------------------------------ #
    # 6. Model and search
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "Модель и подбор гиперпараметров")
    _table(
        slide,
        Inches(0.5),
        Inches(1.55),
        Inches(6.2),
        Inches(2.3),
        ["Блок пайплайна", "Состав"],
        [
            ["Числовые", "SimpleImputer(median) + StandardScaler"],
            ["Категориальные", "SimpleImputer(most_frequent) + OneHotEncoder"],
            ["Классификатор", "HistGradientBoostingClassifier"],
        ],
        col_widths=[1.8, 4.4],
        font_size=12,
    )
    _bullets(
        slide,
        Inches(0.5),
        Inches(4.15),
        Inches(6.2),
        Inches(2.5),
        [
            ("Поиск:", "RandomizedSearchCV, 5-fold StratifiedKFold, 25 кандидатов, метрика ROC-AUC."),
            ("Порог:", "F1-оптимальный порог считается на train и сохраняется в бандл — тест не участвует."),
            ("Почему HGB:", "то же качество, что у классического GB, но ~25× быстрее (1.9 с против 51 с)."),
        ],
        size=13,
        spacing=10,
    )
    _add_image(slide, FIGURES / "threshold_sweep.png", Inches(6.9), Inches(1.45), width=Inches(5.7))
    _textbox(
        slide,
        Inches(6.9),
        Inches(5.75),
        Inches(5.95),
        Inches(0.4),
        f"Порог решения {MODEL_CARD['decision_threshold']:.2f} (подобран на train по F1)",
        size=13,
        bold=True,
        colour=NAVY,
    )
    _textbox(
        slide,
        Inches(6.9),
        Inches(6.2),
        Inches(5.95),
        Inches(0.5),
        f"F1 {METRICS['f1']:.2f} · Precision {METRICS['precision']:.2f} · Recall {METRICS['recall']:.2f} · "
        f"KS {METRICS['ks']:.2f}",
        size=12,
        colour=DARK,
    )
    _footer(slide, "src/models/pipeline.py · src/models/train.py", 6)
    _notes(
        slide,
        "Pipeline целиком сериализуется в joblib-бандл вместе с порогом и метаданными — API не дублирует препроцессинг.",
    )

    # ------------------------------------------------------------------ #
    # 7. Results
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(
        slide,
        "Результаты на отложенной выборке",
        f"6 000 клиентов · порог {MODEL_CARD['decision_threshold']:.2f} · тест не участвовал в выборе модели",
    )
    _table(
        slide,
        Inches(0.5),
        Inches(1.55),
        Inches(5.3),
        Inches(3.35),
        ["Метрика", "Значение"],
        [
            ["ROC-AUC", f"{METRICS['roc_auc']:.4f}"],
            ["PR-AUC", f"{METRICS['pr_auc']:.4f}"],
            ["KS", f"{METRICS['ks']:.4f}"],
            ["F1", f"{METRICS['f1']:.4f}"],
            ["Precision", f"{METRICS['precision']:.4f}"],
            ["Recall", f"{METRICS['recall']:.4f}"],
            ["Brier score", f"{METRICS['brier_score']:.4f}"],
        ],
        col_widths=[1, 1],
        font_size=12,
    )
    _table(
        slide,
        Inches(0.5),
        Inches(5.25),
        Inches(5.3),
        Inches(1.2),
        ["Топ-риск", "Захват дефолтов"],
        [
            ["10 % клиентов", f"{METRICS['top_10pct_capture_rate']:.1%}"],
            ["20 % клиентов", f"{METRICS['top_20pct_capture_rate']:.1%}"],
        ],
        col_widths=[1.4, 1.2],
        font_size=12,
    )
    _add_image(slide, FIGURES / "roc_curve.png", Inches(6.3), Inches(1.45), width=Inches(6.5))
    _footer(slide, "metrics/metrics.json · reports/figures/roc_curve.png", 7)
    _notes(slide, "20 % самых рискованных клиентов содержат половину всех дефолтов — это бизнес-ценность модели.")

    # ------------------------------------------------------------------ #
    # 8. MLflow experiments
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "MLflow: шесть экспериментов", "параметры, метрики, графики и модель логируются в каждый run")
    rows = []
    for experiment in sorted(EXPERIMENTS, key=lambda item: item["roc_auc"], reverse=True):
        rows.append(
            [
                experiment["name"].split("_", 1)[1],
                f"{experiment['roc_auc']:.4f}",
                f"{experiment['pr_auc']:.4f}",
                f"{experiment['f1']:.4f}",
                f"{experiment['ks']:.4f}",
            ]
        )
    _table(
        slide,
        Inches(0.5),
        Inches(1.55),
        Inches(6.9),
        Inches(3.4),
        ["Эксперимент", "ROC-AUC", "PR-AUC", "F1", "KS"],
        rows,
        col_widths=[2.9, 1.2, 1.2, 1.0, 1.0],
        font_size=11,
    )
    _bullets(
        slide,
        Inches(0.5),
        Inches(5.15),
        Inches(6.9),
        Inches(1.6),
        [
            (
                "Вывод:",
                "градиентные бустинги ≈ 0.78 ROC-AUC; логистическая регрессия 0.767 — разрыв из-за нелинейности просрочек.",
            ),
            (
                "Что логируется:",
                "params, train/test-метрики, ROC/PR/CM/importance, model_card.json и модель в registry CreditDefaultModel.",
            ),
        ],
        size=12,
        spacing=8,
    )
    _add_image(slide, FIGURES / "pr_curve.png", Inches(7.7), Inches(1.5), width=Inches(5.15))
    _footer(slide, "python -m src.models.experiment · UI: mlflow ui --backend-store-uri sqlite:///mlflow.db", 8)
    _notes(slide, "В UI MLflow можно сравнить run'ы, открыть артефакты и версии модели в registry.")

    # ------------------------------------------------------------------ #
    # 9. DVC
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "DVC: версионирование данных и пайплайн")
    _table(
        slide,
        Inches(0.5),
        Inches(1.55),
        Inches(12.35),
        Inches(2.4),
        ["Стадия", "Команда", "Входы", "Выходы"],
        [
            [
                "prepare",
                "python -m src.data.make_dataset",
                "raw CSV, код FE",
                "train.csv, test.csv, feature_metadata.json",
            ],
            ["validate", "python -m src.data.validation", "raw CSV, suite", "validation_result.json"],
            ["train", "python -m src.models.train", "processed, params.yaml", "модель, model_card.json, metrics.json"],
        ],
        col_widths=[1.1, 3.1, 2.5, 4.6],
        font_size=12,
    )
    _bullets(
        slide,
        Inches(0.5),
        Inches(4.3),
        Inches(5.9),
        Inches(2.3),
        [
            ("Что версионируется:", "датасет, обработанные данные, модель и метрики — через .dvc-указатели в git."),
            ("Параметры:", "params.yaml (модель, поиск, порог, test_size) — изменение инвалидирует кеш train."),
        ],
        size=13,
        spacing=10,
    )
    _code_panel(
        slide,
        Inches(6.65),
        Inches(4.25),
        Inches(6.2),
        Inches(2.1),
        [
            "$ dvc repro",
            "Running stage 'prepare': ...",
            "Running stage 'validate': ...",
            "Running stage 'train': ...",
            "Updating lock file 'dvc.lock'",
        ],
        font_size=12,
    )
    _footer(slide, "dvc.yaml · dvc.lock · data/raw/UCI_Credit_Card.csv.dvc", 9)
    _notes(
        slide,
        "Три стадии воспроизводятся одной командой; кеш переиспользует результаты при неизменных коде, данных и params.",
    )

    # ------------------------------------------------------------------ #
    # 10. Tests and CI
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "Тесты и CI", "GitHub Actions: 4 джоба на каждый push и pull request")
    _table(
        slide,
        Inches(0.5),
        Inches(1.55),
        Inches(6.7),
        Inches(3.1),
        ["Джоб", "Что делает"],
        [
            ["lint", f"black --check + flake8 ({N_PY_FILES} файлов)"],
            ["test", "pytest + coverage (100 тестов)"],
            ["data-validation", "Great Expectations на сэмпле 1 400 строк"],
            ["docker", "обучение на сэмпле → docker build → /predict"],
        ],
        col_widths=[1.8, 4.9],
        font_size=12,
    )
    _bullets(
        slide,
        Inches(0.5),
        Inches(4.95),
        Inches(6.7),
        Inches(1.7),
        [
            (
                "Покрытие:",
                "FE, ingestion, валидация, 5 алгоритмов, метрики, бандл, API, дрифт, сквозной prepare→train.",
            ),
            ("Локально:", "pytest -q · flake8 src tests · black --check src tests"),
        ],
        size=12,
        spacing=8,
    )
    _table(
        slide,
        Inches(7.45),
        Inches(1.55),
        Inches(5.4),
        Inches(2.9),
        ["Файл тестов", "Проверяет"],
        [
            ["test_build_features", "агрегаты, тренды, биннинг"],
            ["test_validation", "аномалии → провал"],
            ["test_pipeline", "5 алгоритмов, imputer"],
            ["test_api", "эндпоинты, 422/503"],
            ["test_drift", "PSI/KS, алерты"],
            ["test_train", "сквозной пайплайн"],
        ],
        col_widths=[2.1, 2.6],
        font_size=10,
    )
    _textbox(
        slide,
        Inches(7.45),
        Inches(4.75),
        Inches(5.4),
        Inches(1.9),
        "96 быстрых тестов + 4 интеграционных\n"
        "flake8: 0 замечаний\n"
        f"black: {N_PY_FILES} файлов без изменений\n"
        "GE: 104/104 проверки",
        size=13,
        colour=NAVY,
        bold=True,
    )
    _footer(slide, "pytest -q · flake8 · black --check · .github/workflows/ci-cd.yml", 10)
    _notes(
        slide, "CI не требует DVC-хранилища: для валидации и сборки образа используется закоммиченный сэмпл датасета."
    )

    # ------------------------------------------------------------------ #
    # 11. API and Docker
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "REST API и Docker")
    _table(
        slide,
        Inches(0.5),
        Inches(1.55),
        Inches(5.9),
        Inches(2.3),
        ["Метод", "Endpoint", "Назначение"],
        [
            ["GET", "/health", "liveness + статус модели"],
            ["GET", "/model/info", "схема признаков, метрики"],
            ["POST", "/predict", "скоринг клиента"],
            ["POST", "/predict/batch", "пакетный скоринг"],
        ],
        col_widths=[0.9, 2.0, 3.0],
        font_size=11,
    )
    _bullets(
        slide,
        Inches(0.5),
        Inches(4.1),
        Inches(5.9),
        Inches(2.6),
        [
            ("Валидация:", "Pydantic-схемы (возраст, PAY_*, типы), 422 при ошибке."),
            ("Отказоустойчивость:", "нет модели → 503 с инструкцией, а не 500."),
            ("Docker:", "python:3.12-slim, non-root, HEALTHCHECK, только зависимости инференса."),
        ],
        size=12,
        spacing=9,
    )
    _code_panel(
        slide,
        Inches(6.6),
        Inches(1.5),
        Inches(6.25),
        Inches(3.6),
        [
            "POST /predict",
            '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,',
            ' "MARRIAGE":1,"AGE":24,"PAY_0":2,',
            ' "BILL_AMT1":3913,"PAY_AMT1":0}',
            "",
            "200 OK",
            '{"default_prediction": 1,',
            f' "default_probability": {EXAMPLE_PD},',
            ' "risk_band": "critical",',
            f' "threshold": {MODEL_CARD["decision_threshold"]},',
            ' "model_name": "hist_gradient_boosting"}',
        ],
        font_size=11,
    )
    _bullets(
        slide,
        Inches(6.6),
        Inches(5.3),
        Inches(6.25),
        Inches(1.4),
        [
            ("Запуск:", "powershell -File scripts\\docker_build_run.ps1 → build → run → /health → /predict → batch."),
        ],
        size=12,
    )
    _footer(slide, "src/api/app.py · Dockerfile · scripts/docker_build_run.ps1", 11)
    _notes(slide, "Приложение совместимо с примером из задания: обязательны 8 полей, остальные опциональны.")

    # ------------------------------------------------------------------ #
    # 12. Monitoring
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(
        slide, "Мониторинг дрифта (PSI)", "срез тестовой выборки отправляется в живой API и сравнивается с train"
    )
    _table(
        slide,
        Inches(0.5),
        Inches(1.55),
        Inches(6.1),
        Inches(3.9),
        ["Метрика", "Реальные данные", "Инжект дрифта"],
        [
            ["PSI (PD)", f"{DRIFT_STABLE['score_psi']:.4f} stable", f"{DRIFT_ALERT['score_psi']:.4f} critical"],
            [
                "PSI LIMIT_BAL",
                f"{DRIFT_STABLE['feature_psi']['LIMIT_BAL']:.4f}",
                f"{DRIFT_ALERT['feature_psi']['LIMIT_BAL']:.4f}",
            ],
            ["PSI PAY_0", f"{DRIFT_STABLE['feature_psi']['PAY_0']:.4f}", f"{DRIFT_ALERT['feature_psi']['PAY_0']:.4f}"],
            ["KS (PD)", f"{DRIFT_STABLE['score_ks']:.4f}", f"{DRIFT_ALERT['score_ks']:.4f}"],
            ["Средний PD", f"{DRIFT_STABLE['current_mean_pd']:.4f}", f"{DRIFT_ALERT['current_mean_pd']:.4f}"],
            [
                "Live ROC-AUC",
                f"{DRIFT_STABLE['performance']['roc_auc']:.4f}",
                f"{DRIFT_ALERT['performance']['roc_auc']:.4f}",
            ],
        ],
        col_widths=[1.9, 2.1, 2.1],
        font_size=11,
    )
    _add_image(slide, FIGURES / "drift_psi_alert.png", Inches(6.95), Inches(1.45), width=Inches(5.35))
    _textbox(
        slide,
        Inches(6.95),
        Inches(4.25),
        Inches(5.9),
        Inches(0.3),
        f"инжект дрифта: PSI(PD)={DRIFT_ALERT['score_psi']:.2f}, средний PD "
        f"{DRIFT_ALERT['reference_mean_pd']:.2f} → {DRIFT_ALERT['current_mean_pd']:.2f}, "
        f"live AUC падает до {DRIFT_ALERT['performance']['roc_auc']:.2f}",
        size=10,
        colour=GREY,
        italic=True,
    )
    _bullets(
        slide,
        Inches(6.95),
        Inches(4.7),
        Inches(5.9),
        Inches(2.0),
        [
            ("Порог:", "PSI < 0.10 — стабильно, 0.10–0.25 — умеренный сдвиг, > 0.25 — алерт."),
            ("Реакция:", "exit code 2 при алерте → cron / Airflow / система алертов."),
            ("Артефакты:", "drift_report.json + drift_psi.png для истории и разбора."),
        ],
        size=12,
        spacing=7,
    )
    _footer(slide, "python -m src.monitoring.drift --api-url http://127.0.0.1:8000 --sample-size 1000", 12)
    _notes(
        slide,
        "На реальных данных дрифта нет (PSI 0.01). С инжектом ухудшающейся когорты PSI 0.51 и падение live AUC — цепочка алертов работает.",
    )

    # ------------------------------------------------------------------ #
    # 13. Summary
    # ------------------------------------------------------------------ #
    slide = _new_slide(prs)
    _title_bar(slide, "Итоги и соответствие критериям")
    _table(
        slide,
        Inches(0.5),
        Inches(1.5),
        Inches(7.9),
        Inches(5.0),
        ["Критерий (баллы)", "Реализация"],
        [
            ["Git и структура (3)", "структура как в задании, осмысленные коммиты"],
            ["Данные и валидация (7)", "EDA-ноутбук, FE, 104 проверки Great Expectations"],
            ["Модель (8)", "Pipeline + RandomizedSearchCV, ROC-AUC/PR-AUC/KS/F1, ROC-кривая"],
            ["MLflow (8)", "6 экспериментов, артефакты, model registry"],
            ["DVC (5)", "3 стадии, версионирование данных и модели, dvc repro"],
            ["Тесты и CI (7)", "100 тестов, black/flake8, 4 джоба GH Actions"],
            ["Docker и FastAPI (7)", "образ, /predict, smoke-тест, живой сервис"],
            ["Мониторинг (3)", "PSI/KS-скрипт, алерты, отчёты"],
            ["Демонстрация (2)", "эта презентация + MLflow UI"],
        ],
        col_widths=[2.5, 5.4],
        font_size=11,
    )
    _bullets(
        slide,
        Inches(8.75),
        Inches(1.6),
        Inches(4.1),
        Inches(4.9),
        [
            ("Дальнейшие шаги", ""),
            "калибровка PD (CalibratedClassifierCV)",
            "переобучение по триггеру дрифта",
            "DVC remote (S3/GCS) для командной работы",
            "SHAP-интерпретация для кредитного комитета",
            "shadow-деплой и A/B на реальном потоке",
        ],
        size=12,
        spacing=8,
    )
    _footer(slide, "README.md · docs/presentation.pptx · reports/experiments.md", 13)
    _notes(slide, "Готов ответить на вопросы по любому блоку: пайплайн, метрики, тесты, деплой.")

    return prs


def main() -> int:
    prs = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    print(f"presentation saved to {OUTPUT} ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
