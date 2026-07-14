import json
import datetime as _dt
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docxtpl import DocxTemplate
from jinja2 import Environment, ChainableUndefined

from config.config_file import cfg
from src.utils.logger import logger


# ----------------------------------------------------------------------------
# Подсветка плейсхолдеров
# ----------------------------------------------------------------------------
# Стратегия: docxtpl при обычном синтаксисе {{ }} заменяет плейсхолдер на
# готовый текст, после чего определить "что было плейсхолдером" невозможно.
# Поэтому каждое подставляемое значение оборачивается в невидимые маркеры
# (символы-разделители Unicode), а после рендера выполняется пост-обработка
# через python-docx: маркеры удаляются, а сам текст подсвечивается:
#   - СВЕТЛО-СЕРЫМ — успешно заполненные значения;
#   - ЖЁЛТЫМ       — незаполненные ({{undefined ...}}), требующие ручной доработки.
#
# Пост-обработка проходит по телу документа, всем таблицам (включая штамп)
# и колонтитулам (header/footer) во всех секциях.

# Невидимые маркеры (Unit/Record separators) — крайне маловероятны в тексте.
MARK_FILLED = "\u241f"   # ␟  — обрамляет заполненные значения
MARK_MISSING = "\u241e"  # ␞  — обрамляет незаполненные значения

FILLED_COLOR = WD_COLOR_INDEX.GRAY_25   # светло-серая подсветка заполненных значений
MISSING_COLOR = WD_COLOR_INDEX.YELLOW   # жёлтая подсветка незаполненных

# Поля, которые НЕ подсвечиваем (большие нормативные списки и т.п.)
_NO_HIGHLIGHT_KEYS: set[str] = {
    "СПИСОК_НОРМАТИВНОЙ_БАЗЫ",
}


# ----------------------------------------------------------------------------
# Undefined-класс: отслеживает незаполненные плейсхолдеры + маркирует их
# ----------------------------------------------------------------------------
class PlaceholderUndefined(ChainableUndefined):
    """Плейсхолдер, отсутствующий в data, но присутствующий в template.

    Маркирует свой вывод символом MARK_MISSING для последующей жёлтой
    подсветки и регистрирует имя поля в общем реестре.
    """

    _registry: set[str] | None = None

    def __init__(self, *args, path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path or self._undefined_name
        reg = type(self)._registry
        if reg is not None and self._path:
            reg.add(str(self._path))

    def __getattr__(self, name):
        return type(self)(name=name, path=f"{self._path}.{name}")

    def __str__(self):
        return f"{MARK_MISSING}{{{{undefined {self._path}}}}}{MARK_MISSING}"


# ----------------------------------------------------------------------------
# Автоподстановка дат
# ----------------------------------------------------------------------------
_RU_MONTHS = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}
_RU_MONTHS_NOM = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}


def build_date_placeholders(now: _dt.datetime | None = None) -> dict[str, str]:
    """Формирует словарь авто-дат для подстановки в шаблоны."""
    now = now or _dt.datetime.now()
    return {
        "ТЕКУЩАЯ_ДАТА": now.strftime("%d.%m.%Y"),
        "ТЕКУЩИЙ_ГОД": now.strftime("%Y"),
        "ТЕКУЩИЙ_МЕСЯЦ": _RU_MONTHS_NOM[now.month],
        "ТЕКУЩИЙ_МЕСЯЦ_ГОД": f"{_RU_MONTHS_NOM[now.month]} {now.year}",
        "ТЕКУЩАЯ_ДАТА_ПРОПИСЬЮ": f"«{now.day:02d}» {_RU_MONTHS[now.month]} {now.year} г.",
        "date": now.strftime("%d.%m.%Y"),
        "year": now.strftime("%Y"),
    }


# ----------------------------------------------------------------------------
# Подготовка контекста
# ----------------------------------------------------------------------------
def _is_meaningful(value) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    if s.startswith("{{") and "undefined" in s:
        return False
    return True


def prepare_context(
    data: dict,
    *,
    highlight: bool = True,
    add_dates: bool = True,
) -> dict:
    """Готовит контекст: авто-даты + обёртка значений в маркеры подсветки."""
    ctx: dict = dict(data)

    # Авто-даты (только для отсутствующих/пустых ключей)
    if add_dates:
        for k, v in build_date_placeholders().items():
            if k not in ctx or not _is_meaningful(ctx.get(k)):
                ctx[k] = v

    if not highlight:
        return ctx

    wrapped: dict = {}
    for key, value in ctx.items():
        if isinstance(value, (dict, list)):
            wrapped[key] = value
            continue
        if not _is_meaningful(value):
            # пустое значение оставляем как есть; если ключ есть, но пуст —
            # пометим как missing-маркер для жёлтой подсветки
            if value in (None, ""):
                wrapped[key] = value
            else:
                wrapped[key] = f"{MARK_MISSING}{value}{MARK_MISSING}"
            continue
        if key in _NO_HIGHLIGHT_KEYS:
            wrapped[key] = value
            continue
        wrapped[key] = f"{MARK_FILLED}{value}{MARK_FILLED}"

    return wrapped


# ----------------------------------------------------------------------------
# Пост-обработка: подсветка по маркерам
# ----------------------------------------------------------------------------
def _highlight_paragraph(paragraph) -> None:
    """Подсвечивает текст между маркерами внутри одного абзаца.

    docxtpl, как правило, помещает значение плейсхолдера в один run, но во
    избежание потери маркеров при их разбиении по нескольким run-ам сначала
    выполняется "слияние" текста абзаца в первый run.
    """
    full_text = "".join(run.text for run in paragraph.runs)
    if MARK_FILLED not in full_text and MARK_MISSING not in full_text:
        return

    # Сегментируем текст на части: (текст, цвет|None)
    segments = _segment(full_text)

    # Удаляем существующие runs и пересоздаём с нужной подсветкой,
    # сохраняя базовое форматирование первого run (шрифт/размер/жирность).
    if not paragraph.runs:
        return
    template_run = paragraph.runs[0]
    font = template_run.font

    # Очищаем все runs
    for run in list(paragraph.runs):
        run._element.getparent().remove(run._element)

    for text, color in segments:
        if text == "":
            continue
        new_run = paragraph.add_run(text)
        # копируем базовые свойства шрифта
        try:
            new_run.bold = template_run.bold
            new_run.italic = template_run.italic
            new_run.underline = template_run.underline
            if font.size:
                new_run.font.size = font.size
            if font.name:
                new_run.font.name = font.name
        except Exception:
            pass
        if color is not None:
            new_run.font.highlight_color = color


def _segment(text: str):
    """Разбивает текст по маркерам на сегменты (часть, цвет)."""
    segments = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == MARK_FILLED or ch == MARK_MISSING:
            color = FILLED_COLOR if ch == MARK_FILLED else MISSING_COLOR
            j = text.find(ch, i + 1)
            if j == -1:
                # незакрытый маркер — берём до конца, цвет применяем
                segments.append((text[i + 1:], color))
                break
            segments.append((text[i + 1:j], color))
            i = j + 1
        else:
            # обычный текст до следующего маркера
            nf = text.find(MARK_FILLED, i)
            nm = text.find(MARK_MISSING, i)
            candidates = [x for x in (nf, nm) if x != -1]
            nxt = min(candidates) if candidates else n
            segments.append((text[i:nxt], None))
            i = nxt
    return segments


def _iter_all_paragraphs(doc: Document):
    """Итерирует абзацы тела, таблиц и колонтитулов всех секций."""
    def _tables(tables):
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    yield from cell.paragraphs
                    yield from _tables(cell.tables)

    # Тело
    yield from doc.paragraphs
    yield from _tables(doc.tables)

    # Колонтитулы всех секций
    for section in doc.sections:
        for hf in (section.header, section.footer,
                   section.first_page_header, section.first_page_footer,
                   section.even_page_header, section.even_page_footer):
            if hf is None:
                continue
            yield from hf.paragraphs
            yield from _tables(hf.tables)


def apply_highlighting(docx_path: Path) -> None:
    """Открывает готовый DOCX и подсвечивает все маркированные значения."""
    doc = Document(str(docx_path))
    for paragraph in _iter_all_paragraphs(doc):
        _highlight_paragraph(paragraph)
    doc.save(str(docx_path))


# ----------------------------------------------------------------------------
# Основная функция заполнения шаблона
# ----------------------------------------------------------------------------
def fill_docx_template(
    template_path: Path,
    data: Path | dict,
    output_docx_path: Path,
    *,
    highlight: bool = True,
    add_dates: bool = True,
) -> list[str]:
    """Заполнение шаблона docx из json/dict и сохранение копии.

    :param template_path: путь к шаблону .docx
    :param data: путь к json-файлу или словарь с данными
    :param output_docx_path: путь для сохранения результата
    :param highlight: подсвечивать заполненные плейсхолдеры зелёным,
        незаполненные — жёлтым
    :param add_dates: добавлять авто-даты (ТЕКУЩАЯ_ДАТА/ТЕКУЩИЙ_ГОД и т.д.)
    :return: список незаполненных (не найденных в данных) плейсхолдеров
    """
    if not isinstance(data, dict):
        try:
            with open(data, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Путь к json файлу не найден: {data}")

    context = prepare_context(data, highlight=highlight, add_dates=add_dates)

    PlaceholderUndefined._registry = set()
    try:
        doc = DocxTemplate(template_path)
        doc.render(context, jinja_env=Environment(undefined=PlaceholderUndefined))
        doc.save(output_docx_path)
        undefined_in_template = sorted(PlaceholderUndefined._registry)
    finally:
        PlaceholderUndefined._registry = None

    if highlight:
        try:
            apply_highlighting(Path(output_docx_path))
        except Exception:
            logger.exception(
                f"Не удалось применить подсветку для {output_docx_path}"
            )

    if undefined_in_template:
        logger.warning(
            f"[{Path(output_docx_path).name}] Не заполнены плейсхолдеры шаблона "
            f"({len(undefined_in_template)}): {undefined_in_template}"
        )
    else:
        logger.info(
            f"[{Path(output_docx_path).name}] Все плейсхолдеры шаблона заполнены."
        )

    logger.debug(f"Шаблон заполнен. Копия сохранена в {output_docx_path}")
    return undefined_in_template


if __name__ == "__main__":
    template = Path(
        r"C:\Users\maxfi\PycharmProjects\NC_ecology\src\ecology_chapters\chapter1\template.docx"
    )
    fill_docx_template(
        template_path=template,
        data={},
        output_docx_path=template.parent / (template.name + "_out" + template.suffix),
    )
