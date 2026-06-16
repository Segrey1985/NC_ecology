import re
import json
import glob
import zipfile
import pypdf
import pymupdf
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, Type

from pydantic import BaseModel, ConfigDict, create_model
from pydantic.alias_generators import to_pascal
from pdfminer.layout import LTTextBox, LTTextLine, LTTextContainer
from pdfminer.high_level import extract_pages as extract_pages_miner
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)

from src.utils.logger import logger

# ___ ALL UTILS ___

UUID4_HEX_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def is_valid_uuid4_hex(value: str) -> bool:
    return bool(UUID4_HEX_PATTERN.fullmatch(value))


def extract_project_parts_pdfs(
    project_parts_zip: bytes | Path, project_parts_dir: Path
) -> None:
    """Рекурсивный сбор pdf файлов и копирование их в project_parts_dir"""
    
    if isinstance(project_parts_zip, Path):
        project_parts_zip_bytes = project_parts_zip.read_bytes()
    else:
        project_parts_zip_bytes = project_parts_zip

    project_parts_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(BytesIO(project_parts_zip_bytes)) as zf:
        zf.extractall(project_parts_dir)

    for path in project_parts_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".pdf":
            path.unlink()

    pdfs = sorted(project_parts_dir.rglob("*.pdf"))
    if not pdfs:
        raise ValueError("В project_parts_zip не найдено ни одного PDF")

    for idx, pdf_path in enumerate(pdfs, start=1):
        if pdf_path.parent == project_parts_dir:
            continue
        dest = project_parts_dir / pdf_path.name
        if dest.exists():
            dest = project_parts_dir / f"{pdf_path.stem}_{idx:04d}.{pdf_path.suffix}"
        pdf_path.rename(dest)

    for path in sorted(project_parts_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            path.rmdir()


def format_rag_context(chunks: list[str]) -> str:
    if not chunks:
        return "Релевантный контекст не найден."
    return "\n\n".join(f"[{idx}] {chunk}" for idx, chunk in enumerate(chunks, start=1))


def pascal_to_snake(name: str) -> str:
    """Facility → facility, NearestObjects → nearest_objects."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_part_names_for_model(model: Type[BaseModel] | None) -> list[str] | None:
    """
    Возвращает список `part_name` для фильтрации поиска в Qdrant по конкретной модели.
    """
    if model is None:
        return None
    if part_names := model.__private_attributes__.get("_part_name", None):
        return part_names.default
    return None


# ___ PDF ___


def count_pages(file_path: str) -> int | None:
    try:
        with open(file_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            return len(reader.pages)
    except Exception:
        return None


def extract_pages(
    input_pdf: str | bytes,
    pages_to_keep: list[int],
    output_pdf_path: Optional[str] = None,
) -> Optional[bytes]:
    """Извлечение страниц из PDF. Если output_pdf_path не задан, возвращает байты.
    
    :param input_pdf: Путь к pdf файлу или байты.
    :param pages_to_keep: Оставляемые страницы (нумерация с 0).
    :param output_pdf_path: Путь для сохранения обрезанного pdf (опционально).
    """

    # Проверяем, что было передано: путь к файлу или байты
    if isinstance(input_pdf, bytes):
        input_pdf_file = BytesIO(input_pdf)
    else:
        input_pdf_file = open(input_pdf, "rb")

    try:
        reader = pypdf.PdfReader(input_pdf_file)
        writer = pypdf.PdfWriter()
        valid_pages = [x + 1 for x in range(len(reader.pages))]
        valid_pages_to_keep = [x for x in pages_to_keep if x in valid_pages]

        # Извлекаем указанные страницы
        for page_num in valid_pages_to_keep:
            # Нумерация страниц в pypdf начинается с 0
            writer.add_page(reader.pages[page_num])

        if output_pdf_path:
            # Записываем результат в новый PDF файл
            with open(output_pdf_path, "wb") as output_pdf_file:
                writer.write(output_pdf_file)
        else:
            # Создаем байтовый буфер для хранения результата
            output_buffer = BytesIO()
            writer.write(output_buffer)

            # Возвращаем байты PDF-файла
            return output_buffer.getvalue()

    finally:
        # Закрываем файл, если он был открыт
        if not isinstance(input_pdf, bytes):
            input_pdf_file.close()


def extract_pages_as_list_with_miner(pdf: str | Path | bytes) -> list[str]:
    """Извлекает текст из PDF постранично."""

    if isinstance(pdf, bytes):
        pdf_source = BytesIO(pdf)
    else:
        pdf_source = pdf

    pages: list[str] = []

    for page_layout in extract_pages_miner(pdf_source):
        page_text_parts: list[str] = []

        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text_parts.append(element.get_text())

        pages.append("".join(page_text_parts).strip())

    return pages


def extract_text_with_miner_coords(
    pdf_path,
    y_tolerance: float = 5,
    page_numbers: list[int] | None = None,
    ignore: tuple[float, float, float, float] | None = (
        0.01,
        0.01,
        0.07,
        0.07,
    ),  # (top, right, bottom, left)
) -> list[str]:
    """
    Извлекает текст из PDF с координатами строк + фильтрация областей страницы.

    :param pdf_path: Путь к PDF
    :param y_tolerance: Допуск по Y для группировки строк
    :param page_numbers: Какие страницы обрабатывать (0-based)
    :param ignore: (top, right, bottom, left) — доли страницы (0..1), которые нужно игнорировать
    :return: Список текстов по страницам
    """

    text_with_coords = []

    pages_iter = (
        extract_pages_miner(pdf_path)
        if page_numbers is None
        else extract_pages_miner(pdf_path, page_numbers=page_numbers)
    )

    for page_layout in pages_iter:
        lines = []

        # размеры страницы
        page_width = page_layout.width
        page_height = page_layout.height

        # границы игнора
        if ignore:
            top, right, bottom, left = ignore

            top_limit = page_height * (1 - top)  # выше этой линии — игнор
            bottom_limit = page_height * bottom  # ниже — игнор
            left_limit = page_width * left
            right_limit = page_width * (1 - right)
        else:
            top_limit = page_height
            bottom_limit = 0
            left_limit = 0
            right_limit = page_width

        for element in page_layout:
            if isinstance(element, LTTextBox):
                # Обрабатываем текстовые блоки
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        # Получаем координаты и текст строки
                        x0, y0, x1, y1 = text_line.bbox
                        text = text_line.get_text().strip()

                        if not text:
                            continue

                        # --- ФИЛЬТР ПО ГРАНИЦАМ ---
                        if (
                            y1 > top_limit  # слишком высоко (top)
                            or y0 < bottom_limit  # слишком низко (bottom)
                            or x0 < left_limit  # слишком слева (left)
                            or x1 > right_limit  # слишком справа (right)
                        ):
                            continue

                        lines.append((text, (x0, y0, x1, y1)))

        # сортировка сверху вниз
        lines.sort(key=lambda x: -x[1][1])

        # Группируем строки по y-координате с учетом запаса смещения
        grouped_lines = []
        current_group = []
        prev_y = None

        for line in lines:
            text, (x0, y0, x1, y1) = line

            if prev_y is None or abs(y1 - prev_y) <= y_tolerance:
                current_group.append(line)
            else:
                # Сортируем текущую группу по x0 (слева направо)
                current_group.sort(key=lambda x: x[1][0])
                grouped_lines.append(current_group)
                current_group = [line]
            prev_y = y1

        # Добавляем последнюю группу
        if current_group:
            current_group.sort(key=lambda x: x[1][0])
            grouped_lines.append(current_group)

        # сборка текста
        page_text = ""
        for group in grouped_lines:
            group_text = " ".join([text for text, _ in group])
            page_text += group_text + "\n"

        text_with_coords.append(page_text)

    return text_with_coords


def extract_pages_as_list_with_pymupdf(pdf: Path | str | bytes) -> list[str]:
    if isinstance(pdf, bytes):
        doc = pymupdf.open(stream=pdf, filetype="pdf")
    elif isinstance(pdf, (Path, str)):
        doc = pymupdf.open(pdf)
    else:
        raise TypeError("Неверный тип аргумента `pdf`. Требуется `Path | str | bytes`")
    return [page.get_text() for page in doc]


def find_page_index_by_first_text(input_pdf: str | bytes, text: str) -> int | None:
    """
    Возвращает индекс страницы (нумерация с 0), где первой текстовой (буквенной) информацией является `text`.
    """

    if not text:
        return None

    if isinstance(input_pdf, bytes):
        input_pdf_file = BytesIO(input_pdf)
    else:
        input_pdf_file = open(input_pdf, "rb")

    try:
        reader = pypdf.PdfReader(input_pdf_file)

        for page_idx, page in enumerate(reader.pages):
            page_content: list[str] = extract_text_with_miner_coords(
                input_pdf_file,
                page_numbers=[page_idx],
            )
            page_text = page_content[0]
            # Нормализуем переносы/табуляции, чтобы startswith работал предсказуемо.
            page_text = " ".join(page_text.replace("\t", " ").split())

            # Ищем первый буквенный символ (Unicode: кириллица/латиница и т.п.).
            first_alpha_pos: int | None = None
            for i, ch in enumerate(page_text):
                if ch.isalpha():
                    first_alpha_pos = i
                    break

            if first_alpha_pos is None:
                continue

            if page_text[first_alpha_pos:].startswith(text.strip()):
                return page_idx

        return None
    finally:
        if not isinstance(input_pdf, bytes):
            input_pdf_file.close()


def find_pages_index_by_text(
    pdf: Path | str | bytes, text: str, max_len: int | None = None
) -> list[int]:
    """Возвращает индексы страниц, содержащих `text`"""

    if not text:
        return []

    if isinstance(pdf, bytes):
        doc = pymupdf.open(stream=pdf, filetype="pdf")
    elif isinstance(pdf, (Path, str)):
        doc = pymupdf.open(pdf)
    else:
        raise TypeError("Неверный тип аргумента `pdf`. Требуется `Path | str | bytes`")

    result_pages: list[int] = []

    for page_idx, page in enumerate(doc):
        page_text = page.get_text().strip()
        page_text = " ".join(page_text.replace("\t", " ").split())

        if text.strip() in page_text:
            result_pages.append(page_idx)

        if max_len and len(result_pages) == max_len:
            return result_pages

    return result_pages


# ___ LangGraph ___


def _safe_print(text: str) -> None:
    """
    Безопасная печать для Windows-консолей с ограниченной кодировкой (например, cp1251).
    Не даёт падать на UnicodeEncodeError: заменяет непечатаемые символы.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        import sys

        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe)


def print_chunk(chunk):
    for node_name, data in chunk.items():

        _safe_print(f"\n\n{'=' * 30} {node_name} {'=' * 30}")
        _safe_print(f"Full response:\n{data}")

        if not data:
            continue

        # state['answer'], state['something'] ...

        ignore_keys = ["rag_context"]
        fields_without_messages = {k: v for k, v in data.items() if k != "messages"}
        for key, value in fields_without_messages.items():
            if key not in ignore_keys:
                _safe_print(f"{key}:\n{value if value else 'None'}")
            else:
                _safe_print(f"{key}: ...")

        # state['messages']

        messages = data.get("messages", [])

        if len(messages) > 1:
            logger.warning("!!! Количество messages, которые вернула node больше 1 !!!")

        for message in messages:

            if isinstance(message, AIMessage) and message.tool_calls:

                for tool_call in message.tool_calls:
                    _safe_print(f"tool_call name: {tool_call['name']}")
                    if tool_call["args"]:
                        _safe_print("tool_calls ARGS:")
                        for i, (k, v) in enumerate(tool_call["args"].items()):
                            _safe_print(f"ARG[{i+1}] = {k}: {v}")
                    else:
                        _safe_print("tool_calls ARGS: None")

            if node_name == "tools":
                _safe_print(f"tool name:\n{message.name}")

            _safe_print(f"content:\n{message.content if message.content else 'None'}")


def build_input_query(model: type[BaseModel]) -> str:
    """Генерация input_query на основе pydantic-класса."""
    doc = (inspect.getdoc(model) or "").strip()
    fields = getattr(model, "model_fields", {}) or {}

    field_lines: list[str] = []
    for f_name, f_info in fields.items():
        desc = getattr(f_info, "description", None)
        if desc:
            field_lines.append(f"- {f_name}: {desc}")
        else:
            field_lines.append(f"- {f_name}")

    return (
        f"Заполни модель `{model.__name__}`.\n"
        f"Описание: {doc or '—'}\n"
        "Поля:\n" + "\n".join(field_lines)
    ).strip()


# ___ inspect & importlib ___


import inspect
import importlib


def iter_models_from_module(module_path: str) -> list[type[BaseModel]]:
    """
    Берём только pydantic-модели, объявленные именно в модуле `module_path`,
    чтобы не тащить импортированные классы (например, из `inner`).
    """
    module = importlib.import_module(module_path)
    out: list[type[BaseModel]] = []
    for _name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, BaseModel):
            continue
        if obj is BaseModel:
            continue
        if getattr(obj, "__module__", "") != module.__name__:
            continue
        out.append(obj)

    out.sort(key=lambda cls: cls.__name__)
    return out


def pick_assembly_model(assembly_module_path: str) -> type[BaseModel]:
    """
    Достаём единственную корневую модель сборки из `<chapter>.assembly`.
    Ищем BaseModel-классы, экспортированные в модуле (в т.ч. собранные через create_model).
    """
    module = importlib.import_module(assembly_module_path)

    candidates: list[type[BaseModel]] = []
    for _name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, BaseModel):
            continue
        if obj is BaseModel:
            continue
        candidates.append(obj)

    if len(candidates) != 1:
        raise RuntimeError(
            f"В модуле `{assembly_module_path}` ожидается ровно 1 BaseModel-класс, "
            f"найдено: {len(candidates)}."
        )

    return candidates[0]


def build_chapter_assembly_model(
    models_module_path: str,
    *,
    model_name: str = "ChapterData",
) -> type[BaseModel]:
    """
    Собирает корневую pydantic-модель главы из всех моделей в `*.models`.
    Поля опциональны — можно валидировать и рендерить docx по частичному JSON.
    """
    field_defs: dict[str, tuple[Any, None]] = {}
    for model_cls in iter_models_from_module(models_module_path):
        field_defs[pascal_to_snake(model_cls.__name__)] = (model_cls | None, None)

    return create_model(
        model_name,
        __config__=ConfigDict(
            alias_generator=to_pascal,
            populate_by_name=True,
        ),
        **field_defs,
    )


def iter_chapter_models(chapter_module_path: str) -> list[type[BaseModel]]:
    """Модели главы с фильтром из `_debug_models.py` (только test_mode='filter' в main2)."""

    def filter_models_by_names(
        models: list[type[BaseModel]],
        names: list[str] | None,
    ) -> list[type[BaseModel]]:
        """None — без фильтра, иначе только модели с именами из `names`."""
        if names is None:
            return models
        allowed = set(names)
        return [m for m in models if m.__name__ in allowed]

    def get_debug_model_names(chapter_module_path: str) -> list[str] | None:
        """Читает ACTIVE_MODEL_NAMES из `<chapter>._debug_models` (для test_mode='filter')."""
        try:
            module = importlib.import_module(f"{chapter_module_path}._debug_models")
        except ModuleNotFoundError:
            return None
        return getattr(module, "ACTIVE_MODEL_NAMES", None)

    return filter_models_by_names(
        iter_models_from_module(f"{chapter_module_path}.models"),
        get_debug_model_names(chapter_module_path),
    )


# ___ filter_mode ___


def _list_item_type(annotation: Any) -> Any | None:
    if get_origin(annotation) is list:
        args = get_args(annotation)
        return args[0] if args else Any
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        item_type = _list_item_type(arg)
        if item_type is not None:
            return item_type
    return None


def _docx_placeholder_value(
    section_name: str, field_name: str, annotation: Any
) -> object:
    placeholder = "{{ " + f"{section_name}.{field_name}" + " }}"
    item_type = _list_item_type(annotation)
    if item_type is None:
        return placeholder
    if inspect.isclass(item_type) and issubclass(item_type, BaseModel):
        return [
            {
                subfield_name: "{{ "
                + f"{section_name}.{field_name}.{subfield_name}"
                + " }}"
                for subfield_name in item_type.model_fields
            }
        ]
    return [placeholder]


def _nested_model_type(annotation: Any) -> type[BaseModel] | None:
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        if inspect.isclass(arg) and issubclass(arg, BaseModel):
            return arg
    return None


def filter_mode_payload_and_validate(
    assembly_model: type[BaseModel],
    results: dict[str, object],
) -> BaseModel:
    """Подставляет в assembly только те блоки, что есть в `results` (ключи — имена классов)."""
    payload: dict[str, object] = {}
    for field_name, field_info in assembly_model.model_fields.items():
        nested = _nested_model_type(field_info.annotation)
        if nested is None:
            continue
        value = results.get(nested.__name__)
        if value is not None:
            payload[field_name] = value
    return assembly_model.model_validate(payload)


def filter_mode_assembly_to_docx_context(
    assembly_model: type[BaseModel],
    data: BaseModel,
    *,
    preserve_unfilled: bool = False,
) -> dict[str, object]:
    """
    Контекст для docxtpl: snake_case-ключи как в шаблоне.
    Незаполненные секции — пустой dict, чтобы Jinja не падал на partial-прогоне.
    В filter_mode можно сохранить плейсхолдеры как строки `{{ section.field }}`.
    """
    dumped = data.model_dump(mode="json")
    ctx: dict[str, object] = {}
    for field_name, field_info in assembly_model.model_fields.items():
        value = dumped.get(field_name)
        if not preserve_unfilled:
            ctx[field_name] = value if value is not None else {}
            continue

        nested = _nested_model_type(field_info.annotation)
        if nested is None:
            ctx[field_name] = value
            continue

        placeholders = {
            subfield_name: _docx_placeholder_value(
                field_name,
                subfield_name,
                subfield_info.annotation,
            )
            for subfield_name, subfield_info in nested.model_fields.items()
        }
        if isinstance(value, dict):
            ctx[field_name] = {
                key: placeholders[key] if item is None or item == [] else item
                for key, item in {**placeholders, **value}.items()
            }
        else:
            ctx[field_name] = placeholders
    return ctx


# ____________ main ____________


if __name__ == "__main__":

    # найти номера страниц файла, где заголовком будет text1, text2
    
    text1 = "3. ВОЗДЕЙСТВИЕ ОБЪЕКТА ПРОЕКТИРОВАНИЯ НА АТМОСФЕРНЫЙ ВОЗДУХ"
    text2 = "4. ВОЗДЕЙСТВИЕ ОБЪЕКТА ПРОЕКТИРОВАНИЯ НА СОСТОЯНИЕ ПОВЕРХНОСТНЫХ И ПОДЗЕМНЫХ ВОД"
    
    results = []
    for f in glob.glob(r"C:\Users\maxfi\Desktop\ПМООС\ПМООСы\*.pdf"):
        # print(f)
        print(page1 := find_pages_index_by_text(f, text=text1, max_len=2))
        print(page2 := find_pages_index_by_text(f, text=text2, max_len=2))
        print("-------------\n")
        results.append((f, page1[1], page2[1]))
        
    print(results)
    
    
    def cut_between_and_save(file_path: str | Path, start, end) -> None:
        """ Обрезать с X по Y и сохранить в out.pdf """
        bytes_ = extract_pages(file_path, pages_to_keep=list(range(start, end)))
        pth = Path(file_path) if isinstance(file_path, str) else file_path
        new_name = pth.parent / f"cut_between_and_save_OUT" /pth.name
        new_name.parent.mkdir(parents=True, exist_ok=True)
        with open(new_name, "wb") as new_file:
            new_file.write(bytes_)


    for f, start, end in results:
        cut_between_and_save(f, start, end)

    # найти номер страницы файла, где заголовком будет text

    # text = "ОБЩИЕ СВЕДЕНИЯ ОБ ОБЪЕКТЕ ПРОЕКТИРОВАНИЯ"
    # for f in glob.glob(r"C:\Users\maxfi\Desktop\ПМООС\ПМООСы\*.pdf"):
    #     print(page := find_page_index_by_first_text(f, text=text))
    #
    #     # здесь уже обрезаем и сохраняем (можно закомментить)
    #
    #     if page is not None:
    #         bytes_ = extract_pages(f, pages_to_keep=list(range(page+1, page+1+8)))
    #         pth = pathlib.Path(f)
    #         new_name = pth.parent / "chapter1" /pth.name
    #         new_name.parent.mkdir(parents=True, exist_ok=True)
    #         with open(new_name, "wb") as new_file:
    #             new_file.write(bytes_)
    #     else:
    #         print(f)

    # обрезать исходные pdf
    #
    # for f in glob.glob(r"../../data/IN/project1/*.pdf"):
    #     bytes_ = extract_pages(f, pages_to_keep=list(range(20)))
    #     pth = pathlib.Path(f)
    #     new_name = pth.parent / "trim" /pth.name
    #     with open(new_name, "wb") as new_file:
    #         new_file.write(bytes_)

    # test extract_text_with_miner_coords()

    # text = extract_text_with_miner_coords(
    #     r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\1_ОК.17.24СТ-ПЗ.pdf",
    #     ignore=(0.01, 0.01, 0.07, 0.07),
    # )
    # print("\n".join(text))
    #
    # pass
