import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from api.api_utils import (
    CHAPTER0,
    CHAPTER1,
    CHAPTER2,
    generate_all_chapters,
    generate_chapter,
)


def _make_upload_file(data: bytes, filename: str) -> UploadFile:
    """Оборачивает байты загруженного zip в UploadFile для api_utils."""
    return UploadFile(
        file=io.BytesIO(data),
        size=len(data),
        filename=filename,
        headers=Headers({"content-type": "application/zip"}),
    )


async def _read_response(response) -> tuple[bytes, str]:
    """Читает StreamingResponse и извлекает имя файла из заголовков."""
    body = b"".join([chunk async for chunk in response.body_iterator])
    disposition = response.headers.get("content-disposition", "")
    filename = "result.zip"
    if "filename=" in disposition:
        filename = disposition.rsplit("filename=", 1)[-1].strip('"')
    return body, filename


async def _run_generate(coro):
    """Выполняет корутину генерации и возвращает zip-байты с именем файла."""
    response = await coro
    return await _read_response(response)


def _run(coro):
    """Синхронная обёртка над _run_generate для вызова из Streamlit."""
    return asyncio.run(_run_generate(coro))


def _generate(spec, zip_bytes: bytes, filename: str):
    """Генерирует одну главу по спецификации CHAPTER0/1/2."""
    upload = _make_upload_file(zip_bytes, filename)
    return _run(generate_chapter(spec=spec, project_parts_zip=upload))


def _generate_all(zip_bytes: bytes, filename: str):
    """Генерирует все главы (0, 1, 2) одним вызовом."""
    upload = _make_upload_file(zip_bytes, filename)
    return _run(generate_all_chapters(project_parts_zip=upload))


def _require_zip() -> tuple[bytes, str] | None:
    """Проверяет наличие zip в session_state; иначе показывает ошибку."""
    uploaded = st.session_state.get("project_parts_zip")
    if uploaded is None:
        st.error("Загрузите zip-архив с PDF смежных разделов")
        return None
    return uploaded.getvalue(), uploaded.name


def _set_result(result_bytes: bytes, result_filename: str):
    """Сохраняет результат генерации в session_state для скачивания."""
    st.session_state.result_bytes = result_bytes
    st.session_state.result_filename = result_filename


def _run_action(label: str, action):
    """Запускает действие генерации и отображает статус выполнения."""
    zip_data = _require_zip()
    if zip_data is None:
        return

    zip_bytes, zip_name = zip_data
    status = st.empty()
    status.info(f"{label}: выполняется...")

    try:
        result_bytes, result_filename = action(zip_bytes, zip_name)
    except HTTPException as exc:
        status.error(str(exc.detail))
        return
    except Exception as exc:
        status.error(f"Ошибка: {exc}")
        return

    _set_result(result_bytes, result_filename)
    status.success(f"{label}: готово")


def _params_table(editor_key: str):
    """Таблица 2×2: первый столбец (a, b) только для чтения."""
    st.subheader("Табличные данные")
    df = pd.DataFrame({"ключ": ["a", "b"], "значение": ["", ""]})
    st.data_editor(
        df,
        column_config={
            "ключ": st.column_config.TextColumn("ключ", disabled=True),
            "значение": st.column_config.TextColumn("значение"),
        },
        disabled=["ключ"],
        hide_index=True,
        use_container_width=True,
        key=editor_key,
    )
    

st.set_page_config(page_title="NC_ecology", layout="centered")
st.title("NC_ecology")

if "result_bytes" not in st.session_state:
    st.session_state.result_bytes = None
    st.session_state.result_filename = None

st.file_uploader(
    "Zip-архив с PDF смежных разделов",
    type=["zip"],
    key="project_parts_zip",
)

tab0, tab1, tab2, tab_all = st.tabs(["Глава 0", "Глава 1", "Глава 2", "Все главы"])

with tab0:
    st.caption("Аннотация и введение")
    _params_table("table_ch0")
    if st.button("Запуск", key="run_ch0"):
        _run_action("Глава 0", lambda b, n: _generate(CHAPTER0, b, n))

with tab1:
    st.caption("ОБЩИЕ СВЕДЕНИЯ ОБ ОБЪЕКТЕ ПРОЕКТИРОВАНИЯ")
    _params_table("table_ch1")
    if st.button("Запуск", key="run_ch1"):
        _run_action("Глава 1", lambda b, n: _generate(CHAPTER1, b, n))

with tab2:
    st.caption("ВОЗДЕЙСТВИЕ ОБЪЕКТА НА ЗЕМЕЛЬНЫЕ РЕСУРСЫ")
    _params_table("table_ch2")
    if st.button("Запуск", key="run_ch2"):
        _run_action("Глава 2", lambda b, n: _generate(CHAPTER2, b, n))

with tab_all:
    st.caption("Генерация глав 0, 1 и 2 одним запросом")
    _params_table("table_all")
    if st.button("Запуск", key="run_all"):
        _run_action("Все главы", _generate_all)

if st.session_state.result_bytes:
    st.download_button(
        "Скачать результат",
        data=st.session_state.result_bytes,
        file_name=st.session_state.result_filename,
        mime="application/zip",
    )
