import os
import httpx
import pandas as pd
import streamlit as st

# переменная окружения передается в docker-compose.yml (API_BASE_URL: "http://app:8000")
DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _check_api_health() -> bool:
    """Проверяет доступность API через эндпоинт /health."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{DEFAULT_API_URL}/health")
            return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception:
        return False


def _run_action(label: str, endpoint: str):
    """
    Запускает обработку главы: валидирует zip, вызывает API и сохраняет результат в session_state.
    :param label: Текстовое описание эндпоинта для отображения на странице (например "Глава 1")
    :param endpoint: Эндпоинт (например "/chapter1")
    """
    uploaded = st.session_state.get("project_parts_zip")
    if uploaded is None:
        st.error("Загрузите zip-архив с PDF смежных разделов")
        return

    if not _check_api_health():
        st.error(f"API недоступен по адресу {DEFAULT_API_URL}. Запустите FastAPI: python -m api.api_prod")
        return

    zip_bytes, zip_name = uploaded.getvalue(), uploaded.name
    status = st.empty()
    status.info(f"{label}: выполняется...")

    try:
        files = {"project_parts_zip": (zip_name, zip_bytes, "application/zip")}
        with httpx.Client(base_url=DEFAULT_API_URL, timeout=httpx.Timeout(None)) as client:
            response = client.post(endpoint, files=files)
            response.raise_for_status()
            disposition = response.headers.get("content-disposition", "")
            result_filename = "result.zip"
            if "filename=" in disposition:
                result_filename = disposition.rsplit("filename=", 1)[-1].strip('"')
            result_bytes = response.content
            
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail")
            if isinstance(detail, list):
                error_msg = "; ".join(str(item) for item in detail)
            elif detail is not None:
                error_msg = str(detail)
            else:
                error_msg = exc.response.text or str(exc)
        except Exception:
            error_msg = exc.response.text or str(exc)
        status.error(error_msg)
        return
    
    except httpx.RequestError as exc:
        status.error(f"Не удалось подключиться к API: {exc}")
        return
    
    except Exception as exc:
        status.error(f"Ошибка: {exc}")
        return

    st.session_state.result_bytes = result_bytes
    st.session_state.result_filename = result_filename
    status.success(f"{label}: готово")


# ______________________________ UI ______________________________

def _params_table(editor_key: str):
    """Отображает редактируемую таблицу параметров (ключ / значение) для главы."""
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
        width="stretch",
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
        _run_action(label="Глава 0", endpoint="/chapter0")

with tab1:
    st.caption("ОБЩИЕ СВЕДЕНИЯ ОБ ОБЪЕКТЕ ПРОЕКТИРОВАНИЯ")
    _params_table("table_ch1")
    if st.button("Запуск", key="run_ch1"):
        _run_action(label="Глава 1", endpoint="/chapter1")

with tab2:
    st.caption("ВОЗДЕЙСТВИЕ ОБЪЕКТА НА ЗЕМЕЛЬНЫЕ РЕСУРСЫ")
    _params_table("table_ch2")
    if st.button("Запуск", key="run_ch2"):
        _run_action(label="Глава 2", endpoint="/chapter2")

with tab_all:
    st.caption("Генерация глав 0, 1 и 2 одним запросом")
    _params_table("table_all")
    if st.button("Запуск", key="run_all"):
        _run_action(label="Все главы", endpoint="/chapters/all")

if st.session_state.result_bytes:
    st.download_button(
        "Скачать результат",
        data=st.session_state.result_bytes,
        file_name=st.session_state.result_filename,
        mime="application/zip",
    )
