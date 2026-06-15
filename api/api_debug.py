import uvicorn
from typing import Annotated
from fastapi import FastAPI, File, Form, Query, UploadFile

from config.config_file import TestMode
from api.api_utils import (
    CHAPTER0,
    CHAPTER1,
    CHAPTER2,
    generate_all_chapters,
    generate_chapter,
)
from api.session_middleware import add_session_middleware
from api.concurrency_middleware import add_concurrency_middleware

app = FastAPI(title="NC_ecology API", version="0.1.0")

# потом по куки блокируем
add_concurrency_middleware(app)

# сначала назначаем куки
add_session_middleware(app)


@app.get(
    "/health",
    summary="Проверка состояния сервиса",
    description="Эндпоинт используется для проверки доступности API.",
    tags=["System"],
)
def health():
    return {"status": "ok"}


@app.post(
    "/chapter0",
    summary="Аннотация и введение",
    description="Эндпоинт используется для генерации глав 'Аннотация' и 'Введение'",
    tags=["Generation"],
)
async def chapter0(
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Zip-архив с документами смежных разделов в формате pdf"
    ),
    placeholders: UploadFile | None = File(
        None,
        description="[### для отладки ###] JSON с плейсхолдерами (по умолчанию из src/ecology_chapters/chapter0)",
    ),
    table_placeholders: UploadFile | None = File(
        None,
        description="[### для отладки ###] JSON с табличными плейсхолдерами (по умолчанию из src/ecology_chapters/chapter0)",
    ),
    template_docx: UploadFile | None = File(
        None,
        description="[### для отладки ###] DOCX шаблон (по умолчанию из src/ecology_chapters/chapter0)",
    ),
    collection_name: str | None = Form(None, description="[### для отладки ###] Имя коллекции"),
):
    return await generate_chapter(
        spec=CHAPTER0,
        placeholders=placeholders,
        template_docx=template_docx,
        table_placeholders=table_placeholders,
        project_parts_zip=project_parts_zip,
        collection_name=collection_name,
    )


@app.post(
    "/chapter1",
    summary="Глава 1. ОБЩИЕ СВЕДЕНИЯ ОБ ОБЪЕКТЕ ПРОЕКТИРОВАНИЯ",
    description="Эндпоинт используется для генерации главы 'ОБЩИЕ СВЕДЕНИЯ ОБ ОБЪЕКТЕ ПРОЕКТИРОВАНИЯ'",
    tags=["Generation"],
)
async def chapter1(
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Zip-архив с PDF смежных разделов"
    ),
    table_placeholders: UploadFile | None = File(
        None,
        description="[### для отладки ###] JSON с табличными плейсхолдерами (по умолчанию из src/ecology_chapters/chapter1)",
    ),
    template_docx: UploadFile | None = File(
        None,
        description="[### для отладки ###] DOCX шаблон для сборки (по умолчанию из src/ecology_chapters/chapter1)",
    ),
    max_workers: int | None = Form(
        CHAPTER1.default_max_workers,
        description="[### для отладки ###] Число потоков для параллельного запуска моделей",
    ),
    collection_name: str | None = Form(None, description="[### для отладки ###] Имя коллекции"),
    extract_base: bool = Form(
        default=CHAPTER1.default_extract_base,
        description="[### для отладки ###] Извлекать базовую информацию",
    ),
):
    return await generate_chapter(
        spec=CHAPTER1,
        template_docx=template_docx,
        table_placeholders=table_placeholders,
        project_parts_zip=project_parts_zip,
        collection_name=collection_name,
        max_workers=max_workers,
        extract_base=extract_base,
    )


@app.post(
    "/chapter2",
    summary="Глава 2. ВОЗДЕЙСТВИЕ ОБЪЕКТА НА ЗЕМЕЛЬНЫЕ РЕСУРСЫ",
    description="Эндпоинт используется для генерации главы 'ВОЗДЕЙСТВИЕ ОБЪЕКТА НА ЗЕМЕЛЬНЫЕ РЕСУРСЫ'",
    tags=["Generation"],
)
async def chapter2(
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Zip-архив с PDF смежных разделов"
    ),
    table_placeholders: UploadFile | None = File(
        None,
        description="[### для отладки ###] JSON с табличными плейсхолдерами (по умолчанию из src/ecology_chapters/chapter2)",
    ),
    template_docx: UploadFile | None = File(
        None,
        description="[### для отладки ###] DOCX шаблон для сборки (по умолчанию из src/ecology_chapters/chapter2)",
    ),
    max_workers: int | None = Form(
        CHAPTER2.default_max_workers,
        description="[### для отладки ###] Число потоков для параллельного запуска моделей",
    ),
    collection_name: str | None = Form(None, description="[### для отладки ###] Имя коллекции"),
    extract_base: bool = Form(
        default=CHAPTER2.default_extract_base,
        description="[### для отладки ###] Извлекать базовую информацию",
    ),
):
    return await generate_chapter(
        spec=CHAPTER2,
        template_docx=template_docx,
        table_placeholders=table_placeholders,
        project_parts_zip=project_parts_zip,
        collection_name=collection_name,
        max_workers=max_workers,
        extract_base=extract_base,
    )


@app.post(
    "/chapters/all",
    summary="Запуск всех глав одним запросом",
    description=(
        "Генерирует главы 0, 1, 2 за один вызов и возвращает единый zip-архив с результатами. "
        "По умолчанию используются шаблоны и плейсхолдеры из `src/ecology_chapters`."
    ),
    tags=["Generation"],
)
async def chapters_all(
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Zip-архив с PDF смежных разделов"
    ),
    placeholders_ch0: UploadFile | None = File(
        None, description="[### для отладки ###] JSON плейсхолдеров для главы 0"
    ),
    template_docx_ch0: UploadFile | None = File(
        None, description="[### для отладки ###] DOCX шаблон для главы 0"
    ),
    table_placeholders_ch0: UploadFile | None = File(
        None, description="[### для отладки ###] JSON с табличными плейсхолдерами для главы 0"
    ),
    table_placeholders_ch1: UploadFile | None = File(
        None, description="[### для отладки ###] JSON с табличными плейсхолдерами для главы 1"
    ),
    table_placeholders_ch2: UploadFile | None = File(
        None, description="[### для отладки ###] JSON с табличными плейсхолдерами для главы 2"
    ),
    template_docx_ch1: UploadFile | None = File(
        None, description="[### для отладки ###] DOCX шаблон для главы 1"
    ),
    template_docx_ch2: UploadFile | None = File(
        None, description="[### для отладки ###] DOCX шаблон для главы 2"
    ),
    max_workers: int | None = Form(
        CHAPTER1.default_max_workers,
        description="[### для отладки ###] Число потоков для параллельного запуска моделей",
    ),
    collection_name: str | None = Form(None, description="[### для отладки ###] Имя коллекции"),
    test_mode: Annotated[TestMode, Query(include_in_schema=False)] = "off",
):
    return await generate_all_chapters(
        project_parts_zip=project_parts_zip,
        collection_name=collection_name,
        max_workers=max_workers,
        test_mode=test_mode,
        placeholders_ch0=placeholders_ch0,
        template_docx_ch0=template_docx_ch0,
        table_placeholders_ch0=table_placeholders_ch0,
        table_placeholders_ch1=table_placeholders_ch1,
        table_placeholders_ch2=table_placeholders_ch2,
        template_docx_ch1=template_docx_ch1,
        template_docx_ch2=template_docx_ch2,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
