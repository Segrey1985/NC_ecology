import uvicorn
from fastapi import FastAPI, File, UploadFile

from api.api_utils import (
    CHAPTER0,
    CHAPTER1,
    CHAPTER2,
    generate_all_chapters,
    generate_chapter,
)

app = FastAPI(title="NC_ecology API", version="0.1.0")


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
):
    return await generate_chapter(
        spec=CHAPTER0,
        project_parts_zip=project_parts_zip,
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
):
    return await generate_chapter(
        spec=CHAPTER1,
        project_parts_zip=project_parts_zip,
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
):
    return await generate_chapter(
        spec=CHAPTER2,
        project_parts_zip=project_parts_zip,
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
):
    return await generate_all_chapters(project_parts_zip=project_parts_zip)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
