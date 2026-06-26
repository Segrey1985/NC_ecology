import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from api.api_utils import (
    CHAPTER0,
    CHAPTER1,
    CHAPTER2,
    CHAPTER6,
    generate_all_chapters,
    generate_chapter,
)
from api.session_middleware import add_session_middleware
from api.concurrency_middleware import add_concurrency_middleware
from src.mongo.mongo_client import connect_mongo, disconnect_mongo

# Каталог со статикой фронтенда (../static относительно api/)
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_mongo()      # старт uvicorn / docker
    yield
    await disconnect_mongo()   # остановка приложения


app = FastAPI(title="NC_ecology API", version="0.1.0", lifespan=lifespan)
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
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с документами смежных разделов в формате pdf"
    ),
):
    return await generate_chapter(
        request=request,
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
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с PDF смежных разделов"
    ),
):
    return await generate_chapter(
        request=request,
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
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с PDF смежных разделов"
    ),
):
    return await generate_chapter(
        request=request,
        spec=CHAPTER2,
        project_parts_zip=project_parts_zip,
    )


@app.post(
    "/chapter6",
    summary="Глава 6. ВОЗДЕЙСТВИЕ НА РАСТИТЕЛЬНЫЙ И ЖИВОТНЫЙ МИР",
    description="Эндпоинт используется для генерации главы 'ВОЗДЕЙСТВИЕ НА РАСТИТЕЛЬНЫЙ И ЖИВОТНЫЙ МИР'",
    tags=["Generation"],
)
async def chapter6(
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с PDF смежных разделов"
    ),
):
    return await generate_chapter(
        request=request,
        spec=CHAPTER6,
        project_parts_zip=project_parts_zip,
    )


@app.post(
    "/chapters/all",
    summary="Запуск всех глав одним запросом",
    description=(
        "Генерирует главы 0, 1, 2 и 6 за один вызов и возвращает единый zip-архив с результатами. "
        "По умолчанию используются шаблоны и плейсхолдеры из `src/ecology_chapters`."
    ),
    tags=["Generation"],
)
async def chapters_all(
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с PDF смежных разделов"
    ),
):
    return await generate_all_chapters(
        request=request,
        project_parts_zip=project_parts_zip
    )


# ── Фронтенд (веб-интерфейс) ──────────────────────────────────────────────────
# Корневая страница отдаёт index.html из каталога static.
@app.get("/", include_in_schema=False)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"detail": "Фронтенд не найден. Откройте /docs для работы с API."}


# Раздача статических ресурсов (если появятся css/js/изображения).
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
