"""
NC_ecology API — асинхронный режим.

Каждый POST /chapter* или /chapters/all немедленно возвращает {"task_id": "..."}.
Генерация выполняется в фоне.
Клиент опрашивает GET /task/{task_id}/status (JSON) каждые несколько секунд.
По готовности скачивает ZIP через GET /task/{task_id}/download.
"""

import asyncio
import os

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
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
from api.task_manager import TaskStatus, task_manager
from src.mongo.mongo_client import connect_mongo, disconnect_mongo

# Каталог со статикой фронтенда
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)


# ── Фоновая очистка устаревших задач ─────────────────────────────────────────
async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(600)  # каждые 10 минут
        await task_manager.cleanup_expired()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_mongo()
    asyncio.create_task(_cleanup_loop())
    yield
    await disconnect_mongo()


app = FastAPI(title="NC_ecology API", version="0.1.0", lifespan=lifespan)
add_concurrency_middleware(app)
add_session_middleware(app)


# ── Вспомогательная функция: запуск генерации в фоне ─────────────────────────
async def _run_in_background(coro, task_id: str) -> None:
    """Выполняет корутину генерации и сохраняет результат в task_manager."""
    await task_manager.set_running(task_id, "Индексируем документы и запускаем генерацию...")
    try:
        response = await coro
        # generate_chapter / generate_all_chapters возвращают StreamingResponse или Response
        # Читаем тело ответа
        if hasattr(response, "body"):
            body = response.body
        elif hasattr(response, "body_iterator"):
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
            body = b"".join(chunks)
        else:
            body = b""

        # Извлекаем имя файла из заголовка Content-Disposition
        filename = "result.zip"
        if hasattr(response, "headers"):
            cd = response.headers.get("content-disposition", "")
            if "filename=" in cd:
                filename = cd.split("filename=")[-1].strip().strip('"')

        await task_manager.set_done(task_id, body, filename)
    except HTTPException as exc:
        await task_manager.set_error(task_id, exc.detail)
    except Exception as exc:
        await task_manager.set_error(task_id, str(exc))


# ── Системные эндпоинты ───────────────────────────────────────────────────────
@app.get("/health", summary="Проверка состояния сервиса", tags=["System"])
def health():
    return {"status": "ok"}


# ── Статус и скачивание задачи ────────────────────────────────────────────────
@app.get(
    "/task/{task_id}/status",
    summary="Статус задачи генерации",
    tags=["Tasks"],
)
async def task_status(task_id: str):
    task = await task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task.to_dict()


@app.get(
    "/task/{task_id}/download",
    summary="Скачать результат задачи",
    tags=["Tasks"],
)
async def task_download(task_id: str):
    task = await task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if task.status == TaskStatus.RUNNING or task.status == TaskStatus.PENDING:
        raise HTTPException(status_code=202, detail="Задача ещё выполняется")
    if task.status == TaskStatus.ERROR:
        raise HTTPException(status_code=500, detail=task.error or "Ошибка генерации")
    if not task.result_bytes:
        raise HTTPException(status_code=500, detail="Результат пуст")

    return Response(
        content=task.result_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{task.filename}"',
            "Content-Length": str(len(task.result_bytes)),
        },
    )


# ── Эндпоинты генерации (возвращают task_id немедленно) ──────────────────────
@app.post(
    "/chapter0",
    summary="Аннотация и введение",
    tags=["Generation"],
)
async def chapter0(
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с PDF смежных разделов"
    ),
):
    task = await task_manager.create()
    coro = generate_chapter(request=request, spec=CHAPTER0, project_parts_zip=project_parts_zip)
    asyncio.create_task(_run_in_background(coro, task.task_id))
    return {"task_id": task.task_id}


@app.post(
    "/chapter1",
    summary="Глава 1. ОБЩИЕ СВЕДЕНИЯ ОБ ОБЪЕКТЕ ПРОЕКТИРОВАНИЯ",
    tags=["Generation"],
)
async def chapter1(
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с PDF смежных разделов"
    ),
):
    task = await task_manager.create()
    coro = generate_chapter(request=request, spec=CHAPTER1, project_parts_zip=project_parts_zip)
    asyncio.create_task(_run_in_background(coro, task.task_id))
    return {"task_id": task.task_id}


@app.post(
    "/chapter2",
    summary="Глава 2. ВОЗДЕЙСТВИЕ ОБЪЕКТА НА ЗЕМЕЛЬНЫЕ РЕСУРСЫ",
    tags=["Generation"],
)
async def chapter2(
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с PDF смежных разделов"
    ),
):
    task = await task_manager.create()
    coro = generate_chapter(request=request, spec=CHAPTER2, project_parts_zip=project_parts_zip)
    asyncio.create_task(_run_in_background(coro, task.task_id))
    return {"task_id": task.task_id}


@app.post(
    "/chapter6",
    summary="Глава 6. ВОЗДЕЙСТВИЕ НА РАСТИТЕЛЬНЫЙ И ЖИВОТНЫЙ МИР",
    tags=["Generation"],
)
async def chapter6(
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с PDF смежных разделов"
    ),
):
    task = await task_manager.create()
    coro = generate_chapter(request=request, spec=CHAPTER6, project_parts_zip=project_parts_zip)
    asyncio.create_task(_run_in_background(coro, task.task_id))
    return {"task_id": task.task_id}


@app.post(
    "/chapters/all",
    summary="Запуск всех глав одним запросом",
    tags=["Generation"],
)
async def chapters_all(
    request: Request,
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Архив (ZIP или RAR) с PDF смежных разделов"
    ),
):
    task = await task_manager.create()
    coro = generate_all_chapters(request=request, project_parts_zip=project_parts_zip)
    asyncio.create_task(_run_in_background(coro, task.task_id))
    return {"task_id": task.task_id}


# ── Фронтенд ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"detail": "Фронтенд не найден. Откройте /docs для работы с API."}


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
