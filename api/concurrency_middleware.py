import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

_active_sessions: set[str] = set()
_meta_lock = asyncio.Lock()

CONCURRENT_REQUEST_DETAIL = (
    "Уже выполняется генерация. Дождитесь завершения текущего запроса."
)

# Пути, которые НЕ блокируются concurrency-проверкой.
# Polling задач, скачивание результата, статика, health — всегда пропускаются.
_PASSTHROUGH_PREFIXES = (
    "/task/",
    "/health",
    "/static/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
)
_PASSTHROUGH_EXACT = {"/", "/health"}


def _is_passthrough(path: str) -> bool:
    if path in _PASSTHROUGH_EXACT:
        return True
    return any(path.startswith(p) for p in _PASSTHROUGH_PREFIXES)


async def try_acquire_session(session_id: str) -> bool:
    """
    Попытка занять сеанс.
    Если куки уже в _active_sessions — False (пользователь уже что-то выполняет);
    Если нет — добавляем и возвращаем True.
    """
    async with _meta_lock:
        if session_id in _active_sessions:
            return False
        _active_sessions.add(session_id)
        return True


async def release_session(session_id: str) -> None:
    async with _meta_lock:
        _active_sessions.discard(session_id)


def reset_active_sessions() -> None:
    _active_sessions.clear()


class ConcurrencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Пути, не требующие проверки конкурентности — пропускаем сразу
        if _is_passthrough(request.url.path):
            return await call_next(request)

        session_cookie = getattr(request.state, "session_cookie", None)
        if not session_cookie:
            return await call_next(request)

        if await try_acquire_session(session_cookie) is False:
            logger.info(
                f"session_cookie={session_cookie} | concurrent generation request rejected"
            )
            return JSONResponse(
                status_code=429,
                content={"detail": CONCURRENT_REQUEST_DETAIL},
            )
        try:
            return await call_next(request)
        finally:
            # В асинхронном режиме сессия освобождается сразу после приёма задачи
            # (task_id уже выдан), а не после завершения генерации.
            await release_session(session_cookie)


def add_concurrency_middleware(app: FastAPI) -> None:
    app.add_middleware(ConcurrencyMiddleware)
