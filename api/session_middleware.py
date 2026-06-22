import uuid

from fastapi import FastAPI, Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.utils.utils import is_valid_uuid4_hex

SESSION_COOKIE_NAME = "nc_ecology_session_id"
SESSION_MAX_AGE = 30 * 24 * 60 * 60  # 30 дней


def _resolve_session_id(request: Request) -> tuple[str, bool]:
    """
    Генерирует куки сессии или возвращает существующий, если он есть
    :return: (session_id, is_new_session)
    """
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if raw and is_valid_uuid4_hex(raw):
        return raw, False
    return uuid.uuid4().hex, True


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        session_cookie, is_new_session = _resolve_session_id(request)

        # резервируем session_cookie (в пределах этого request) для других обработчиков (здесь не используется)
        request.state.session_cookie = session_cookie


        logger.info(f"session_cookie={session_cookie} | {request.method} {request.url.path}")
        response = await call_next(request)

        if is_new_session:
            logger.info(f"session_cookie={session_cookie} | set new session cookie")
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=session_cookie,
                max_age=SESSION_MAX_AGE,
                httponly=True,
                samesite="lax",
                secure=False,
            )
        return response


def add_session_middleware(app: FastAPI) -> None:
    app.add_middleware(SessionMiddleware)
