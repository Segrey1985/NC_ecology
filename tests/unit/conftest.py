import pytest
from fastapi import FastAPI

from api.session_middleware import add_session_middleware


@pytest.fixture
def session_app() -> FastAPI:
    app = FastAPI()
    add_session_middleware(app)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
