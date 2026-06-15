from starlette.testclient import TestClient

from api.api_prod import app
from api.session_middleware import SESSION_COOKIE_NAME
from src.utils.utils import is_valid_uuid4_hex


def test_session_cookie_is_set_on_first_request():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert SESSION_COOKIE_NAME in response.cookies
    session_id = response.cookies[SESSION_COOKIE_NAME]
    assert is_valid_uuid4_hex(session_id)


def test_session_cookie_is_reused_on_subsequent_requests():
    with TestClient(app) as client:
        first = client.get("/health")
        session_id = first.cookies[SESSION_COOKIE_NAME]
        second = client.get("/health")

    assert client.cookies[SESSION_COOKIE_NAME] == session_id
    assert "set-cookie" not in second.headers
