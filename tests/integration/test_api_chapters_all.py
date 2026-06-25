import io
import uuid
import zipfile
from pathlib import Path

from starlette.testclient import TestClient

from api.api_debug import app
from src.utils.logger import logger


def _make_project_parts_zip(pdf_paths: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdf_path in pdf_paths:
            zf.write(pdf_path, arcname=pdf_path.name)
    return buf.getvalue()


def test_chapters_all_test_mode_mock():
    """
    POST /chapters/all с test_mode=mock:
    - возвращает zip с результатами chapter0/1/2/6
    """
    base = Path(__file__).resolve().parents[2]
    trim_dir = base / "data" / "IN" / "project1" / "trim"
    pdfs = sorted(trim_dir.glob("*.pdf"))[:2]
    assert len(pdfs) == 2, f"Ожидалось 2 pdf в {trim_dir}, найдено: {len(pdfs)}"

    logs: list[str] = []
    handler_id = logger.add(lambda x: logs.append(x.record["message"]))
    collection_name = uuid.uuid4().hex

    try:
        zip_bytes = _make_project_parts_zip(pdfs)
        with TestClient(app) as client:
            response = client.post(
                "/chapters/all?test_mode=mock",
                data={
                    "collection_name": collection_name,
                    "extract_base": "true",
                    "max_workers": "1",
                },
                files={
                    "project_parts_zip": ("project_parts.zip", zip_bytes, "application/zip"),
                },
            )

        assert response.status_code == 200, response.text
        assert "application/zip" in response.headers.get("content-type", "")

        with zipfile.ZipFile(io.BytesIO(response.content)) as result_zip:
            names = set(result_zip.namelist())
            assert any(n.startswith("chapter0/") and n.endswith("placeholders.json") for n in names)
            assert any(n.startswith("chapter1/") and n.endswith("chapter1_output.json") for n in names)
            assert any(n.startswith("chapter2/") and n.endswith("chapter2_output.json") for n in names)
            assert any(n.startswith("chapter6/") and n.endswith("chapter6_output.json") for n in names)
    finally:
        logger.remove(handler_id)
