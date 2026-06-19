import importlib
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from src.utils.utils import FIELD_TO_MODEL_ATTR

def test_iter_models_from_module_filters_imported_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Создаём временный пакет и модуль, чтобы проверить поведение importlib.
    pkg_dir = tmp_path / "tmp_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    (pkg_dir / "inner.py").write_text(
        "from pydantic import BaseModel\n\n"
        "class Imported(BaseModel):\n"
        "    x: int\n",
        encoding="utf-8",
    )
    (pkg_dir / "models_mod.py").write_text(
        "from pydantic import BaseModel\n"
        "from .inner import Imported\n\n"
        "class LocalA(BaseModel):\n"
        "    a: int\n\n"
        "class LocalB(BaseModel):\n"
        "    b: int\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    from src.utils.utils import iter_models_from_module

    models = iter_models_from_module("tmp_pkg.models_mod")
    assert [m.__name__ for m in models] == ["LocalA", "LocalB"]


def test_thread_run_graph_for_model_parses_json(monkeypatch: pytest.MonkeyPatch):
    from main import thread_run_graph_for_model

    class M(BaseModel):
        x: int

    def fake_run_graph(*_args, **_kwargs):
        return '{"x": 1}'

    monkeypatch.setattr("main._run_graph", fake_run_graph)

    _results = thread_run_graph_for_model(
        graph=object(), model=M, chapter_module_path="tests.fake_chapter", verbose=False
    )
    assert _results["model_name"] == "M"
    assert _results["result"] == {"x": 1}


def test_main_writes_output_and_deletes_uuid_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Не запускаем реальный граф/LLM/Qdrant. Проверяем:
    - запись результата в chapter1_models_output.json
    - удаление коллекции, если имя uuid4 hex и collection_exists=True
    """
    import main

    class M(BaseModel):
        x: int

    class FakeClient:
        def __init__(self):
            self.deleted: list[str] = []

        def collection_exists(self, _name: str) -> bool:
            return True

        def delete_collection(self, name: str):
            self.deleted.append(name)

    class FakeQdrantService:
        def __init__(self):
            self.client = FakeClient()

    class FakeResources:
        def __init__(self):
            self.qdrant_service = FakeQdrantService()

    fake_resources = FakeResources()

    class FakeAssembly(BaseModel):
        m: str | None = None

    setattr(FakeAssembly, FIELD_TO_MODEL_ATTR, {"m": M})

    monkeypatch.setattr("main.init_graph", lambda *args, **kwargs: (object(), fake_resources))
    monkeypatch.setattr("main.iter_models_from_module", lambda _p: [M])
    monkeypatch.setattr("main.pick_assembly_model", lambda _p: FakeAssembly)
    monkeypatch.setattr(
        "main.thread_run_graph_for_model",
        lambda **_kwargs: {
            "model_name":"M",
            "result": '{"x": 123}',
            "logs_lines": [],
            "model": M,
        }
    )

    collection_name = "a" * 32  # валидный uuid4 hex проходит проверку в is_valid_uuid4_hex
    out_dir = tmp_path / "out"
    main.main(
        template_docx_path=None,
        project_parts_zip=None,
        table_placeholders_path=None,
        output_path=out_dir,
        chapter_module_path="does.not.matter",
        collection_name=collection_name,
        test_mode="off",
        max_workers=1,
    )

    out_json = out_dir / "matter_output.json"
    assert out_json.exists()
    assert out_json.read_text(encoding="utf-8").strip() != ""
    assert '"m"' in out_json.read_text(encoding="utf-8")
    assert fake_resources.qdrant_service.client.deleted == [collection_name]


def test_main_filter_mode_uses_iter_chapter_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import main

    class M(BaseModel):
        x: int

    called = {"filter": False}

    class FakeAssembly(BaseModel):
        m: dict | None = None

    setattr(FakeAssembly, FIELD_TO_MODEL_ATTR, {"m": M})

    def fake_iter_chapter(_path: str):
        called["filter"] = True
        return [M]

    monkeypatch.setattr("main.init_graph", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr("main.iter_chapter_models", fake_iter_chapter)
    monkeypatch.setattr("main.pick_assembly_model", lambda _p: FakeAssembly)
    monkeypatch.setattr(
        "main.thread_run_graph_for_model",
        lambda **_kwargs: {
            "model_name": "M",
            "result": {"x": 1},
            "logs_lines": [],
            "model": M,
        },
    )

    main.main(
        template_docx_path=None,
        project_parts_zip=None,
        table_placeholders_path=None,
        output_path=tmp_path / "out",
        chapter_module_path="x.y",
        collection_name="main",
        test_mode="filter",
        max_workers=1,
    )
    assert called["filter"]


def test_main_raises_if_no_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from main import main

    monkeypatch.setattr("main.init_graph", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr("main.iter_models_from_module", lambda _p: [])

    with pytest.raises(RuntimeError, match="Не нашёл pydantic-моделей"):
        main(
            template_docx_path=None,
            project_parts_zip=None,
            table_placeholders_path=None,
            output_path=tmp_path / "out",
            chapter_module_path="x.y",
            collection_name="main",
            test_mode="off",
            max_workers=1,
        )

