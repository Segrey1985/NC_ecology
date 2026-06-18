from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

from main import main
from config.config_file import cfg
from tests.conftest import make_project_parts_zip

def test_main_test_mode_on_processes_only_first_model_and_cleanup():
    """
    Интеграционный тест уровня main2 БЕЗ моков/подмен:
    - реальный init_graph_2 + Qdrant + LLM
    - test_mode="on" должен обработать только 1 pydantic-модель

    После теста обязательно удаляем созданные файлы (output JSON).
    """
    repo_root = cfg.BASE_DIR
    trim_dir = repo_root / "data" / "IN" / "project1" / "trim"
    pdfs = sorted(trim_dir.glob("*.pdf"))[:2]
    assert len(pdfs) >= 1, f"Ожидался хотя бы 1 pdf в {trim_dir}"
    
    table_placeholders_path= Path("src/ecology_chapters/chapter1/table_placeholders.json")

    collection_name = uuid.uuid4().hex  # временная коллекция

    with tempfile.TemporaryDirectory(prefix="nc_ecology_main2_it_") as tmp:
        tmp_dir = Path(tmp)
        out_dir = tmp_dir / "out"
        out_json = out_dir / "chapter1_output.json"

        main(
            template_docx_path=None,
            project_parts_zip=make_project_parts_zip(pdfs),
            table_placeholders_path=table_placeholders_path,
            output_path=out_dir,
            chapter_module_path="src.ecology_chapters.chapter1",
            collection_name=collection_name,
            verbose=False,
            test_mode="on",
            max_workers=2,
        )

        assert out_json.exists()
        data = json.loads(out_json.read_text(encoding="utf-8"))
        data = {k: v for k, v in data.items() if v}
        assert isinstance(data, dict)
        
        def _is_placeholder(value) -> bool:
            if isinstance(value, str):
                return value.startswith("{{") and value.endswith("}}")
            if isinstance(value, dict):
                return all(_is_placeholder(v) for v in value.values())
            if isinstance(value, list) and len(value) == 1:
                return _is_placeholder(value[0])
            return False
        
        filled = [k for k, v in data.items() if not _is_placeholder(v)]
        assert len(filled) == 1 + len(json.loads(table_placeholders_path.read_text(encoding="utf-8")))
