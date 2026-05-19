from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

from main2 import main
from config.config_file import cfg

def test_main2_test_mode_on_processes_only_first_model_and_cleanup():
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

    collection_name = uuid.uuid4().hex  # временная коллекция

    with tempfile.TemporaryDirectory(prefix="nc_ecology_main2_it_") as tmp:
        tmp_dir = Path(tmp)
        
        parts_dir = tmp_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        
        for pdf in pdfs:
            shutil.copy2(pdf, parts_dir / pdf.name)

        out_dir = tmp_dir / "out"
        out_json = out_dir / "chapter1_models_output.json"

        main(
            template_docx_path=None,
            project_parts_path=parts_dir,
            table_placeholders_path=Path("src/ecology_chapters/chapter1/table_placeholders.json"),
            output_path=out_dir,
            chapter_module_path="src.ecology_chapters.chapter1",
            collection_name=collection_name,
            verbose=False,
            test_mode="on",
            max_workers=2,
        )

        assert out_json.exists()
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert len(data) == 1  # test_mode="on" => только 1 модель
