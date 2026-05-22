import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from src.agents.agent_base import init_graph
from src.retrieval.qdrant import build_qdrant_service
from config.config_file import build_runtime_config


def test_init_graph_unknown_collection_without_project_parts_path_raises():
    with pytest.raises(ValueError, match="project_parts_path не передан"):
        init_graph(collection_name=uuid.uuid4().hex, project_parts_path=None, runtime_cfg=build_runtime_config('on'))


def test_init_graph_unknown_collection_with_project_parts_path_first2_then_cleanup():
    base = Path(__file__).resolve().parents[2]
    trim_dir = base / "data" / "IN" / "project1" / "trim"
    pdfs = sorted(trim_dir.glob("*.pdf"))[:2]
    assert len(pdfs) == 2, f"Ожидалось 2 pdf в {trim_dir}, найдено: {len(pdfs)}"

    qdrant_service = build_qdrant_service(runtime_cfg=build_runtime_config('on'))
    collection_name = uuid.uuid4().hex

    with tempfile.TemporaryDirectory(prefix="nc_ecology_agent_first2_") as tmp:
        tmp_dir = Path(tmp)
        for pdf_path in pdfs:
            shutil.copy2(pdf_path, tmp_dir / pdf_path.name)

        try:
            if qdrant_service.client.collection_exists(collection_name):
                qdrant_service.client.delete_collection(collection_name)

            init_graph(collection_name=collection_name, project_parts_path=tmp_dir, runtime_cfg=build_runtime_config('on'))
            assert qdrant_service.client.collection_exists(collection_name)
        finally:
            if qdrant_service.client.collection_exists(collection_name):
                qdrant_service.client.delete_collection(collection_name)
