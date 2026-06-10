import uuid
import tempfile
from pathlib import Path

from main_base import main
from src.utils.logger import logger


def test_main_base():
    base = Path(__file__).parents[2]
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "project1"
        input_dir = base / "src" / "ecology_chapters" / "chapter0"
        main(
            template_docx_path=input_dir / "template.docx",
            placeholders_path=input_dir / "placeholders.json",
            table_placeholders_path=input_dir / "table_placeholders.json",
            project_parts_path=base / "data" / "IN" / "project1" / "trim" / "mini",
            output_path=output_dir,
            collection_name="test_data",
            test_mode="on",
        )

        assert (output_dir / "placeholders.json").is_file()
        assert (output_dir / "result_template.docx").is_file()
        assert (output_dir / "run.log").is_file()
        assert any((output_dir / "logs").glob("*.log"))


def test_main_base_create_new_uuid_collection_and_delete():
    logs = []
    handler_id = logger.add(lambda x: logs.append(x.record["message"]))
    try:
        base = Path(__file__).parents[2]
        collection_name = uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "project1"
            input_dir = base / "src" / "ecology_chapters" / "chapter0"
            main(
                template_docx_path=input_dir / "template.docx",
                placeholders_path=input_dir / "placeholders.json",
                table_placeholders_path=input_dir / "table_placeholders.json",
                project_parts_path=base / "data" / "OUT" / "project1",
                output_path=output_dir,
                collection_name=collection_name,
                test_mode="mock",
            )
            print(logs)
            assert (
                f"collection <{collection_name}> name is valid uuid and was deleted"
                in logs
            )
    finally:
        logger.remove(handler_id)
