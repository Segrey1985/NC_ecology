from pathlib import Path
import tempfile

from main import main


def test_main():
    base = Path(__file__).parents[2]
    with tempfile.TemporaryDirectory(prefix="nc_ecology_test_") as tmp_dir:
        output_dir = Path(tmp_dir) / "project1"
        input_dir = base / "data" / "IN" / "templates" / "0_Анализ_и_Введение"
        main(
            template_docx_path=input_dir / "template.docx",
            placeholders_path=input_dir / "placeholders.json",
            table_placeholders_path=input_dir / "table_placeholders.json",
            project_parts_path=None,
            output_path=output_dir,
            collection_name="test_data",
            test_mode=True,
        )

        created_files = [p for p in output_dir.rglob("*") if p.is_file()]
        assert (
            len(created_files) == 2
        ), f"Ожидалось 2 файла, найдено {len(created_files)}: {created_files}"
