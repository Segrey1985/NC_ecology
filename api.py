import io
import json
import uuid
import tempfile
import shutil
import zipfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from main import main as run_main
from src.utils.validators import validate_docx, validate_json, validate_zip
from src.utils.logger import logger

app = FastAPI(title="NC_ecology API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    placeholders: UploadFile = File(..., description="JSON с плейсхолдерами"),
    template_docx: UploadFile = File(..., description="DOCX шаблон"),
    table_placeholders: UploadFile | None = File(
        None, description="JSON с табличными плейсхолдерами (опционально)"
    ),
    project_parts_zip: UploadFile | None = File(
        None, description="Zip-архив с документами смежных разделов в формате pdf"
    ),
    collection_name: str = Form(uuid.uuid4().hex),
):

    # проверка типов приложенных файлов

    await validate_json(placeholders)

    if table_placeholders:
        await validate_json(table_placeholders)

    await validate_docx(template_docx)

    if project_parts_zip:
        await validate_zip(project_parts_zip)

    with tempfile.TemporaryDirectory(prefix="nc_ecology_") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "in"
        output_dir = tmp_dir / "out"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        placeholders_path = input_dir / "placeholders.json"
        placeholders_bytes = await placeholders.read()
        try:
            json.loads(placeholders_bytes.decode("utf-8"))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Некорректный JSON: {e}"
            ) from e
        placeholders_path.write_bytes(placeholders_bytes)

        table_placeholders_path: Path | None = None
        if table_placeholders:
            table_placeholders_path = input_dir / "table_placeholders.json"
            table_bytes = await table_placeholders.read()
            try:
                json.loads(table_bytes.decode("utf-8"))
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Некорректный JSON: {e}"
                ) from e
            table_placeholders_path.write_bytes(table_bytes)

        template_docx_path: Path | None = None
        if template_docx:
            template_docx_path = input_dir / "template.docx"
            template_docx_path.write_bytes(await template_docx.read())
        
        project_parts_dir: Path | None = None
        if project_parts_zip:
            project_parts_dir = tmp_dir / "project_parts"
            project_parts_dir.mkdir(parents=True, exist_ok=True)
            # Распаковываем project_parts_zip и собираем PDFs в директорию,
            # которую ожидает collect_project_parts (она смотрит только верхний уровень).
            project_parts_zip_bytes = await project_parts_zip.read()
            await project_parts_zip.seek(0)

            project_parts_raw_dir = tmp_dir / "project_parts_raw"
            project_parts_raw_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(project_parts_zip_bytes)) as zf:
                zf.extractall(project_parts_raw_dir)

            pdfs = sorted(project_parts_raw_dir.rglob("*.pdf"))
            if not pdfs:
                raise HTTPException(
                    status_code=400,
                    detail="В project_parts_zip не найдено ни одного PDF",
                )

            # Кладём PDFs в один уровень, избегая коллизий имён.
            for idx, pdf_path in enumerate(pdfs, start=1):
                safe_name = pdf_path.name
                dest = project_parts_dir / safe_name
                if dest.exists():
                    dest = project_parts_dir / f"{idx:04d}_{safe_name}"
                shutil.copy2(pdf_path, dest)
        
        logger.info(f"{template_docx_path=}")
        logger.info(f"{placeholders_path=}")
        logger.info(f"{table_placeholders_path=}")
        logger.info(f"{project_parts_dir=}")
        logger.info(f"{output_dir=}")
        logger.info(f"{collection_name=}")
        
        run_main(
            template_docx_path=template_docx_path,
            placeholders_path=placeholders_path,
            table_placeholders_path=table_placeholders_path,
            project_parts_path=project_parts_dir,
            output_path=output_dir,
            collection_name=collection_name,
            verbose=False,
            test_mode='off',
        )

        zip_buf = io.BytesIO()

        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    # сохраняем относительный путь внутри архива
                    arc_name = file_path.relative_to(output_dir)
                    zf.write(file_path, arcname=arc_name)

        zip_buf.seek(0)

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=result.zip"},
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
