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

from main_base import main as run_main_base
from main import main as run_main
from src.utils.validators import validate_docx, validate_json, validate_zip
from src.utils.logger import logger

app = FastAPI(title="NC_ecology API", version="0.1.0")


def _extract_project_parts_pdfs(
    project_parts_zip_bytes: bytes, project_parts_dir: Path
) -> None:
    project_parts_raw_dir = project_parts_dir.parent / "project_parts_raw"
    project_parts_raw_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(project_parts_zip_bytes)) as zf:
        zf.extractall(project_parts_raw_dir)

    pdfs = sorted(project_parts_raw_dir.rglob("*.pdf"))
    if not pdfs:
        raise HTTPException(
            status_code=400,
            detail="В project_parts_zip не найдено ни одного PDF",
        )

    for idx, pdf_path in enumerate(pdfs, start=1):
        safe_name = pdf_path.name
        dest = project_parts_dir / safe_name
        if dest.exists():
            dest = project_parts_dir / f"{idx:04d}_{safe_name}"
        shutil.copy2(pdf_path, dest)


@app.get(
    "/health",
    summary="Проверка состояния сервиса",
    description="Эндпоинт используется для проверки доступности API.",
    tags=["System"],
)
def health():
    return {"status": "ok"}


@app.post(
    "/chapter0",
    summary="Аннотация и введение",
    description="Эндпоинт используется для генерации глав 'Аннотация' и 'Введение'",
    tags=["Generation"],
)
async def chapter0(
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Zip-архив с документами смежных разделов в формате pdf"
    ),
    placeholders: UploadFile = File(
        ..., description="[для отладки] JSON с плейсхолдерами"
    ),
    table_placeholders: UploadFile | None = File(
        None, description="[для отладки] JSON с табличными плейсхолдерами (опционально)"
    ),
    template_docx: UploadFile = File(
        ..., description="[для отладки] DOCX шаблон"
    ),
    collection_name: str = Form(uuid.uuid4().hex, description="[для отладки] Имя коллекции"),
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
            _extract_project_parts_pdfs(project_parts_zip_bytes, project_parts_dir)

        logger.info(f"{template_docx_path=}")
        logger.info(f"{placeholders_path=}")
        logger.info(f"{table_placeholders_path=}")
        logger.info(f"{project_parts_dir=}")
        logger.info(f"{output_dir=}")
        logger.info(f"{collection_name=}")

        run_main_base(
            template_docx_path=template_docx_path,
            placeholders_path=placeholders_path,
            table_placeholders_path=table_placeholders_path,
            project_parts_path=project_parts_dir,
            output_path=output_dir,
            collection_name=collection_name,
            verbose=False,
            test_mode="off",
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


# -------------------------------------------- chapters 1, 2, ... --------------------------------------------


async def _generate_chapter(
    *,
    chapter_module_path: str,
    template_docx: UploadFile | None,
    project_parts_zip: UploadFile | None,
    collection_name: str | None,
    max_workers: int | None,
    table_placeholders: UploadFile | None = None,
    result_filename: str = "result.zip",
):
    if template_docx:
        await validate_docx(template_docx)

    if table_placeholders:
        await validate_json(table_placeholders)

    if project_parts_zip:
        await validate_zip(project_parts_zip)

    with tempfile.TemporaryDirectory(prefix="nc_ecology_") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "in"
        output_dir = tmp_dir / "out"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        template_docx_path: Path | None = None
        if template_docx:
            template_docx_path = input_dir / "template.docx"
            template_docx_path.write_bytes(await template_docx.read())

        table_placeholders_path: Path | None = None
        if table_placeholders:
            table_placeholders_path = input_dir / "table_placeholders.json"
            table_placeholders_path.write_bytes(await table_placeholders.read())

        project_parts_dir: Path | None = None
        if project_parts_zip:
            project_parts_dir = tmp_dir / "project_parts"
            project_parts_dir.mkdir(parents=True, exist_ok=True)
            project_parts_zip_bytes = await project_parts_zip.read()
            _extract_project_parts_pdfs(project_parts_zip_bytes, project_parts_dir)

        collection_name = collection_name or uuid.uuid4().hex

        logger.info(f"generate {template_docx_path=}")
        logger.info(f"generate {table_placeholders_path=}")
        logger.info(f"generate {project_parts_dir=}")
        logger.info(f"generate {output_dir=}")
        logger.info(f"generate {collection_name=}")
        logger.info(f"generate {max_workers=}")

        run_main(
            template_docx_path=template_docx_path,
            project_parts_path=project_parts_dir,
            table_placeholders_path=table_placeholders_path,
            output_path=output_dir,
            chapter_module_path=chapter_module_path,
            collection_name=collection_name,
            verbose=False,
            test_mode="off",
            max_workers=max_workers,
        )

        zip_buf = io.BytesIO()

        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    arc_name = file_path.relative_to(output_dir)
                    zf.write(file_path, arcname=arc_name)

        zip_buf.seek(0)

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={result_filename}"},
        )


@app.post(
    "/chapter1",
    summary="Глава 1. ОБЩИЕ СВЕДЕНИЯ ОБ ОБЪЕКТЕ ПРОЕКТИРОВАНИЯ",
    description="Эндпоинт используется для генерации главы 'ОБЩИЕ СВЕДЕНИЯ ОБ ОБЪЕКТЕ ПРОЕКТИРОВАНИЯ'",
    tags=["Generation"],
)
async def chapter1(
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Zip-архив с PDF смежных разделов"
    ),
    template_docx: UploadFile | None = File(
        None, description="[для отладки] DOCX шаблон для сборки (опционально)"
    ),
    max_workers: int | None = Form(
        8, description="[для отладки] Число потоков для параллельного запуска моделей"
    ),
    collection_name: str = Form(uuid.uuid4().hex, description="[для отладки] Имя коллекции"),
):
    return await _generate_chapter(
        chapter_module_path="src.ecology_chapters.chapter1",
        template_docx=template_docx,
        project_parts_zip=project_parts_zip,
        collection_name=collection_name,
        max_workers=max_workers,
        result_filename="result2.zip",
    )


@app.post(
    "/chapter2",
    summary="Глава 2. ВОЗДЕЙСТВИЕ ОБЪЕКТА НА ЗЕМЕЛЬНЫЕ РЕСУРСЫ",
    description="Эндпоинт используется для генерации главы 'ВОЗДЕЙСТВИЕ ОБЪЕКТА НА ЗЕМЕЛЬНЫЕ РЕСУРСЫ'",
    tags=["Generation"],
)
async def chapter2(
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Zip-архив с PDF смежных разделов"
    ),
    template_docx: UploadFile | None = File(
        None, description="[для отладки] DOCX шаблон для сборки (опционально)"
    ),
    max_workers: int | None = Form(
        8, description="[для отладки] Число потоков для параллельного запуска моделей"
    ),
    collection_name: str = Form(uuid.uuid4().hex, description="[для отладки] Имя коллекции"),
):
    return await _generate_chapter(
        chapter_module_path="src.ecology_chapters.chapter2",
        template_docx=template_docx,
        project_parts_zip=project_parts_zip,
        collection_name=collection_name,
        max_workers=max_workers,
        result_filename="chapter2.zip",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
