import io
import uuid
import json
import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import Literal, Annotated
from qdrant_client import QdrantClient
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import StreamingResponse

from config.config_file import cfg, TestMode
from main_base import main as run_main_base
from main import main as run_main
from src.utils.validators import validate_docx, validate_json, validate_zip
from src.utils.utils import is_valid_uuid4_hex
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


def _zip_output_dir(output_dir: Path, result_filename: str) -> StreamingResponse:
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


async def _generate_chapter(
    *,
    pipeline: Literal["base", "chapter"],
    template_docx: UploadFile | None,
    project_parts_zip: UploadFile | None,
    collection_name: str | None,
    chapter_module_path: str | None = None,
    placeholders: UploadFile | None = None,
    max_workers: int | None = None,
    table_placeholders: UploadFile | None = None,
    result_filename: str = "result.zip",
    extract_base: bool = True,
):
    
    # валидация входных документов
    
    if pipeline == "base":
        if placeholders is None or template_docx is None:
            raise HTTPException(
                status_code=500,
                detail="Для pipeline='base' требуются placeholders и template_docx",
            )
        await validate_json(placeholders)
        await validate_docx(template_docx)
    else:
        if chapter_module_path is None:
            raise HTTPException(
                status_code=500,
                detail="Для pipeline='chapter' требуется chapter_module_path",
            )
        if template_docx:
            await validate_docx(template_docx)

    if table_placeholders:
        await validate_json(table_placeholders)

    if project_parts_zip:
        await validate_zip(project_parts_zip)

    # создание временной папки
    
    with tempfile.TemporaryDirectory(prefix="nc_ecology_") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "in"
        output_dir = tmp_dir / "out"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # запись входных документов во временную папку

        placeholders_path: Path | None = None
        if pipeline == "base":
            placeholders_path = input_dir / "placeholders.json"
            placeholders_path.write_bytes(await placeholders.read())

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

        logger.info(f"{pipeline=}")
        logger.info(f"{template_docx_path=}")
        logger.info(f"{placeholders_path=}")
        logger.info(f"{table_placeholders_path=}")
        logger.info(f"{project_parts_dir=}")
        logger.info(f"{collection_name=}")
        logger.info(f"{chapter_module_path=}")
        logger.info(f"{max_workers=}")
        logger.info(f"{output_dir=}")

        if pipeline == "base":
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
        else:
            
            if extract_base:
                logger.info(f"[extract_base] START")
                project_path = Path(__file__).parent
                chapter0_path = project_path / "src" / "ecology_chapters" / "chapter0"
                
                # запускаем main_base и получаем chapter0-плейсхолдеры
                base_placeholders: dict = run_main_base(
                    template_docx_path=chapter0_path / "template.docx",
                    placeholders_path=chapter0_path / "placeholders.json",
                    table_placeholders_path=chapter0_path / "table_placeholders.json",
                    project_parts_path=project_parts_dir,
                    output_path=None,
                    collection_name=collection_name,
                    verbose=False,
                    test_mode="off",
                )
                logger.info(f"[extract_base] {base_placeholders.keys()=}")
                # достаем табличные плейсхолдеры главы X, если они есть
                chapter_x_table_placeholders = {}
                if table_placeholders:
                    with open(table_placeholders_path, 'r', encoding='utf-8') as table_placeholders_file:
                        chapter_x_table_placeholders = json.load(table_placeholders_file)
                logger.info(f"[extract_base] {chapter_x_table_placeholders.keys()=}")
                
                # добавляем к табличным плейсхолдерам главы X, chapter0-плейсхолдеры, сохраняем
                chapter_x_table_placeholders.update(base_placeholders)
                logger.info(f"[extract_base] {chapter_x_table_placeholders.keys()=}")
                if not table_placeholders_path:
                    table_placeholders_path = input_dir / "table_placeholders.json"
                json.dump(chapter_x_table_placeholders, open(table_placeholders_path, 'w', encoding='utf-8'), indent=4)
                logger.info(f"[extract_base] END")
            
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

        return _zip_output_dir(output_dir, result_filename)


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
        ..., description="[### для отладки ###] JSON с плейсхолдерами"
    ),
    table_placeholders: UploadFile | None = File(
        None, description="[### для отладки ###] JSON с табличными плейсхолдерами (опционально)"
    ),
    template_docx: UploadFile = File(
        ..., description="[### для отладки ###] DOCX шаблон"
    ),
    collection_name: str = Form(uuid.uuid4().hex, description="[### для отладки ###] Имя коллекции"),
):
    return await _generate_chapter(
        pipeline="base",
        placeholders=placeholders,
        template_docx=template_docx,
        table_placeholders=table_placeholders,
        project_parts_zip=project_parts_zip,
        collection_name=collection_name,
        result_filename="result.zip",
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
    table_placeholders: UploadFile | None = File(
        None, description="[опционально] JSON с табличными плейсхолдерами"
    ),
    template_docx: UploadFile | None = File(
        None, description="[### для отладки ###] DOCX шаблон для сборки (опционально)"
    ),
    max_workers: int | None = Form(
        8, description="[### для отладки ###] Число потоков для параллельного запуска моделей"
    ),
    collection_name: str = Form(uuid.uuid4().hex, description="[### для отладки ###] Имя коллекции"),
    extract_base: bool = Form(default=True, description="[### для отладки ###] Извлекать базовую информацию"),
):
    return await _generate_chapter(
        pipeline="chapter",
        chapter_module_path="src.ecology_chapters.chapter1",
        template_docx=template_docx,
        table_placeholders=table_placeholders,
        project_parts_zip=project_parts_zip,
        collection_name=collection_name,
        max_workers=max_workers,
        result_filename="result2.zip",
        extract_base=extract_base,
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
    table_placeholders: UploadFile | None = File(
        None, description="[опционально] JSON с табличными плейсхолдерами"
    ),
    template_docx: UploadFile | None = File(
        None, description="[### для отладки ###] DOCX шаблон для сборки (опционально)"
    ),
    max_workers: int | None = Form(
        8, description="[### для отладки ###] Число потоков для параллельного запуска моделей"
    ),
    collection_name: str = Form(uuid.uuid4().hex, description="[### для отладки ###] Имя коллекции"),
    extract_base: bool = Form(default=True, description="[### для отладки ###] Извлекать базовую информацию"),
):
    return await _generate_chapter(
        pipeline="chapter",
        chapter_module_path="src.ecology_chapters.chapter2",
        template_docx=template_docx,
        table_placeholders=table_placeholders,
        project_parts_zip=project_parts_zip,
        collection_name=collection_name,
        max_workers=max_workers,
        result_filename="chapter2.zip",
        extract_base=extract_base,
    )


@app.post(
    "/chapters/all",
    summary="Запуск всех глав одним запросом",
    description=(
        "Генерирует главы 0, 1, 2 за один вызов и возвращает единый zip-архив с результатами. "
        "По умолчанию используются шаблоны и плейсхолдеры из `src/ecology_chapters`."
    ),
    tags=["Generation"],
)
async def chapters_all(
    project_parts_zip: UploadFile | None = File(
        None, description="[обязательно] Zip-архив с PDF смежных разделов"
    ),
    # Переопределения для главы 0 (по умолчанию берём файлы из репозитория)
    placeholders_ch0: UploadFile | None = File(
        None, description="[опционально] JSON плейсхолдеров для главы 0"
    ),
    template_docx_ch0: UploadFile | None = File(
        None, description="[опционально] DOCX шаблон для главы 0"
    ),
    table_placeholders_ch0: UploadFile | None = File(
        None, description="[опционально] JSON с табличными плейсхолдерами для главы 0"
    ),
    table_placeholders_ch1: UploadFile | None = File(
        None, description="[опционально] JSON с табличными плейсхолдерами для главы 1"
    ),
    table_placeholders_ch2: UploadFile | None = File(
        None, description="[опционально] JSON с табличными плейсхолдерами для главы 2"
    ),
    # Переопределения шаблонов сборки для глав 1 и 2
    template_docx_ch1: UploadFile | None = File(
        None, description="[опционально] DOCX шаблон для главы 1"
    ),
    template_docx_ch2: UploadFile | None = File(
        None, description="[опционально] DOCX шаблон для главы 2"
    ),
    max_workers: int | None = Form(
        8, description="[### для отладки ###] Число потоков для параллельного запуска моделей"
    ),
    collection_name: str = Form(uuid.uuid4().hex, description="[### для отладки ###] Имя коллекции"),
    test_mode: Annotated[TestMode, Query(include_in_schema=False)] = 'off',
):
    try:
        if project_parts_zip:
            await validate_zip(project_parts_zip)
        if placeholders_ch0:
            await validate_json(placeholders_ch0)
        if template_docx_ch0:
            await validate_docx(template_docx_ch0)
        if template_docx_ch1:
            await validate_docx(template_docx_ch1)
        if template_docx_ch2:
            await validate_docx(template_docx_ch2)
        if table_placeholders_ch0:
            await validate_json(table_placeholders_ch0)
        if table_placeholders_ch1:
            await validate_json(table_placeholders_ch1)
        if table_placeholders_ch2:
            await validate_json(table_placeholders_ch2)
    
        project_path = Path(__file__).parent
        ch0_dir = project_path / "src" / "ecology_chapters" / "chapter0"
        ch1_dir = project_path / "src" / "ecology_chapters" / "chapter1"
        ch2_dir = project_path / "src" / "ecology_chapters" / "chapter2"
    
        with tempfile.TemporaryDirectory(prefix="nc_ecology_all_") as tmp:
            tmp_dir = Path(tmp)
            input_dir = tmp_dir / "in"
            output_dir = tmp_dir / "out"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
    
            project_parts_dir: Path | None = None
            if project_parts_zip:
                project_parts_dir = tmp_dir / "project_parts"
                project_parts_dir.mkdir(parents=True, exist_ok=True)
                project_parts_zip_bytes = await project_parts_zip.read()
                _extract_project_parts_pdfs(project_parts_zip_bytes, project_parts_dir)
    
            # Глава 0: плейсхолдеры/шаблон по умолчанию из репозитория, можно переопределить загрузкой
            placeholders_ch0_path = ch0_dir / "placeholders.json"
            if placeholders_ch0:
                placeholders_ch0_path = input_dir / "chapter0_placeholders.json"
                placeholders_ch0_path.write_bytes(await placeholders_ch0.read())
    
            table_placeholders_ch0_path = ch0_dir / "table_placeholders.json"
            if table_placeholders_ch0:
                table_placeholders_ch0_path = input_dir / "chapter0_table_placeholders.json"
                table_placeholders_ch0_path.write_bytes(await table_placeholders_ch0.read())
    
            template_ch0_path = ch0_dir / "template.docx"
            if template_docx_ch0:
                template_ch0_path = input_dir / "chapter0_template.docx"
                template_ch0_path.write_bytes(await template_docx_ch0.read())
    
            ch0_out = output_dir / "chapter0"
            ch1_out = output_dir / "chapter1"
            ch2_out = output_dir / "chapter2"
    
            base_placeholders = run_main_base(
                template_docx_path=template_ch0_path,
                placeholders_path=placeholders_ch0_path,
                table_placeholders_path=table_placeholders_ch0_path,
                project_parts_path=project_parts_dir,
                output_path=ch0_out,
                collection_name=collection_name,
                verbose=False,
                test_mode=test_mode,
                save_db=1
            )
    
            def _resolve_chapter_table_placeholders_path(
                chapter_dir: Path,
                upload: UploadFile | None,
            ) -> Path | None:
                """ Добавляет базовые плейсхолдеры в файл table_placeholders главы X  """
                
                if upload:
                    table_placeholders = json.loads((upload.read()).decode("utf-8"))
                else:
                    default_path = chapter_dir / "table_placeholders.json"
                    if default_path.exists():
                        table_placeholders = json.loads(default_path.read_text(encoding="utf-8"))
                    else:
                        table_placeholders = {}
                
                table_placeholders.update(base_placeholders)
                
                if not table_placeholders:
                    return None
                
                out_path = input_dir / f"{chapter_dir.name}_table_placeholders.json"
                out_path.write_text(
                    json.dumps(table_placeholders, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return out_path
    
            table_placeholders_ch1_path = _resolve_chapter_table_placeholders_path(
                ch1_dir, table_placeholders_ch1
            )
            table_placeholders_ch2_path = _resolve_chapter_table_placeholders_path(
                ch2_dir, table_placeholders_ch2
            )
    
            template_ch1_path = ch1_dir / "template.docx"
            if template_docx_ch1:
                template_ch1_path = input_dir / "chapter1_template.docx"
                template_ch1_path.write_bytes(await template_docx_ch1.read())
    
            template_ch2_path = ch2_dir / "template.docx"
            if template_docx_ch2:
                template_ch2_path = input_dir / "chapter2_template.docx"
                template_ch2_path.write_bytes(await template_docx_ch2.read())
            
            logger.info("\nНАЧИНАЮ ФОРМИРОВАТЬ ГЛАВЫ 1 И 2\n")
            
            common_args = {
                "project_parts_path": project_parts_dir,
                "collection_name": collection_name,
                "verbose": False,
                "test_mode": test_mode,
                "max_workers": max_workers,
                "save_db": 1,
            }
            
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(
                        run_main,
                        template_docx_path=template_ch1_path,
                        table_placeholders_path=table_placeholders_ch1_path,
                        output_path=ch1_out,
                        chapter_module_path="src.ecology_chapters.chapter1",
                        **common_args,
                    ),
                    pool.submit(
                        run_main,
                        template_docx_path=template_ch2_path,
                        table_placeholders_path=table_placeholders_ch2_path,
                        output_path=ch2_out,
                        chapter_module_path="src.ecology_chapters.chapter2",
                        **common_args,
                    ),
                ]
                
                for future in futures:
                    future.result()
            
            logger.info("\nГЛАВЫ 1 И 2 СФОРМИРОВАНЫ\n")
            
            return _zip_output_dir(output_dir, "all_chapters.zip")
        
    finally:
        client = QdrantClient(url=cfg.QDRANT_URL)
        if client.collection_exists(collection_name) and is_valid_uuid4_hex(collection_name):
            client.delete_collection(collection_name)
            logger.info(
                f"collection <{collection_name}> name is valid uuid and was deleted"
            )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
