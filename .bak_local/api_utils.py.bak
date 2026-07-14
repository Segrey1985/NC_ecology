import asyncio
import io
import json
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, UploadFile, Request
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient

from config.config_file import TestMode, cfg
from main import main as run_main
from main_base import main as run_main_base
from src.utils.logger import logger
from src.utils.utils import is_valid_uuid4_hex, file_hash
from src.utils.validators import (
    validate_docx,
    validate_json,
    validate_zip,
    normalize_archive_to_zip,
)
from src.mongo.user_collections import allocate_qdrant_collection, find_qdrant_collection_by_hash


ECOLOGY_CHAPTERS_ROOT = Path(__file__).resolve().parent.parent / "src" / "ecology_chapters"


@dataclass(frozen=True)
class ChapterSpec:
    name: str
    pipeline: Literal["base", "chapter"]
    module_path: str | None = None
    result_filename: str = "result.zip"
    default_max_workers: int = 8
    default_extract_base: bool = True
    
    
    @property
    def directory(self) -> Path:
        return ECOLOGY_CHAPTERS_ROOT / self.name
    
    
    def default_placeholders(self) -> Path:
        return self.directory / "placeholders.json"
    
    
    def default_template(self) -> Path:
        return self.directory / "template.docx"
    
    
    def default_table_placeholders(self) -> Path:
        return self.directory / "table_placeholders.json"


CHAPTER0 = ChapterSpec(
    name="chapter0",
    pipeline="base",
    result_filename="chapter0.zip",
)
CHAPTER1 = ChapterSpec(
    name="chapter1",
    pipeline="chapter",
    module_path="src.ecology_chapters.chapter1",
    result_filename="chapter1.zip",
)
CHAPTER2 = ChapterSpec(
    name="chapter2",
    pipeline="chapter",
    module_path="src.ecology_chapters.chapter2",
    result_filename="chapter2.zip",
)
CHAPTER6 = ChapterSpec(
    name="chapter6",
    pipeline="chapter",
    module_path="src.ecology_chapters.chapter6",
    result_filename="chapter6.zip",
)
ALL_CHAPTER_SPECS = (CHAPTER0, CHAPTER1, CHAPTER2, CHAPTER6)


async def write_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await upload.read())
    return dest


async def resolve_path(
    upload: UploadFile | None,
    default_path: Path,
    temp_path: Path | None = None,
) -> Path:
    if upload is not None:
        if temp_path is None:
            raise ValueError("temp_path is required when upload is provided")
        return await write_upload(upload, temp_path)
    return default_path


async def resolve_table_placeholders_path(
    upload: UploadFile | None,
    spec: ChapterSpec,
    input_dir: Path,
    base_placeholders: dict | None = None,
    source_path: Path | None = None,
) -> Path | None:
    """ Слить table_plh-s + base_plh-s и сохранить в 'input_dir/chapterX_table_placeholders.json' """
    
    if source_path is not None and source_path.exists():
        data = json.loads(source_path.read_text(encoding="utf-8"))
    elif upload is not None:
        data = json.loads((await upload.read()).decode("utf-8"))
    else:
        default_path = spec.default_table_placeholders()
        if default_path.exists():
            data = json.loads(default_path.read_text(encoding="utf-8"))
        else:
            data = {}
    
    if base_placeholders:
        data.update(base_placeholders)
    
    if not data:
        return None
    
    out_path = input_dir / f"{spec.name}_table_placeholders.json"
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def zip_output_dir(output_dir: Path, result_filename: str) -> StreamingResponse:
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


async def resolve_collection_name(
    *,
    request: Request | None,
    collection_name: str | None,
    zip_hash: str | None,
    zip_name: str | None = None,
) -> str:
    # Имя коллекции передано явно (api_debug) - используем его
    if collection_name is not None:
        return collection_name
    # есть request и zip_hash
    if request is not None and zip_hash is not None:
        existing = await find_qdrant_collection_by_hash(
            session_cookie=request.state.session_cookie,
            zip_hash=zip_hash,
        )
        # используем существующую
        if existing:
            return existing
        # выделяем новую коллекцию
        return await allocate_qdrant_collection(
            cookie=request.state.session_cookie,
            zip_hash=zip_hash,
            zip_name=zip_name,
        )
    # Генерируем случайное uuid имя (api_debug)
    return uuid.uuid4().hex


def delete_qdrant_collection_if_temp(collection_name: str) -> None:
    """Удаляет qdrant-коллекцию если она существует и является валидным uuid"""
    client = QdrantClient(url=cfg.QDRANT_URL)
    if client.collection_exists(collection_name) and is_valid_uuid4_hex(collection_name):
        client.delete_collection(collection_name)
        logger.info(
            f"collection <{collection_name}> name is valid uuid and was deleted"
        )


def _run_chapter_pipelines_in_threads(
    *,
    chapter_jobs: list[dict],
    common_args: dict,
) -> None:
    with ThreadPoolExecutor(max_workers=max(4, len(chapter_jobs))) as pool:
        futures = [
            pool.submit(
                run_main,
                template_docx_path=job["template_docx_path"],
                table_placeholders_path=job["table_placeholders_path"],
                output_path=job["output_path"],
                chapter_module_path=job["chapter_module_path"],
                **common_args,
            )
            for job in chapter_jobs
        ]

        for future in futures:
            future.result()


async def generate_chapter(
    *,
    request: Request | None = None,
    spec: ChapterSpec,
    template_docx: UploadFile | None = None,
    project_parts_zip: UploadFile | None = None,
    collection_name: str | None = None,
    placeholders: UploadFile | None = None,
    max_workers: int | None = None,
    table_placeholders: UploadFile | None = None,
    extract_base: bool | None = None,
):
    pipeline = spec.pipeline
    chapter_module_path = spec.module_path
    result_filename = spec.result_filename
    max_workers = spec.default_max_workers if max_workers is None else max_workers
    extract_base = spec.default_extract_base if extract_base is None else extract_base

    # Валидация загруженных файлов до создания временной директории.
    if placeholders:
        await validate_json(placeholders)
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
        
        # --- Разрешение путей к входным файлам ---
        # resolve_path: upload → запись во временный файл, иначе — дефолт из spec.
        
        placeholders_path: Path | None = None
        if pipeline == "base":
            placeholders_path = await resolve_path(
                placeholders,
                spec.default_placeholders(),
                input_dir / "placeholders.json",
            )
            if not placeholders_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail="Не заданы placeholders и отсутствует файл по умолчанию",
                )
        
        template_docx_path = await resolve_path(
            template_docx,
            spec.default_template(),
            input_dir / "template.docx",
        )
        if not template_docx_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Не задан template_docx и отсутствует файл по умолчанию",
            )
        
        table_placeholders_path: Path | None = None
        if table_placeholders:
            # Upload уже прочитан здесь — повторно читать UploadFile нельзя.
            table_placeholders_path = await resolve_path(
                table_placeholders,
                spec.default_table_placeholders(),
                input_dir / "table_placeholders.json",
            )
        elif spec.default_table_placeholders().exists():
            table_placeholders_path = spec.default_table_placeholders()
        
        project_parts_zip_bytes: bytes | None = None
        zip_hash: str | None = None
        zip_name: str | None = None
        
        if project_parts_zip is not None:
            # Нормализуем архив (RAR → ZIP при необходимости) перед пайплайном
            project_parts_zip_bytes = await normalize_archive_to_zip(project_parts_zip)
            zip_hash = file_hash(project_parts_zip_bytes)
            zip_name = project_parts_zip.filename
        
        collection_name = await resolve_collection_name(
            request=request,
            collection_name=collection_name,
            zip_hash=zip_hash,
            zip_name=zip_name
        )
        
        logger.info(f"{pipeline=}")
        logger.info(f"{template_docx_path=}")
        logger.info(f"{placeholders_path=}")
        logger.info(f"{table_placeholders_path=}")
        logger.info(f"{project_parts_zip_bytes is not None=}")
        logger.info(f"{collection_name=}")
        logger.info(f"{chapter_module_path=}")
        logger.info(f"{max_workers=}")
        logger.info(f"{output_dir=}")
        
        # --- Запуск пайплайна "base" или "chapter" ---
        try:
            if pipeline == "base":
                await asyncio.to_thread(
                    run_main_base,
                    template_docx_path=template_docx_path,
                    placeholders_path=placeholders_path,
                    table_placeholders_path=table_placeholders_path,
                    project_parts_zip=project_parts_zip_bytes,
                    output_path=output_dir,
                    collection_name=collection_name,
                    verbose=True,
                    test_mode="off",
                    save_db=True if request else False,
                )
            else:
                if extract_base:
                    # Шаг 1: извлечь общие плейсхолдеры
                    logger.info("[extract_base] START")
                    base_placeholders: dict = await asyncio.to_thread(
                        run_main_base,
                        template_docx_path=CHAPTER0.default_template(),
                        placeholders_path=CHAPTER0.default_placeholders(),
                        table_placeholders_path=CHAPTER0.default_table_placeholders(),
                        project_parts_zip=project_parts_zip_bytes,
                        output_path=output_dir / "__debug__" / CHAPTER0.name,
                        collection_name=collection_name,
                        verbose=True,
                        test_mode="off",
                        save_db=1,
                    )
                    logger.info(f"[extract_base] {base_placeholders.keys()=}")
                    
                    # Шаг 2: слить base_placeholders с табличными плейсхолдерами главы.
                    # Если table_placeholders_path уже есть — читаем с диска (source_path),
                    # а не из UploadFile
                    table_placeholders_path = await resolve_table_placeholders_path(
                        None if table_placeholders_path else table_placeholders,
                        spec,
                        input_dir,
                        base_placeholders,
                        source_path=table_placeholders_path,
                    )
                    logger.info(f"[extract_base] {table_placeholders_path=}")
                    logger.info("[extract_base] END")
                
                # Шаг 3 (или единственный шаг без extract_base): генерация главы.
                await asyncio.to_thread(
                    run_main,
                    template_docx_path=template_docx_path,
                    project_parts_zip=project_parts_zip_bytes,
                    table_placeholders_path=table_placeholders_path,
                    output_path=output_dir,
                    chapter_module_path=chapter_module_path,
                    collection_name=collection_name,
                    verbose=True,
                    test_mode="off",
                    max_workers=max_workers,
                    save_db=True if request else False,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if not request:
                delete_qdrant_collection_if_temp(collection_name)
        return zip_output_dir(output_dir, result_filename)


async def generate_all_chapters(
    *,
    request: Request | None = None,
    project_parts_zip: UploadFile | None,
    collection_name: str | None = None,
    max_workers: int | None = None,
    test_mode: TestMode = "off",
    placeholders_ch0: UploadFile | None = None,
    template_docx_ch0: UploadFile | None = None,
    table_placeholders_ch0: UploadFile | None = None,
    table_placeholders_ch1: UploadFile | None = None,
    table_placeholders_ch2: UploadFile | None = None,
    table_placeholders_ch6: UploadFile | None = None,
    template_docx_ch1: UploadFile | None = None,
    template_docx_ch2: UploadFile | None = None,
    template_docx_ch6: UploadFile | None = None,
):
    max_workers = CHAPTER1.default_max_workers if max_workers is None else max_workers

    try:
        if project_parts_zip:
            await validate_zip(project_parts_zip)
        for upload, validator in (
                (placeholders_ch0, validate_json),
                (table_placeholders_ch0, validate_json),
                (table_placeholders_ch1, validate_json),
                (table_placeholders_ch2, validate_json),
                (table_placeholders_ch6, validate_json),
                (template_docx_ch0, validate_docx),
                (template_docx_ch1, validate_docx),
                (template_docx_ch2, validate_docx),
                (template_docx_ch6, validate_docx),
        ):
            if upload:
                await validator(upload)
        
        with tempfile.TemporaryDirectory(prefix="nc_ecology_all_") as tmp:
            tmp_dir = Path(tmp)
            input_dir = tmp_dir / "in"
            output_dir = tmp_dir / "out"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            ch0_out = output_dir / CHAPTER0.name
            ch1_out = output_dir / CHAPTER1.name
            ch2_out = output_dir / CHAPTER2.name
            ch6_out = output_dir / CHAPTER6.name
            
            placeholders_ch0_path = await resolve_path(
                placeholders_ch0,
                CHAPTER0.default_placeholders(),
                input_dir / "chapter0_placeholders.json",
            )
            table_placeholders_ch0_path = await resolve_path(
                table_placeholders_ch0,
                CHAPTER0.default_table_placeholders(),
                input_dir / "chapter0_table_placeholders.json",
            )
            template_ch0_path = await resolve_path(
                template_docx_ch0,
                CHAPTER0.default_template(),
                input_dir / "chapter0_template.docx",
            )
            
            project_parts_zip_bytes: bytes | None = None
            zip_hash: str | None = None
            
            if project_parts_zip is not None:
                # Нормализуем архив (RAR → ZIP при необходимости) перед пайплайном
                project_parts_zip_bytes = await normalize_archive_to_zip(project_parts_zip)
                zip_hash = file_hash(project_parts_zip_bytes)
            
            collection_name = await resolve_collection_name(
                request=request,
                collection_name=collection_name,
                zip_hash=zip_hash,
            )
            
            try:
                base_placeholders = await asyncio.to_thread(
                    run_main_base,
                    template_docx_path=template_ch0_path,
                    placeholders_path=placeholders_ch0_path,
                    table_placeholders_path=table_placeholders_ch0_path,
                    project_parts_zip=project_parts_zip_bytes,
                    output_path=ch0_out,
                    collection_name=collection_name,
                    verbose=True,
                    test_mode=test_mode,
                    save_db=1,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            
            table_placeholders_ch1_path = await resolve_table_placeholders_path(
                table_placeholders_ch1,
                CHAPTER1,
                input_dir,
                base_placeholders,
            )
            table_placeholders_ch2_path = await resolve_table_placeholders_path(
                table_placeholders_ch2,
                CHAPTER2,
                input_dir,
                base_placeholders,
            )
            table_placeholders_ch6_path = await resolve_table_placeholders_path(
                table_placeholders_ch6,
                CHAPTER6,
                input_dir,
                base_placeholders,
            )
            
            template_ch1_path = await resolve_path(
                template_docx_ch1,
                CHAPTER1.default_template(),
                input_dir / "chapter1_template.docx",
            )
            template_ch2_path = await resolve_path(
                template_docx_ch2,
                CHAPTER2.default_template(),
                input_dir / "chapter2_template.docx",
            )
            template_ch6_path = await resolve_path(
                template_docx_ch6,
                CHAPTER6.default_template(),
                input_dir / "chapter6_template.docx",
            )
            
            logger.info("\nНАЧИНАЮ ФОРМИРОВАТЬ ГЛАВЫ 1, 2 И 6\n")
            
            common_args = {
                "project_parts_zip": project_parts_zip_bytes,
                "collection_name": collection_name,
                "verbose": True,
                "test_mode": test_mode,
                "max_workers": max_workers,
                "save_db": 1,
            }
            
            chapter_jobs = [
                {
                    "template_docx_path": template_ch1_path,
                    "table_placeholders_path": table_placeholders_ch1_path,
                    "output_path": ch1_out,
                    "chapter_module_path": CHAPTER1.module_path,
                },
                {
                    "template_docx_path": template_ch2_path,
                    "table_placeholders_path": table_placeholders_ch2_path,
                    "output_path": ch2_out,
                    "chapter_module_path": CHAPTER2.module_path,
                },
                {
                    "template_docx_path": template_ch6_path,
                    "table_placeholders_path": table_placeholders_ch6_path,
                    "output_path": ch6_out,
                    "chapter_module_path": CHAPTER6.module_path,
                },
            ]
            
            try:
                await asyncio.to_thread(
                    _run_chapter_pipelines_in_threads,
                    chapter_jobs=chapter_jobs,
                    common_args=common_args,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            
            logger.info("\nГЛАВЫ 1, 2 И 6 СФОРМИРОВАНЫ\n")
            
            return zip_output_dir(output_dir, "all_chapters.zip")
    
    finally:
        if not request:
            delete_qdrant_collection_if_temp(collection_name)
