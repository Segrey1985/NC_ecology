import io
import json
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient

from config.config_file import TestMode, cfg
from main import main as run_main
from main_base import main as run_main_base
from src.utils.logger import logger
from src.utils.utils import is_valid_uuid4_hex
from src.utils.validators import validate_docx, validate_json, validate_zip

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
ALL_CHAPTER_SPECS = (CHAPTER0, CHAPTER1, CHAPTER2)


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


def delete_qdrant_collection_if_temp(collection_name: str) -> None:
    client = QdrantClient(url=cfg.QDRANT_URL)
    if client.collection_exists(collection_name) and is_valid_uuid4_hex(collection_name):
        client.delete_collection(collection_name)
        logger.info(
            f"collection <{collection_name}> name is valid uuid and was deleted"
        )


async def generate_chapter(
    *,
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
        if project_parts_zip:
            project_parts_zip_bytes = await project_parts_zip.read()

        collection_name = collection_name or uuid.uuid4().hex

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
                run_main_base(
                    template_docx_path=template_docx_path,
                    placeholders_path=placeholders_path,
                    table_placeholders_path=table_placeholders_path,
                    project_parts_zip=project_parts_zip_bytes,
                    output_path=output_dir,
                    collection_name=collection_name,
                    verbose=False,
                    test_mode="off",
                )
            else:
                if extract_base:
                    # Шаг 1: извлечь общие плейсхолдеры
                    logger.info("[extract_base] START")
                    base_placeholders: dict = run_main_base(
                        template_docx_path=CHAPTER0.default_template(),
                        placeholders_path=CHAPTER0.default_placeholders(),
                        table_placeholders_path=CHAPTER0.default_table_placeholders(),
                        project_parts_zip=project_parts_zip_bytes,
                        output_path=output_dir / "__debug__" / CHAPTER0.name,
                        collection_name=collection_name,
                        verbose=False,
                        test_mode="off",
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
                run_main(
                    template_docx_path=template_docx_path,
                    project_parts_zip=project_parts_zip_bytes,
                    table_placeholders_path=table_placeholders_path,
                    output_path=output_dir,
                    chapter_module_path=chapter_module_path,
                    collection_name=collection_name,
                    verbose=False,
                    test_mode="off",
                    max_workers=max_workers,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return zip_output_dir(output_dir, result_filename)


async def generate_all_chapters(
    *,
    project_parts_zip: UploadFile | None,
    collection_name: str | None = None,
    max_workers: int | None = None,
    test_mode: TestMode = "off",
    placeholders_ch0: UploadFile | None = None,
    template_docx_ch0: UploadFile | None = None,
    table_placeholders_ch0: UploadFile | None = None,
    table_placeholders_ch1: UploadFile | None = None,
    table_placeholders_ch2: UploadFile | None = None,
    template_docx_ch1: UploadFile | None = None,
    template_docx_ch2: UploadFile | None = None,
):
    collection_name = collection_name or uuid.uuid4().hex
    max_workers = CHAPTER1.default_max_workers if max_workers is None else max_workers

    try:
        if project_parts_zip:
            await validate_zip(project_parts_zip)
        for upload, validator in (
            (placeholders_ch0, validate_json),
            (table_placeholders_ch0, validate_json),
            (table_placeholders_ch1, validate_json),
            (table_placeholders_ch2, validate_json),
            (template_docx_ch0, validate_docx),
            (template_docx_ch1, validate_docx),
            (template_docx_ch2, validate_docx),
        ):
            if upload:
                await validator(upload)

        with tempfile.TemporaryDirectory(prefix="nc_ecology_all_") as tmp:
            tmp_dir = Path(tmp)
            input_dir = tmp_dir / "in"
            output_dir = tmp_dir / "out"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            project_parts_zip_bytes: bytes | None = None
            if project_parts_zip:
                project_parts_zip_bytes = await project_parts_zip.read()

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

            ch0_out = output_dir / CHAPTER0.name
            ch1_out = output_dir / CHAPTER1.name
            ch2_out = output_dir / CHAPTER2.name

            try:
                base_placeholders = run_main_base(
                    template_docx_path=template_ch0_path,
                    placeholders_path=placeholders_ch0_path,
                    table_placeholders_path=table_placeholders_ch0_path,
                    project_parts_zip=project_parts_zip_bytes,
                    output_path=ch0_out,
                    collection_name=collection_name,
                    verbose=False,
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

            logger.info("\nНАЧИНАЮ ФОРМИРОВАТЬ ГЛАВЫ 1 И 2\n")

            common_args = {
                "project_parts_zip": project_parts_zip_bytes,
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
                        chapter_module_path=CHAPTER1.module_path,
                        **common_args,
                    ),
                    pool.submit(
                        run_main,
                        template_docx_path=template_ch2_path,
                        table_placeholders_path=table_placeholders_ch2_path,
                        output_path=ch2_out,
                        chapter_module_path=CHAPTER2.module_path,
                        **common_args,
                    ),
                ]

                for future in futures:
                    try:
                        future.result()
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc

            logger.info("\nГЛАВЫ 1 И 2 СФОРМИРОВАНЫ\n")

            return zip_output_dir(output_dir, "all_chapters.zip")

    finally:
        delete_qdrant_collection_if_temp(collection_name)
