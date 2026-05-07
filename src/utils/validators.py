import json
import io
from pathlib import Path
import zipfile
from posixpath import normpath as posix_normpath
import re
from fastapi.exceptions import HTTPException
from fastapi import UploadFile
from pydantic import BaseModel

from src.utils.logger import logger


async def validate_zip(file: UploadFile):
    # Быстрая проверка по заголовку (magic bytes) и расширению
    header = await file.read(4)
    await file.seek(0)

    if not (
        file.content_type in {"application/zip", "application/x-zip-compressed"}
        or Path(file.filename).suffix.lower() == ".zip"
        or header.startswith(b"PK")
    ):
        raise HTTPException(
            status_code=400, detail=f"{file.filename} не является ZIP-архивом"
        )

    data = await file.read()
    await file.seek(0)

    if not data:
        raise HTTPException(
            status_code=400, detail=f"{file.filename}: пустой ZIP-архив"
        )

    # Страховка от случайных гигантских загрузок/zip-bomb.
    # Если потребуется — вынесем лимиты в конфиг.
    max_zip_bytes = 200 * 1024 * 1024  # 200 MB
    if len(data) > max_zip_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: ZIP-архив слишком большой (>{max_zip_bytes} байт)",
        )

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            try:
                bad = zf.testzip()
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename}: повреждённый ZIP (ошибка чтения: {e})",
                ) from e

            if bad:
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename}: повреждённый ZIP (битый файл: {bad})",
                )

            infos = zf.infolist()
            if not infos:
                raise HTTPException(
                    status_code=400, detail=f"{file.filename}: ZIP-архив пуст"
                )

            # Ограничения на количество файлов и суммарный распакованный размер
            max_files = 5000
            if len(infos) > max_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename}: слишком много файлов в архиве (>{max_files})",
                )

            total_uncompressed = 0
            has_pdf = False

            for info in infos:
                name = info.filename.replace("\\", "/")

                # zip slip / абсолютные пути
                normalized = posix_normpath(name).lstrip("/")
                if (
                    normalized.startswith("..")
                    or name.startswith(("/", "\\"))
                    or re.match(r"^[A-Za-z]:", normalized) is not None
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"{file.filename}: небезопасный путь внутри ZIP: {info.filename}",
                    )

                # Зашифрованные ZIP'ы не поддерживаем (иначе упадём позже при распаковке/чтении)
                if info.flag_bits & 0x1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{file.filename}: зашифрованные ZIP-архивы не поддерживаются",
                    )

                total_uncompressed += int(getattr(info, "file_size", 0) or 0)
                if Path(normalized).suffix.lower() == ".pdf":
                    has_pdf = True

            max_uncompressed = 800 * 1024 * 1024  # 800 MB
            if total_uncompressed > max_uncompressed:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{file.filename}: слишком большой распакованный размер "
                        f"(>{max_uncompressed} байт)"
                    ),
                )

            if not has_pdf:
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename}: в архиве не найдено ни одного PDF",
                )

    except HTTPException:
        raise
    except zipfile.BadZipFile as e:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename} не является корректным ZIP-архивом",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: ошибка проверки ZIP ({type(e).__name__}: {e})",
        ) from e


async def validate_pdf(file: UploadFile):
    header = await file.read(5)
    await file.seek(0)

    if not (
        file.content_type == "application/pdf"
        or Path(file.filename).suffix.lower() == ".pdf"
        or header == b"%PDF-"
    ):
        raise HTTPException(
            status_code=400, detail=f"{file.filename} is not a valid PDF"
        )


async def validate_docx(file: UploadFile):
    header = await file.read(2)
    await file.seek(0)

    if not (
        file.content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or Path(file.filename).suffix.lower() == ".docx"
        or header == b"PK"
    ):
        raise HTTPException(status_code=400, detail="template_docx must be DOCX")


async def validate_json(file: UploadFile):
    try:
        content = await file.read()
        await file.seek(0)
        json.loads(content)
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"{file.filename} is not valid JSON"
        )


def validate_and_dump_json_str(obj: type[BaseModel], json_str: str) -> str:
    """
    Гарантирует валидный JSON-стринг, соответствующий `output_model`.
    Если не удаётся распарсить/провалидировать — возвращает "{}".
    """
    json_str = (json_str or "").strip()
    if not json_str:
        return "{}"

    # 1) Пробуем распарсить как есть
    try:
        data = json.loads(json_str)
    except Exception:
        # 2) Пытаемся вытащить объект между первой { и последней }
        try:
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return "{}"
            data = json.loads(json_str[start : end + 1])
        except Exception:
            return "{}"

    # 3) Pydantic-валидация
    try:
        validated = obj.model_validate(data)
        return validated.model_dump_json()
    except Exception:
        logger.exception("Pydantic validation failed for fallback JSON.")
        return "{}"