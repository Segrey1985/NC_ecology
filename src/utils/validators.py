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

try:
    import rarfile  # опциональная зависимость для поддержки RAR

    _RARFILE_AVAILABLE = True
except Exception:  # библиотека может быть не установлена
    rarfile = None
    _RARFILE_AVAILABLE = False

# Magic bytes для определения типа архива
_ZIP_MAGIC = b"PK\x03\x04"
_ZIP_EMPTY_MAGIC = b"PK\x05\x06"  # пустой zip
_RAR4_MAGIC = b"Rar!\x1a\x07\x00"
_RAR5_MAGIC = b"Rar!\x1a\x07\x01\x00"

MAX_ARCHIVE_BYTES = 200 * 1024 * 1024  # 200 MB — лимит на размер архива
MAX_UNCOMPRESSED_BYTES = 800 * 1024 * 1024  # 800 MB — лимит на распаковку
MAX_FILES = 5000


def detect_archive_type(data: bytes) -> str | None:
    """Определяет тип архива по сигнатуре: 'zip', 'rar' или None."""
    if data.startswith(_ZIP_MAGIC) or data.startswith(_ZIP_EMPTY_MAGIC):
        return "zip"
    if data.startswith(_RAR4_MAGIC) or data.startswith(_RAR5_MAGIC):
        return "rar"
    return None


def _is_safe_member_path(name: str) -> bool:
    """Проверяет путь внутри архива от zip-slip / абсолютных путей."""
    name = name.replace("\\", "/")
    normalized = posix_normpath(name).lstrip("/")
    if (
        normalized.startswith("..")
        or name.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", normalized) is not None
    ):
        return False
    return True


def rar_bytes_to_zip_bytes(data: bytes, filename: str = "archive.rar") -> bytes:
    """Конвертирует RAR-архив (байты) в ZIP-архив (байты).

    Позволяет остальному пайплайну работать только с ZIP, не зная о RAR.
    Использует библиотеку rarfile (под капотом — системная утилита unrar).
    """
    if not _RARFILE_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{filename}: поддержка RAR недоступна (не установлены "
                "библиотека rarfile и/или утилита unrar)"
            ),
        )

    try:
        rf = rarfile.RarFile(io.BytesIO(data))
    except rarfile.NeedFirstVolume as e:
        raise HTTPException(
            status_code=400,
            detail=f"{filename}: это не первый том многотомного RAR-архива",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"{filename}: повреждённый или некорректный RAR ({type(e).__name__}: {e})",
        ) from e

    try:
        if rf.needs_password():
            raise HTTPException(
                status_code=400,
                detail=f"{filename}: зашифрованные RAR-архивы не поддерживаются",
            )

        infos = rf.infolist()
        if not infos:
            raise HTTPException(
                status_code=400, detail=f"{filename}: RAR-архив пуст"
            )
        if len(infos) > MAX_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"{filename}: слишком много файлов в архиве (>{MAX_FILES})",
            )

        total_uncompressed = 0
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for info in infos:
                name = info.filename
                if not _is_safe_member_path(name):
                    raise HTTPException(
                        status_code=400,
                        detail=f"{filename}: небезопасный путь внутри RAR: {name}",
                    )
                if getattr(info, "isdir", lambda: False)():
                    continue
                total_uncompressed += int(getattr(info, "file_size", 0) or 0)
                if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"{filename}: слишком большой распакованный размер "
                            f"(>{MAX_UNCOMPRESSED_BYTES} байт)"
                        ),
                    )
                with rf.open(info) as src:
                    zf.writestr(name.replace("\\", "/"), src.read())

        zip_buf.seek(0)
        logger.info(
            f"{filename}: RAR успешно сконвертирован в ZIP "
            f"({len(infos)} записей, {total_uncompressed} байт)"
        )
        return zip_buf.getvalue()
    finally:
        try:
            rf.close()
        except Exception:
            pass


async def normalize_archive_to_zip(file: UploadFile) -> bytes:
    """Читает загруженный архив и возвращает его байты в формате ZIP.

    Если архив уже ZIP — возвращает как есть; если RAR — конвертирует.
    Используется в API перед передачей архива в пайплайн.
    """
    data = await file.read()
    await file.seek(0)
    if not data:
        raise HTTPException(
            status_code=400, detail=f"{file.filename}: пустой архив"
        )
    if len(data) > MAX_ARCHIVE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: архив слишком большой (>{MAX_ARCHIVE_BYTES} байт)",
        )

    kind = detect_archive_type(data)
    if kind == "rar" or (kind is None and Path(file.filename or "").suffix.lower() == ".rar"):
        return rar_bytes_to_zip_bytes(data, file.filename or "archive.rar")
    # По умолчанию считаем ZIP — возвращаем как есть
    return data


async def validate_zip(file: UploadFile):
    """Валидирует загруженный архив (ZIP или RAR).

    RAR предварительно конвертируется в ZIP в памяти, после чего проходит
    те же проверки, что и обычный ZIP (наличие PDF, лимиты, zip-slip).
    """
    # Быстрая проверка по заголовку (magic bytes) и расширению
    header = await file.read(8)
    await file.seek(0)

    suffix = Path(file.filename or "").suffix.lower()
    is_zip_like = (
        file.content_type in {"application/zip", "application/x-zip-compressed"}
        or suffix == ".zip"
        or header.startswith(b"PK")
    )
    is_rar_like = (
        file.content_type in {"application/x-rar-compressed", "application/vnd.rar"}
        or suffix == ".rar"
        or header.startswith(_RAR4_MAGIC)
        or header.startswith(_RAR5_MAGIC)
    )

    if not (is_zip_like or is_rar_like):
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename} не является ZIP- или RAR-архивом",
        )

    raw = await file.read()
    await file.seek(0)

    if not raw:
        raise HTTPException(
            status_code=400, detail=f"{file.filename}: пустой архив"
        )

    max_zip_bytes = MAX_ARCHIVE_BYTES  # 200 MB
    if len(raw) > max_zip_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: архив слишком большой (>{max_zip_bytes} байт)",
        )

    # Если это RAR — конвертируем в ZIP и дальше валидируем как ZIP
    if detect_archive_type(raw) == "rar" or (detect_archive_type(raw) is None and is_rar_like):
        data = rar_bytes_to_zip_bytes(raw, file.filename or "archive.rar")
    else:
        data = raw

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