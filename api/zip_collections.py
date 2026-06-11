import hashlib
import re
import threading
import time

from fastapi import HTTPException
from qdrant_client import QdrantClient

from config.config_file import cfg
from src.utils.logger import logger

ZIP_COLLECTION_PREFIX = "zip_"
ZIP_HASH_LENGTH = 24
ZIP_TIMESTAMP_WIDTH = 20
MAX_ZIP_COLLECTIONS = 10

ZIP_COLLECTION_PATTERN = re.compile(
    rf"^{ZIP_COLLECTION_PREFIX}(\d{{{ZIP_TIMESTAMP_WIDTH}}})_([0-9a-f]{{{ZIP_HASH_LENGTH}}})$"
)

_resolve_lock = threading.Lock()


def zip_hash_from_bytes(zip_bytes: bytes) -> str:
    """Сформировать хэш из байтов"""
    return hashlib.sha256(zip_bytes).hexdigest()[:ZIP_HASH_LENGTH]


def make_zip_collection_name(zip_bytes: bytes, *, created_ns: int | None = None) -> str:
    """Имя новой коллекции: zip_{time_ns:020d}_{hash24}."""
    hash24 = zip_hash_from_bytes(zip_bytes)
    ts = created_ns if created_ns is not None else time.time_ns()
    return f"{ZIP_COLLECTION_PREFIX}{ts:0{ZIP_TIMESTAMP_WIDTH}d}_{hash24}"


def parse_zip_collection_name(name: str) -> tuple[int, str] | None:
    """Из имени коллекции получить tuple(время, имя)"""
    match = ZIP_COLLECTION_PATTERN.fullmatch(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def list_zip_collections(client: QdrantClient) -> list[str]:
    """Вернуть отсортированные коллекции по времени создания коллекции (с самой старой)"""
    names = [
        coll.name
        for coll in client.get_collections().collections
        if parse_zip_collection_name(coll.name) is not None
    ]
    return sorted(names, key=lambda n: parse_zip_collection_name(n)[0])


def find_zip_collection_by_hash(client: QdrantClient, hash24: str) -> str | None:
    """Найти имя коллекции по хэшу"""
    for coll in client.get_collections().collections:
        parsed = parse_zip_collection_name(coll.name)
        if parsed and parsed[1] == hash24:
            return coll.name
    return None


def _evict_oldest_zip_collection(client: QdrantClient) -> None:
    """Удалить самую старую коллекцию"""
    names = list_zip_collections(client)
    if not names:
        return
    oldest = names[0]
    if client.collection_exists(oldest):
        client.delete_collection(collection_name=oldest)
        logger.info(f"FIFO: удалена коллекция <{oldest}>")


def resolve_zip_collection_name(client: QdrantClient, zip_bytes: bytes) -> str:
    """
    Основная функция.
    Один zip → одна коллекция; при переполнении — FIFO (макс. MAX_ZIP_COLLECTIONS).
    """
    hash24 = zip_hash_from_bytes(zip_bytes)
    with _resolve_lock:
        existing = find_zip_collection_by_hash(client, hash24)
        if existing:
            logger.info(f"Переиспользуем zip-коллекцию <{existing}>")
            return existing

        while len(list_zip_collections(client)) >= MAX_ZIP_COLLECTIONS:
            _evict_oldest_zip_collection(client)

        new_name = make_zip_collection_name(zip_bytes)
        logger.info(f"Новая zip-коллекция <{new_name}>")
        return new_name


def resolve_collection_name(
    *,
    client: QdrantClient,
    collection_name: str | None,
    zip_bytes: bytes | None,
) -> str:
    """
    Основная функция для доступа из api_utils.py
    """
    if zip_bytes is not None:
        return resolve_zip_collection_name(client, zip_bytes)

    if not collection_name:
        raise HTTPException(
            status_code=400,
            detail="Укажите project_parts_zip или collection_name",
        )
    if not client.collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Коллекция «{collection_name}» не найдена",
        )
    return collection_name


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=cfg.QDRANT_URL)
