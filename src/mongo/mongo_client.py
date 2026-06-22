import asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config.config_file import cfg
from src.utils.logger import logger

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None

DB_NAME: str = "ecology"


async def connect_mongo(
    uri: str | None = None,
    *,
    max_attempts: int = 5,
    delay_sec: float = 3.0,
) -> AsyncIOMotorDatabase:
    """Подключиться к MongoDB и вернуть объект базы данных."""
    global _client, _db

    if _db is not None:
        return _db

    mongo_uri = uri or cfg.MONGO_URI
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        client = AsyncIOMotorClient(mongo_uri)
        try:
            await client.admin.command("ping")
            _client = client
            _db = _client.get_database(DB_NAME)
            logger.info(f"MongoDB connected: {mongo_uri}")
            return _db
        except Exception as exc:
            last_error = exc
            client.close()
            if attempt == max_attempts:
                break
            logger.warning(
                f"MongoDB connect attempt {attempt}/{max_attempts} failed: {exc}"
            )
            await asyncio.sleep(delay_sec)

    assert last_error is not None
    raise last_error


async def disconnect_mongo() -> None:
    """Закрыть соединение с MongoDB."""
    global _client, _db

    if _client is None:
        return

    _client.close()
    _client = None
    _db = None
    logger.info("MongoDB disconnected")


def get_database() -> AsyncIOMotorDatabase:
    """Вернуть текущее подключение к базе (после connect_mongo)."""
    if _db is None:
        raise RuntimeError("MongoDB is not connected. Call connect_mongo() first.")
    return _db


async def ping_mongo() -> bool:
    """Проверить доступность MongoDB."""
    if _client is None:
        return False

    try:
        await _client.admin.command("ping")
        return True
    except Exception as exc:
        logger.warning(f"MongoDB ping failed: {exc}")
        return False
