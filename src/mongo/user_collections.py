import uuid
from datetime import UTC, datetime

from qdrant_client import QdrantClient

from config.config_file import cfg
from src.mongo.db_model import QdrantCollection, parse_user
from src.mongo.mongo_client import get_database
from src.utils.logger import logger
from src.utils.utils import is_valid_uuid4_hex

MAX_QDRANT_COLLECTIONS_PER_USER = 3
USERS_COLLECTION = "users"


def _delete_qdrant_collection(collection_name: str) -> None:
    if not is_valid_uuid4_hex(collection_name):
        return

    client = QdrantClient(url=cfg.QDRANT_URL)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        logger.info(f"Qdrant collection <{collection_name}> was deleted")


async def allocate_qdrant_collection(cookie: str) -> str:
    """Выделить новую Qdrant-коллекцию для пользователя (не более MAX_QDRANT_COLLECTIONS_PER_USER на cookie)."""
    db = get_database()
    users = db[USERS_COLLECTION]

    user_doc = await users.find_one({"cookie": cookie})
    user = parse_user(user_doc)
    collections: list[QdrantCollection] = list(user.qdrant_collections) if user else []

    while len(collections) >= MAX_QDRANT_COLLECTIONS_PER_USER:
        oldest = min(collections, key=lambda item: item.created_at)
        oldest_uuid = oldest.uuid
        _delete_qdrant_collection(oldest_uuid)
        await users.update_one(
            {"cookie": cookie},
            {"$pull": {"qdrant_collections": {"uuid": oldest_uuid}}},
        )
        collections = [item for item in collections if item.uuid != oldest_uuid]
        logger.info(
            f"Qdrant overflow for cookie <{cookie}>: evicted collection <{oldest_uuid}>"
        )
    
    new_id = uuid.uuid4().hex
    new_collection = QdrantCollection(
        uuid=new_id,
        created_at=datetime.now(UTC).isoformat(),
    )
    await users.update_one(
        {"cookie": cookie},
        {
            "$push": {"qdrant_collections": new_collection.model_dump()},
            "$setOnInsert": {"cookie": cookie},
        },
        upsert=True,
    )
    return new_id
