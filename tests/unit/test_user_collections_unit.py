import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.mongo.db_model import QdrantCollection, User, parse_user
from src.mongo.user_collections import allocate_qdrant_collection


def test_parse_user_validates_collections():
    user = parse_user(
        {
            "cookie": "a" * 32,
            "qdrant_collections": [
                {"uuid": "b" * 32, "created_at": "2026-01-01T00:00:00+00:00"},
            ],
        }
    )
    assert isinstance(user, User)
    assert len(user.qdrant_collections) == 1
    assert isinstance(user.qdrant_collections[0], QdrantCollection)


def test_allocate_evicts_oldest_when_limit_reached():
    cookie = "a" * 32
    store = {
        "cookie": cookie,
        "qdrant_collections": [
            {"uuid": "1" * 32, "created_at": "2026-01-01T00:00:00+00:00"},
            {"uuid": "2" * 32, "created_at": "2026-01-02T00:00:00+00:00"},
            {"uuid": "3" * 32, "created_at": "2026-01-03T00:00:00+00:00"},
        ],
    }

    users = MagicMock()
    users.find_one = AsyncMock(return_value=store)
    users.update_one = AsyncMock()

    db = MagicMock()
    db.__getitem__.return_value = users

    async def _run():
        with (
            patch("src.mongo.user_collections.get_database", return_value=db),
            patch("src.mongo.user_collections._delete_qdrant_collection") as delete_mock,
        ):
            return await allocate_qdrant_collection(cookie), delete_mock

    new_id, delete_mock = asyncio.run(_run())

    assert len(new_id) == 32
    delete_mock.assert_called_once_with("1" * 32)
    assert users.update_one.await_count == 2

    push_update = users.update_one.await_args_list[1].args[1]
    pushed = push_update["$push"]["qdrant_collections"]
    validated = QdrantCollection.model_validate(pushed)
    assert validated.uuid == new_id
