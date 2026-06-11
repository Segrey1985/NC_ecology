from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from api import zip_collections as zc


def _mock_client(collection_names: list[str]) -> MagicMock:
    names = list(collection_names)
    client = MagicMock()

    def _get_collections():
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names])

    def _collection_exists(name: str) -> bool:
        return name in names

    def _delete_collection(*, collection_name: str) -> None:
        if collection_name in names:
            names.remove(collection_name)

    client.get_collections.side_effect = _get_collections
    client.collection_exists.side_effect = _collection_exists
    client.delete_collection.side_effect = _delete_collection
    return client


def test_zip_hash_stable():
    data = b"PK\x03\x04fake zip content"
    assert zc.zip_hash_from_bytes(data) == zc.zip_hash_from_bytes(data)
    assert len(zc.zip_hash_from_bytes(data)) == zc.ZIP_HASH_LENGTH


def test_make_and_parse_collection_name():
    zip_bytes = b"test-zip-bytes"
    name = zc.make_zip_collection_name(zip_bytes, created_ns=123)
    assert name == f"zip_{123:020d}_{zc.zip_hash_from_bytes(zip_bytes)}"
    parsed = zc.parse_zip_collection_name(name)
    assert parsed == (123, zc.zip_hash_from_bytes(zip_bytes))


def test_find_zip_collection_by_hash():
    zip_bytes = b"archive-a"
    hash24 = zc.zip_hash_from_bytes(zip_bytes)
    existing = f"zip_{1:020d}_{hash24}"
    client = _mock_client([existing, "zip_00000000000000000002_bbbb"])
    assert zc.find_zip_collection_by_hash(client, hash24) == existing
    assert zc.find_zip_collection_by_hash(client, "cccc") is None


def test_resolve_zip_collection_reuses_existing():
    zip_bytes = b"same-archive"
    hash24 = zc.zip_hash_from_bytes(zip_bytes)
    existing = f"zip_{100:020d}_{hash24}"
    client = _mock_client([existing])

    assert zc.resolve_zip_collection_name(client, zip_bytes) == existing
    client.delete_collection.assert_not_called()


def test_resolve_zip_collection_fifo_eviction():
    zip_bytes = b"new-archive"
    hash24 = zc.zip_hash_from_bytes(zip_bytes)
    old_collections = [f"zip_{i:020d}_{'a' * 24}" for i in range(10)]
    client = _mock_client(old_collections)

    name = zc.resolve_zip_collection_name(client, zip_bytes)

    client.delete_collection.assert_called_once_with(collection_name=old_collections[0])
    assert name.endswith(f"_{hash24}")
    assert zc.parse_zip_collection_name(name) is not None


def test_resolve_collection_name_requires_zip_or_name():
    client = _mock_client([])
    with pytest.raises(HTTPException) as exc:
        zc.resolve_collection_name(client=client, collection_name=None, zip_bytes=None)
    assert exc.value.status_code == 400


def test_resolve_collection_name_missing_collection():
    client = _mock_client([])
    with pytest.raises(HTTPException) as exc:
        zc.resolve_collection_name(
            client=client,
            collection_name="missing_collection",
            zip_bytes=None,
        )
    assert exc.value.status_code == 404


def test_resolve_collection_name_explicit_existing():
    client = _mock_client(["my_manual_collection"])
    name = zc.resolve_collection_name(
        client=client,
        collection_name="my_manual_collection",
        zip_bytes=None,
    )
    assert name == "my_manual_collection"


def test_resolve_collection_name_zip_ignores_explicit_name():
    zip_bytes = b"zip-wins"
    hash24 = zc.zip_hash_from_bytes(zip_bytes)
    client = _mock_client([])

    name = zc.resolve_collection_name(
        client=client,
        collection_name="should_be_ignored",
        zip_bytes=zip_bytes,
    )

    assert name.endswith(f"_{hash24}")
