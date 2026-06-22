from pydantic import BaseModel, Field


class QdrantCollection(BaseModel):
    uuid: str
    created_at: str
    zip_hash: str
    zip_name: str | None = Field(default='untitled')


class User(BaseModel):
    cookie: str
    qdrant_collections: list[QdrantCollection] = Field(default_factory=list)


def parse_user(document: dict | None) -> User | None:
    if document is None:
        return None
    return User.model_validate(document)

