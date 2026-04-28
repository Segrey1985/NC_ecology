from argparse import ArgumentParser
from typing import Literal, Optional
from pydantic import BaseModel, Field

from config.config_file import cfg

ALLOWED_DISCIPLINES = tuple(cfg.DISCIPLINE_BY_NUMBER.values())
DisciplinesLiteral = Literal[*ALLOWED_DISCIPLINES]


class RelatedDisciplinesSearch(BaseModel):
    query: str = Field(
        ..., description="RAG запрос для поиска наиболее релевантных текстов"
    )
    # disciplines: list[DisciplinesLiteral] = Field(
    #     ...,
    #     description="Список разделов в которых будет поиск релевантных текстов. Использовать только при необходимости",
    # )
