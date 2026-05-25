from typing import Optional
from pydantic import BaseModel, Field


class StructuredResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Краткий, точный (но полный) ответ на запрос пользователя. "
        "Любые вводные слова и повторение сути вопроса запрещены.",
    )
    explanation: Optional[str] = Field(
        None, description="Дополнительные пояснения или контекст, если необходимы"
    )
