from typing import Optional, Literal
from pydantic import BaseModel, Field


# основная модель

class StructuredResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Краткий, точный (но полный) ответ на запрос пользователя. "
        "Любые вводные слова и повторение сути вопроса запрещены.",
    )
    explanation: Optional[str] = Field(
        None, description="Дополнительные пояснения или контекст, если необходимы"
    )


# дополнительные модели к которым могут обращаться placeholders.json

class TypeOfWork(BaseModel):
    answer: Literal["строительство", "реконструкция", "техническое перевооружение"] = Field(
        ..., description="Тип строительных работ (например, «строительство», «реконструкция», «техническое перевооружение»)"
    )
    explanation: Optional[str] = Field(
        None, description="Дополнительные пояснения или контекст, если необходимы"
    )
