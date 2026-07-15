from typing import Optional, Literal, Union, Any
from pydantic import BaseModel, Field, field_validator

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

    @field_validator("answer", mode="before")
    @classmethod
    def coerce_answer_to_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)

# дополнительные модели к которым могут обращаться placeholders.json
class TypeOfWork(BaseModel):
    answer: Literal["строительство", "реконструкция", "техническое перевооружение"] = Field(
        ..., description="Тип строительных работ (например, «строительство», «реконструкция», «техническое перевооружение»)"
    )
    explanation: Optional[str] = Field(
        None, description="Дополнительные пояснения или контекст, если необходимы"
    )
