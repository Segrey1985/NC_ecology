from typing import Optional, Literal, Any
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
        ..., description="Тип строительных работ. Допустимые значения: «строительство», «реконструкция», «техническое перевооружение»"
    )
    explanation: Optional[str] = Field(
        None, description="Дополнительные пояснения или контекст, если необходимы"
    )

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_type_of_work(cls, v: Any) -> str:
        if not isinstance(v, str):
            v = str(v)
        v_lower = v.lower().strip()
        # Нормализация вариантов
        if "перевооружение" in v_lower or "перевооружение" in v_lower:
            return "техническое перевооружение"
        if "реконструкц" in v_lower:
            return "реконструкция"
        # Всё остальное (строительство, новое строительство и т.д.)
        return "строительство"
