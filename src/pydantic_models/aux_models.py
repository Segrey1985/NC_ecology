from pydantic import BaseModel, Field


class AuxSchema(BaseModel):
    type_nominative_short: str = Field(
        ...,
        description="Краткое наименование объекта (одно слово) в именительном падеже. (например: котельная)",
    )
    type_genitive_short: str = Field(
        ...,
        description="Краткое наименование объекта (одно слово) в родительном падеже. (например: котельной)",
    )
