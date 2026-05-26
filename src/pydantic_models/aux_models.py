from pydantic import BaseModel, Field


class AuxSchema(BaseModel):
    name_nominative_short: str = Field(
        ...,
        description="Краткое наименование объекта (одно слово) в именительном падеже. (например: котельная)",
    )
    name_genitive_short: str = Field(
        ...,
        description="Краткое наименование объекта (одно слово) в родительном падеже. (например: котельной)",
    )
