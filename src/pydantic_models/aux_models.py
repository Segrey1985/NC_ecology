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
    type_of_work_prepositional: str = Field(
        ...,
        description="ТИП_РАБОТ в предложном падеже"
    )
    type_of_work_genitive: str = Field(
        ...,
        description="ТИП_РАБОТ в родительном падеже"
    )
