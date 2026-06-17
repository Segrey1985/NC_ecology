from pydantic import BaseModel, Field


class AuxSchema(BaseModel):
    name_genitive: str = Field(
        ...,
        description="НАИМЕНОВАНИЕ_ПРОЕКТА в родительном падеже. (например: строительства автоматизированной газовой котельной)"
    )
    region: str = Field(
        ...,
        description="Область или край, где находится объект в именительном падеже."
    )
    region_genitive: str = Field(
        ...,
        description="Область или край, где находится объект в родительном падеже."
    )
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
    purpose_of_object_genitive: str = Field(
        ...,
        description="НАЗНАЧЕНИЕ_ОБЪЕКТА в родительном падеже"
    )
