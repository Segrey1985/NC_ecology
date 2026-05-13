from __future__ import annotations
from typing import Type
from pydantic import BaseModel

from config.config_file import cfg

from .models import (
    Architecture,
    Boilers,
    Facility,
    Fuel,
    GeneralPlan,
    HeatSupply,
    LandPlot,
    NearestObjects,
    Ownership,
    PowerSupply,
    Pumps,
    SanitaryZone,
    Structures,
    Surroundings,
    UtilityNetworks,
    Ventilation,
    WaterTreatment,
)
from src.utils.utils import iter_models_from_module


MODEL_TO_PART_NAMES: dict[Type[BaseModel], list[str]] = {
    Facility: [],
    LandPlot: ["ПЗУ"],
    SanitaryZone: ["ПЗУ"],
    Structures: ["ПЗУ", "АР"],
    Ownership: ["ПЗУ"],
    Surroundings: ["ПЗУ"],
    GeneralPlan: ["ПЗУ"],
    NearestObjects: ["ПЗУ"],
    Architecture: ["АР", "КР"],
    HeatSupply: ["ТМ", "ТП"],
    Boilers: ["ТМ", "ТП"],
    Pumps: ["Система водоснабжения", "Система водоотведения", "ТМ", "ТП"],
    Fuel: ["ТМ", "ТП"],
    PowerSupply: ["Система электроснабжения"],
    WaterTreatment: ["ТМ", "ТП", "Система водоснабжения", "Система водоотведения"],
    UtilityNetworks: ["Система водоснабжения", "Система водоотведения", "Система газоснабжения"],
    Ventilation: ["ОВИК"],
}
constant_parts = ['ПЗ']
for key in MODEL_TO_PART_NAMES:
    MODEL_TO_PART_NAMES[key].extend(constant_parts)


assert len(iter_models_from_module("src.ecology_chapters.chapter1.models")) == len(
    MODEL_TO_PART_NAMES), "chapter1 некорректное кол-во моделей"


def get_part_names_for_model(model: Type[BaseModel] | None) -> list[str] | None:
    """
    Возвращает список `part_name` для фильтрации поиска в Qdrant по конкретной модели.

    - Если `model` отсутствует или для неё нет записи — возвращаем None (поиск по всем разделам).
    - Если список пустой — тоже возвращаем None (поиск по всем), чтобы не получать "пустой" результат.
    """
    if model is None:
        return None
    part_names = MODEL_TO_PART_NAMES.get(model)
    if not part_names:
        return None
    return part_names
