from __future__ import annotations
from typing import Type
from pydantic import BaseModel

from .models import (
    DangerousProcesses,
    Facility,
    Geology,
    Hydrogeology,
    LandPlot,
    LandUse,
    Measures,
    PhysicalFactors,
    Soil,
    Subsoil,
    Technogenic,
)
from src.utils.utils import iter_models_from_module


MODEL_TO_PART_NAMES: dict[Type[BaseModel], list[str]] = {
    DangerousProcesses: ["Изыскания"],
    Facility: ["ПЗ"],
    Geology: ["Изыскания"],
    Hydrogeology: ["Изыскания"],
    LandPlot: ["ПЗУ"],
    LandUse: ["ПЗУ"],
    Measures: ["Изыскания", "ПЗ"],
    PhysicalFactors: ["Изыскания"],
    Soil: ["Изыскания"],
    Subsoil: ["Изыскания"],
    Technogenic: ["Изыскания"],
}

constant_parts = []
for key in MODEL_TO_PART_NAMES:
    MODEL_TO_PART_NAMES[key].extend(constant_parts)


assert len(iter_models_from_module("src.ecology_chapters.chapter2.models")) == len(
    MODEL_TO_PART_NAMES), "chapter2 некорректное кол-во моделей"
