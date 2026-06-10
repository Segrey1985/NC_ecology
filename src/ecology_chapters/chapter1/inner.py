from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────


class Gender(str, Enum):
    """Грамматический род существительного"""
    masculine = "m"
    feminine = "f"
    neuter = "n"


class WorkType(str, Enum):
    """Вид проектных работ"""
    construction = "строительство"
    technical_reequipment = "техническое перевооружение"
    reconstruction = "реконструкция"


class HazardClass(str, Enum):
    """Класс опасности по СанПиН"""
    I = "I"
    II = "II"
    III = "III"
    IV = "IV"
    V = "V"


class BuildingType(str, Enum):
    """Тип здания котельной"""
    modular = "modular"
    stationary = "stationary"


class ReliabilityCategory(str, Enum):
    """Категория надёжности электроснабжения"""
    I = "I"
    II = "II"
    III = "III"
    
    
# ─────────────────────────────────────────────
# auxiliary
# ─────────────────────────────────────────────

class SurroundingDirection(BaseModel):
    """Описание окружения с одной стороны света"""

    cardinal_direction: str = Field(
        ...,
        description="Сторона света. Примеры формулировок: «с севера», «с юго-востока», «с запада».",
    )
    description: str = Field(
        ...,
        description="Описание объектов с данной стороны, включая: "
                    "- расстояние от контура объекта, "
                    "- кадастровый номер, "
                    "- категория земель, "
                    "- вид разрешенного использования.",
    )
    

class TerritorialZone(BaseModel):
    """Территориальная зона из Генерального плана"""

    direction: str = Field(
        ...,
        description="Направление (сторона света)",
    )
    zone_name: str = Field(
        ...,
        description="Наименование территориальной зоны",
    )
    

class NearestObject(BaseModel):
    """Ближайший нормируемый объект"""

    number: int = Field(
        ...,
        description="Порядковый номер",
    )
    cadastral_number: Optional[str] = Field(
        None,
        description="Кадастровый номер нормируемого объекта",
    )
    description: str = Field(
        ...,
        description="Вид нормируемого объекта, адрес",
    )
    distance: str = Field(
        ...,
        description="Расстояние до контура объекта (м)",
    )


class Boiler(BaseModel):
    """Котёл"""

    name: str = Field(
        ...,
        description="Наименование (марка) котла",
    )
    manufacturer: str = Field(
        ...,
        description="Производитель котла",
    )
    capacity_kw: float = Field(
        ...,
        description="Единичная мощность котла, кВт",
    )
    burner_type: str = Field(
        ...,
        description="Тип горелки (например: газовой, газомазутной).",
    )
    burner_model: str = Field(
        ...,
        description="Марка горелки",
    )
    burner_capacity: str = Field(
        ...,
        description="Диапазон мощности горелки, кВт",
    )


class Pump(BaseModel):
    """Насос (вспомогательное оборудование)"""

    name: str = Field(
        ...,
        description="Наименование (функциональное назначение) насоса",
    )
    model: str = Field(
        ...,
        description="Марка насоса",
    )
    manufacturer: str = Field(
        ...,
        description="Производитель насоса",
    )
    flow_rate: float = Field(
        ...,
        description="Расход, м³/ч",
    )
    head: float = Field(
        ...,
        description="Напор, м в.ст.",
    )
    operation_scheme: str = Field(
        ...,
        description="Схема работы (например: «1 раб., 1 рез.» или «1 раб.»).",
    )
    
    
class FuelConsumptionRow(BaseModel):
    """Строка таблицы расходов топлива"""

    consumer_name: str = Field(
        ...,
        description="Наименование потребителя (котла)",
    )
    max_rate: float = Field(
        ...,
        description="Расход при макс. производительности, м³/ч",
    )
    cold_month_rate: Optional[float] = Field(
        None,
        description="Расход в режиме холодного месяца, м³/ч",
    )
    min_rate: Optional[float] = Field(
        None,
        description="Расход при мин. производительности, м³/ч",
    )