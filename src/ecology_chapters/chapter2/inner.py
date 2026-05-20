"""
Вспомогательные Pydantic-модели (inner) для Главы 2
'Воздействие объекта на земельные ресурсы' раздела ООС.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────


class SoilContaminationCategory(str, Enum):
    """Категория загрязнения почв по СанПиН 2.1.3684-21"""
    clean = "чистая"
    permissible = "допустимая"
    moderately_hazardous = "умеренно опасная"
    hazardous = "опасная"
    extremely_hazardous = "чрезвычайно опасная"


class WasteHazardClass(str, Enum):
    """Класс опасности отходов по Приказу Минприроды"""
    I = "I"
    II = "II"
    III = "III"
    IV = "IV"
    V = "V"


class FloodingCategory(str, Enum):
    """Категория подтопляемости по СП 11-105-97"""
    natural_flooded = "естественно подтопленный"
    potentially_flooded = "потенциально подтопляемый"
    not_flooded = "неподтопляемый"


# ─────────────────────────────────────────────
# Auxiliary models
# ─────────────────────────────────────────────


class IGE(BaseModel):
    """Инженерно-геологический элемент"""

    name: str = Field(
        ...,
        description="Наименование ИГЭ (например: ИГЭ-1, ИГЭ-2)",
    )
    description: str = Field(
        ...,
        description="Описание ИГЭ: тип грунта, мощность, глубина залегания, характеристики",
    )


class FloodingInfo(BaseModel):
    """Информация о подтоплении"""

    area_type: str = Field(
        ...,
        description="Тип района по подтопляемости (например: потенциально подтопляемому)",
    )
    category: str = Field(
        ...,
        description="Категория подтопляемости (например: IIIа – территория, подтопление которой возможно...)",
    )


class SeismicInfo(BaseModel):
    """Информация о сейсмических процессах"""

    locality: str = Field(
        ...,
        description="Название населённого пункта для привязки сейсмических данных",
    )
    rating_a: str = Field(
        ...,
        description="Сейсмичность степени А (10%), баллы",
    )
    rating_b: str = Field(
        ...,
        description="Сейсмичность степени В (5%), баллы",
    )
    rating_c: str = Field(
        ...,
        description="Сейсмичность степени С (1%), баллы",
    )
    ground_category: str = Field(
        ...,
        description="Категория грунтов по сейсмическим свойствам (например: II)",
    )
    site_rating: str = Field(
        ...,
        description="Итоговая сейсмичность площадки, баллы",
    )


class FrostHeaveInfo(BaseModel):
    """Информация о морозном пучении"""

    description: str = Field(
        ...,
        description="Описание морозного пучения грунтов на площадке",
    )


class SuffosionInfo(BaseModel):
    """Информация о суффозии"""

    description: str = Field(
        ...,
        description="Описание суффозионных процессов",
    )


class SoilDepthRecommendation(BaseModel):
    """Рекомендация по использованию почв на определённой глубине"""

    category: str = Field(
        ...,
        description="Категория загрязнения (например: допустимой, чистой)",
    )
    depth_range: str = Field(
        ...,
        description="Диапазон глубин (например: 0,0-0,2 м, 0,2-2,0 м)",
    )
    recommendation: str = Field(
        ...,
        description="Рекомендация по использованию (например: использование без ограничений)",
    )
