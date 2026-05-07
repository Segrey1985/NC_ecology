# ─────────────────────────────────────────────
# Pydantic-модели данных для генерации главы 1
# 'Общие сведения об объекте проектирования' раздела ООС.
#
# Используется совместно с Jinja2-шаблоном chapter1_template.jinja2.
# Рендеринг: template.render(**chapter1_data.model_dump())
# ─────────────────────────────────────────────

from typing import Optional
from pydantic import BaseModel, Field

from .inner import *


class Facility(BaseModel):
    """Основные сведения об объекте проектирования"""
    
    type_nominative: str = Field(
        ...,
        description="Полное наименование объекта в именительном падеже",
        examples=["блочно-модульная газовая котельная", "котельная"],
    )
    type_genitive: str = Field(
        ...,
        description="Наименование объекта в родительном падеже",
        examples=["котельной", "блочно-модульной газовой котельной"],
    )
    gender: Gender = Field(
        ...,
        description="Грамматический род существительного (m/f/n)",
    )
    work_type: WorkType = Field(
        ...,
        description="Вид проектных работ",
    )
    project_goal: Optional[str] = Field(
        None,
        description="Цель строительства/реконструкции/перевооружения",
        examples=["перевод котельной на газообразное топливо"],
    )
    capacity_text: str = Field(
        ...,
        description="Мощность объекта в текстовом формате",
        examples=["0,8 МВт", "4,0 МВт"],
    )
    capacity_mw: float = Field(
        ...,
        description="Мощность объекта в МВт (число)",
    )
    capacity_gcal: float = Field(
        ...,
        description="Мощность объекта в Гкал/ч (число)",
    )
    address: str = Field(
        ...,
        description="Полный адрес объекта",
    )
    heat_consumers: str = Field(
        ...,
        description="Описание потребителей тепловой энергии",
        examples=["многоквартирных домов и объектов социального назначения"],
    )


class LandPlot(BaseModel):
    """Сведения о земельном участке"""
    
    cadastral_number: str = Field(
        ...,
        description="Кадастровый номер земельного участка",
        pattern=r"^\d{2}:\d{2}:\d{7}:\d+$",
    )
    land_category: str = Field(
        ...,
        description="Категория земель",
        examples=["земли населенных пунктов"],
    )
    permitted_use: str = Field(
        ...,
        description="Вид разрешенного использования",
        examples=["предоставление коммунальных услуг", "коммунальное обслуживание"],
    )
    area_sqm: float = Field(
        ...,
        description="Площадь земельного участка в кв.м",
    )
    territorial_zone: Optional[str] = Field(
        None,
        description="Код территориальной зоны по ПЗЗ",
        examples=["Т2Ж1", "ПР", "П2"],
    )
    coordinate_system: Optional[str] = Field(
        None,
        description="Система координат для поворотных точек",
        examples=["МСК-47", "МСК-53"],
    )
    boundary_coordinates: Optional[str] = Field(
        None,
        description="Координаты поворотных точек границ участка",
    )


class Structures(BaseModel):
    """Перечень сооружений, размещаемых на участке"""
    
    structures: list[str] = Field(
        default_factory=list,
        examples=[["здание котельной", "дымовая труба", "ограждение из 3D сварной сетки"]]
    )


class Ownership(BaseModel):
    """Документ, подтверждающий право пользования участком"""
    
    ownership: Optional[str] = Field(
        None,
        examples=["договором аренды земельного участка № 123 от 01.01.2024"]
    )


class SanitaryZone(BaseModel):
    """Санитарно-защитная зона"""
    
    include_description: bool = Field(
        True,
        description="Включать ли описание СЗЗ в текст главы",
    )
    hazard_class: Optional[HazardClass] = Field(
        None,
        description="Класс опасности по СанПиН",
    )
    size_m: Optional[int] = Field(
        None,
        description="Размер СЗЗ в метрах",
        examples=[50, 100, 300, 500, 1000],
    )
    sanpin_reference: Optional[str] = Field(
        None,
        description="Ссылка на конкретный пункт СанПиН",
        examples=["гл. VII, табл. 7.1. р.10, п.10.4.1"],
    )


class Surroundings(BaseModel):
    """Описание окружения объекта"""
    
    directions: list[SurroundingDirection] = Field(
        ...,
        description="Описание окружения по сторонам света",
    )
    data_sources: list[str] = Field(
        ...,
        description="Источники данных для анализа окружения",
        examples=[["публичная кадастровая карта (https://pkk.rosreestr.ru)",
                   "ситуационный план размещения котельной М 1:500 (Приложение 1)"]],
    )


class GeneralPlan(BaseModel):
    """Сведения из Генерального плана МО"""
    
    municipality_name: str = Field(
        ...,
        description="Наименование муниципального образования",
    )
    approval_details: str = Field(
        ...,
        description="Реквизиты документа об утверждении Генплана",
    )
    territorial_zones: list[TerritorialZone] = Field(
        ...,
        description="Территориальные зоны вокруг объекта",
    )


class NearestObjects(BaseModel):
    """"Перечень ближайших нормируемых объектов"""
    
    nearest_objects: list[NearestObject] = Field(
        default_factory=list,
    )


class Architecture(BaseModel):
    """Архитектурно-планировочные решения"""
    
    building_type: BuildingType = Field(
        ...,
        description="Тип здания котельной (modular — БМК, stationary — стационарное)",
    )
    floors: Optional[str] = Field(
        None,
        description="Этажность здания",
        examples=["одноэтажное"],
    )
    construction_type: Optional[str] = Field(
        None,
        description="Тип исполнения здания",
        examples=["отдельно стоящее, кирпичное", "блочно-модульное"],
    )
    module_count: Optional[int] = Field(
        None,
        description="Количество модулей (для БМК)",
    )
    axes: Optional[str] = Field(
        None,
        description="Обозначение осей здания",
        examples=["А-Б/1-4"],
    )
    dimensions: str = Field(
        ...,
        description="Габаритные размеры здания в осях",
        examples=["7,8 x 2,35"],
    )
    building_height_m: Optional[float] = Field(
        None,
        description="Максимальная высота здания, м",
    )
    room_height_m: Optional[float] = Field(
        None,
        description="Высота помещений до низа строительных конструкций (для БМК), м",
    )
    wall_thickness_mm: Optional[int] = Field(
        None,
        description="Толщина стен-сэндвич (для БМК), мм",
    )
    structural_scheme: Optional[str] = Field(
        None,
        description="Конструктивная схема здания",
        examples=["безкаркасная", "каркасная"],
    )
    stack_count: int = Field(
        ...,
        description="Количество газоотводящих стволов",
    )
    stack_diameter_mm: int = Field(
        ...,
        description="Внутренний диаметр газоотводящих стволов, мм",
    )
    stack_height_m: float = Field(
        ...,
        description="Высота верха газоотводящих стволов, м",
    )
    entrance_description: Optional[str] = Field(
        None,
        description="Описание въезда/выезда на промплощадку",
        examples=["с северо-западной стороны"],
    )


class HeatSupply(BaseModel):
    """Параметры системы теплоснабжения"""
    
    operation_mode: str = Field(
        ...,
        description="Режим работы",
        examples=["круглосуточно, отопительный период"],
    )
    system_type: str = Field(
        ...,
        description="Тип системы теплоснабжения",
        examples=["2-х трубная, закрытая"],
    )
    heat_carrier: str = Field(
        ...,
        description="Теплоноситель",
        examples=["вода"],
    )
    temperature_schedule: str = Field(
        ...,
        description="Температурный график",
        examples=["95/70 °С", "105/80 °С"],
    )


class Boilers(BaseModel):
    """"Перечень котлов"""
    
    boilers: list[Boiler] = Field(
        default_factory=list,
    )


class Pumps(BaseModel):
    """"Перечень вспомогательного оборудования (насосы)"""
    
    pumps: list[Pump] = Field(
        default_factory=list,
    )


class Fuel(BaseModel):
    """Сведения о топливе"""
    
    primary_fuel: str = Field(
        ...,
        description="Вид основного топлива",
        examples=["природный газ"],
    )
    calorific_value: Optional[float] = Field(
        None,
        description="Теплотворная способность, ккал/м³",
    )
    fuel_density: Optional[float] = Field(
        None,
        description="Плотность топлива, кг/м³",
    )
    has_emergency_fuel: bool = Field(
        False,
        description="Предусмотрено ли аварийное/резервное топливо",
    )
    emergency_fuel_description: Optional[str] = Field(
        None,
        description="Описание аварийного/резервного топлива",
    )
    annual_consumption: float = Field(
        ...,
        description="Годовой расход топлива, тыс. м³",
    )
    total_max: Optional[float] = Field(
        None,
        description="Суммарный расход при макс. производительности, м³/ч",
    )
    total_cold_month: Optional[float] = Field(
        None,
        description="Суммарный расход в режиме холодного месяца, м³/ч",
    )
    total_min: Optional[float] = Field(
        None,
        description="Суммарный расход при мин. производительности, м³/ч",
    )
    consumption_table: list[FuelConsumptionRow] = Field(
        ...,
        description="Таблица расходов топлива по потребителям",
    )


class PowerSupply(BaseModel):
    """Сведения об электроснабжении"""
    
    source: str = Field(
        ...,
        description="Источник электроснабжения",
    )
    reliability_category: Optional[ReliabilityCategory] = Field(
        None,
        description="Категория надёжности электроснабжения",
    )
    has_diesel_generator: bool = Field(
        False,
        description="Наличие ДГУ",
    )
    diesel_generator_model: Optional[str] = Field(
        None,
        description="Марка ДГУ",
    )
    diesel_generator_power_kw: Optional[float] = Field(
        None,
        description="Мощность ДГУ, кВт",
    )
    diesel_tank_volume_l: Optional[float] = Field(
        None,
        description="Объём топливного бака ДГУ, л",
    )
    diesel_runtime_hours: Optional[float] = Field(
        None,
        description="Время автономной работы ДГУ, часов",
    )


class WaterTreatment(BaseModel):
    """Химводоочистка"""
    
    equipment: list[str] = Field(
        ...,
        description="Перечень блоков системы водоподготовки",
    )


class UtilityNetworks(BaseModel):
    """Инженерные сети"""
    
    gas_supply: Optional[str] = Field(
        None,
        description="Описание газоснабжения",
    )
    water_supply: Optional[str] = Field(
        None,
        description="Описание водоснабжения",
    )
    sewerage: Optional[str] = Field(
        None,
        description="Описание водоотведения",
    )


class Ventilation(BaseModel):
    """Отопление и вентиляция"""
    
    no_summer_operation: bool = Field(
        True,
        description="Котельная не работает в тёплый период",
    )
    has_emergency_ventilation: bool = Field(
        False,
        description="Наличие аварийной вентиляции",
    )
    has_heating: bool = Field(
        False,
        description="Наличие отопления в помещении",
    )
    heating_equipment: Optional[str] = Field(
        None,
        description="Оборудование отопления",
    )
    louver_size: Optional[str] = Field(
        None,
        description="Размер жалюзийных решеток",
        examples=["2400 х 1200 мм"],
    )
    deflector_diameter_mm: Optional[int] = Field(
        None,
        description="Диаметр дефлектора, мм",
    )
    deflector_count: Optional[int] = Field(
        None,
        description="Количество дефлекторов",
    )
