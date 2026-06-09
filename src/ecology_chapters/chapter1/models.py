from typing import Optional
from pydantic import BaseModel, Field, PrivateAttr

from .inner import *


class LandPlot(BaseModel):
    """Сведения о земельном участке"""

    cadastral_number: str = Field(
        ...,
        description="Кадастровый номер земельного участка",
        pattern=r"^\d{2}:\d{2}:\d{7}:\d+$",
    )
    land_category: str = Field(
        "не определено",
        description="Категория земель (например: земли населенных пунктов).",
    )
    permitted_use: str = Field(
        "не определено",
        description="Вид разрешенного использования (например: предоставление коммунальных услуг, коммунальное обслуживание).",
    )
    area_sqm: float = Field(
        ...,
        description="Площадь земельного участка в кв.м",
    )
    territorial_zone: Optional[str] = Field(
        None,
        description="Код территориальной зоны по ПЗЗ (например: Т2Ж1, ПР, П2).",
    )
    _part_name: str = PrivateAttr(default=['ПЗУ', 'ПЗ'])


class Structures(BaseModel):
    """Перечень сооружений, размещаемых на участке"""

    structures: list[str] = Field(
        default_factory=list,
        description="Перечень сооружений на участке. Пример элементов: здание котельной, дымовая труба, ограждение из 3D сварной сетки.",
    )
    _part_name: str = PrivateAttr(default=['ПЗУ', 'АР', 'ПЗ'])


class Ownership(BaseModel):
    """Документ, подтверждающий право пользования участком"""

    ownership: Optional[str] = Field(
        None,
        description="Формулировка права пользования участком (например: договором аренды земельного участка № 123 от 01.01.2024).",
    )
    _part_name: str = PrivateAttr(default=['ПЗУ', 'ПЗ'])


class SanitaryZone(BaseModel):
    """Санитарно-защитная зона"""

    hazard_class: Optional[HazardClass] = Field(
        None,
        description="Класс опасности по СанПиН",
    )
    sanpin_reference: Optional[str] = Field(
        None,
        description="Ссылка на конкретный пункт СанПиН (например: гл. VII, табл. 7.1. р.10, п.10.4.1).",
    )
    _part_name: str = PrivateAttr(default=['ПЗУ', 'ПЗ'])


class Surroundings(BaseModel):
    """Земельный участок ограничен ... (перечень объектов окружения кадастровые номера, расстояния, назначение)"""

    directions: list[SurroundingDirection] = Field(
        ...,
        description="Описание окружения по сторонам света",
    )
    _part_name: str = PrivateAttr(default=['ПЗУ', 'ПЗ'])
    _use_parent: bool = PrivateAttr(default=True)


class GeneralPlan(BaseModel):
    """Ограничения площадки по сведениям из Генерального плана"""

    constraints_on_the_industrial_site: str = Field(
        None,
        description="""
        Перечь зон согласно генеральному плану граничащих со строительной площадкой"
        Например: В соответствии со сведениями Генерального плана ...
        ... муниципального района ... области, утвержденного решением ...
         от 01.01.2000 г. промплощадка ограничена
         - с севера - Зона ...
         - c востока - Зона застройки жилыми домами
         - с юга - функциональная зона ...
         """
    )
    _part_name: str = PrivateAttr(default=['ПЗУ', 'ПЗ'])


class NearestObjects(BaseModel):
    """Перечень ближайших нормируемых объектов в радиусе нормативной санитарно-защитной зоны"""

    nearest_objects: list[NearestObject] = Field(
        default_factory=list,
    )
    _part_name: str = PrivateAttr(default=['ПЗУ', 'ПЗ'])


class Architecture(BaseModel):
    """Архитектурно-планировочные решения"""

    building_type: BuildingType = Field(
        ...,
        description="Тип здания котельной (modular — БМК, stationary — стационарное)",
    )
    floors: Optional[str] = Field(
        None,
        description="Этажность здания (например: одноэтажное).",
    )
    construction_type: Optional[str] = Field(
        None,
        description="Тип исполнения здания (например: отдельно стоящее, кирпичное; блочно-модульное).",
    )
    module_count: Optional[int] = Field(
        None,
        description="Количество модулей (для БМК)",
    )
    axes: Optional[str] = Field(
        None,
        description="Обозначение осей здания (например: А-Б/1-4).",
    )
    dimensions: str = Field(
        ...,
        description="Габаритные размеры здания в осях (например: 7,8 x 2,35).",
    )
    building_height_m: Optional[float] = Field(
        None,
        description="Максимальная высота здания, м",
    )
    room_height_m: Optional[float] = Field(
        None,
        description="Высота помещений до низа строительных конструкций (для БМК), м",
    )
    _part_name: str = PrivateAttr(default=['АР'])

    
class Constructive(BaseModel):
    """Конструктивные решения"""
    
    wall_thickness_mm: Optional[int] = Field(
        None,
        description="Толщина наружных панелей стен-сэндвич, мм",
    )
    structural_scheme: Optional[str] = Field(
        None,
        description="Конструктивная схема здания (например: бескаркасная, каркасная, полный каркас).",
    )
    entrance_description: Optional[str] = Field(
        None,
        description="Описание въезда/выезда на промплощадку. Пример: Въезд и выезд на промплощадку предусмотрен с южной стороны.",
    )
    _part_name: str = PrivateAttr(default=['АР'])
    
    
class GasFlue(BaseModel):
    """Газоотводящие (дымовые) трубы"""
    
    stack_count: int = Field(
        ...,
        description="Количество газоотводящих стволов (дымовых труб)",
    )
    stack_diameter_mm: int = Field(
        ...,
        description="Диаметр газоотводящих стволов (дымовых труб), мм (Ду)",
    )
    stack_height_m: float = Field(
        ...,
        description="Высота верха газоотводящих стволов (дымовых труб), м",
    )
    _part_name: str = PrivateAttr(default=['АР', 'КР'])
    


class HeatSupply(BaseModel):
    """Параметры системы теплоснабжения"""

    operation_mode: str = Field(
        ...,
        description="Режим работы (например: круглосуточно, отопительный период).",
    )
    system_type: str = Field(
        ...,
        description="Тип системы теплоснабжения (например: 2-х трубная, закрытая).",
    )
    heat_carrier: str = Field(
        ...,
        description="Теплоноситель (например: вода).",
    )
    temperature_schedule: str = Field(
        ...,
        description="Температурный график (например: 95/70 °С, 105/80 °С).",
    )
    _part_name: str = PrivateAttr(default=['ТМ', 'ПЗ'])


class Boilers(BaseModel):
    """Перечень котлов"""

    boilers: list[Boiler] = Field(
        default_factory=list,
    )
    _part_name: str = PrivateAttr(default=['ТМ', 'ПЗ'])


class Pumps(BaseModel):
    """Перечень вспомогательного оборудования (насосы) без дублирования"""

    pumps: list[Pump] = Field(
        description="Вспомогательное оборудование 1 и 2 очереди строительства. Насос ...",
        default_factory=list,
    )
    _part_name: str = PrivateAttr(default=['ПЗ'])


class Fuel(BaseModel):
    """Сведения об основном топливе"""

    primary_fuel: str = Field(
        ...,
        description="Вид основного топлива (например: природный газ).",
    )
    calorific_value: Optional[float] = Field(
        None,
        description="Теплотворная способность, ккал/м³",
    )
    fuel_density: Optional[float] = Field(
        None,
        description="Плотность топлива, кг/м³",
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
    _part_name: str = PrivateAttr(default=['ТП'])


class EmergencyFuel(BaseModel):
    """Сведения об аварийном топливе"""
    
    has_emergency_fuel: bool = Field(
        False,
        description="Предусмотрено ли аварийное/резервное топливо",
    )
    emergency_fuel_description: Optional[str] = Field(
        None,
        description="Полное описание аварийного/резервного топлива. ... в случае прекращения подачи газа на котельную, проектом предусматривается ...",
    )
    _part_name: str = PrivateAttr(default=['ТП'])


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
        description="В тексте явно говорится о наличии дизельного генератора",
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
    _part_name: str = PrivateAttr(default=['Система электроснабжения', 'ПЗ'])


class WaterTreatment(BaseModel):
    """Химводоочистка"""

    equipment: list[str] = Field(
        ...,
        description="Перечень блоков системы водоподготовки. ... рекомендуем установить систему водоподготовки, состоящую из следующих блоков...",
    )
    _part_name: str = PrivateAttr(default=['ТМ', 'ПЗ'])


class UtilityNetworks(BaseModel):
    """Инженерные сети"""

    gas_supply: Optional[str] = Field(
        None,
        description="Описание газоснабжения. Пример: Газоснабжение предусмотрено в соответствие с Техническими условиями на подключение (технологическое присоединение) газоиспользующего оборудования и объектов капитального строительства к сетям газорапределения, выданными ...",
    )
    water_supply: Optional[str] = Field(
        None,
        description="Описание водоснабжения. Пример: Проектом предусматривается подключение объекта к централизованной системе холодного водоснабжения согласно Техническим условиям, выданными...",
    )
    sewerage: Optional[str] = Field(
        None,
        description="Описание водоотведения. Пример: Водоотведение предусматривается в емкость для приема стоков / водосборный колодец.",
    )
    _part_name: str = PrivateAttr(default=['Система водоснабжения', 'Система водоотведения', 'Система газоснабжения', 'ПЗ'])


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
        description="Размер жалюзийных решеток (например: 2400 х 1200 мм).",
    )
    deflector_diameter_mm: Optional[int] = Field(
        None,
        description="Диаметр дефлектора, мм",
    )
    deflector_count: Optional[int] = Field(
        None,
        description="Количество дефлекторов",
    )
    _part_name: str = PrivateAttr(default=['ОВИК', 'ПЗ'])
