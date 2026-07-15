"""
Pydantic-модели данных для генерации Главы 2
'Воздействие объекта на земельные ресурсы' раздела ООС.

Используется совместно с docxtpl-шаблоном chapter2_template.docx.
Рендеринг: template.render(**chapter2_data.model_dump())
"""

from typing import Optional
from pydantic import BaseModel, Field, PrivateAttr

from .inner import (
    IGE,
    FloodingInfo,
    SeismicInfo,
    FrostHeaveInfo,
    SuffosionInfo,
    SoilDepthRecommendation,
)


class Geology(BaseModel):
    """2.1.1. Инженерно-геологические условия района расположения объекта проектирования"""

    description: Optional[str] = Field(
        None,
        description=(
            "Полное описание инженерно-геологических условий площадки: "
            "глубина бурения, типы отложений, характеристики грунтов. "
            "Пример: 'Геологическое строение площадки изучено до глубины 10,0 м. "
            "Площадка сложена четвертичными отложениями...'"
        ),
    )
    ige_table: Optional[list[IGE]] = Field(
        None,
        description="Перечень инженерно-геологических элементов (ИГЭ). Должен отражать все перечисленные ИГЭ в разделах 'description', 'num_layers'",
        json_schema_extra={"vanish": True},
    )
    num_layers_: Optional[str] = Field(None, description="Количество инженерно-геологических элементов (ИГЭ)")
    _use_parent: bool = PrivateAttr(default=True)
    _part_name: str = PrivateAttr(default=["ИГИ"])


class Hydrogeology(BaseModel):
    """2.1.2. Гидрогеологические условия района расположения объекта проектирования"""

    description: Optional[str] = Field(
        None,
        description=(
            "Описание гидрогеологических условий: наличие подземных вод, глубина залегания, "
            "абсолютные отметки, тип питания, прогноз подъёма уровня. "
            "Пример: 'В период проведения полевых работ подземные воды вскрыты на глубине 2,5 м...'"
        ),
    )
    surface_water_description: Optional[str] = Field(
        None,
        description=(
            "Описание поверхностных водных объектов вблизи площадки (при наличии). "
            "Пример: 'Ближайший водный объект – р. Нева, расположена в 500 м к югу от площадки.'"
        ),
        json_schema_extra={"vanish": True},
    )
    _part_name: str = PrivateAttr(default=["ИГИ"])


class DangerousProcesses(BaseModel):
    """2.1.3. Характеристика опасных экзогенных процессов"""

    not_detected: bool = Field(
        False,
        description="Если True – опасные процессы не выявлены, выводится стандартная формулировка.",
        json_schema_extra={"vanish": True},
    )
    flooding: Optional[FloodingInfo] = Field(
        None,
        description="Информация о подтоплении (при наличии)",
        json_schema_extra={"vanish": True},
    )
    seismic: Optional[SeismicInfo] = Field(
        None,
        description="Информация о сейсмических процессах (при наличии)",
        json_schema_extra={"vanish": True},
    )
    frost_heave: Optional[FrostHeaveInfo] = Field(
        None,
        description="Информация о морозном пучении (при наличии)",
        json_schema_extra={"vanish": True},
    )
    suffosion: Optional[SuffosionInfo] = Field(
        None,
        description="Информация о суффозии (при наличии)",
        json_schema_extra={"vanish": True},
    )
    additional_description: Optional[str] = Field(
        None,
        description="Дополнительное описание иных опасных процессов (при наличии)",
        json_schema_extra={"vanish": True},
    )
    _part_name: str = PrivateAttr(default=["ИГИ"])


class Soil(BaseModel):
    """2.1.4. Почвенные условия территории"""

    sampling_description: Optional[str] = Field(
        None,
        description=(
            "Описание процедуры отбора проб: количество проб, глубины, лаборатория. "
            "Пример: 'На территории участка были отобраны 3 объединённые пробы с глубин 0,0-0,2 м и 0,2-2,0 м.'"
        ),
        json_schema_extra={"vanish": True},
    )
    chemical_results: Optional[str] = Field(
        None,
        description=(
            "Результаты химического анализа почв: суммарный показатель загрязнения Zc, "
            "категория загрязнения, превышения ПДК. "
            "Пример: 'По результатам химического анализа суммарный показатель загрязнения Zc "
            "составил менее 16 ед., что соответствует категории «допустимая».'"
        ),
    )
    petroleum_results: Optional[str] = Field(
        None,
        description=(
            "Результаты анализа на нефтепродукты. "
            "Пример: 'Содержание нефтепродуктов составляет 42 мг/кг, что не превышает ОДК (300 мг/кг).'"
        ),
        json_schema_extra={"vanish": True},
    )
    benzpyrene_results: Optional[str] = Field(
        None,
        description=(
            "Результаты анализа на 3,4-бенз(а)пирен. "
            "Пример: 'Содержание 3,4-бенз(а)пирена составляет 0,005 мг/кг при ПДК 0,02 мг/кг.'"
        ),
        json_schema_extra={"vanish": True},
    )
    microbiology_results: Optional[str] = Field(
        None,
        description=(
            "Результаты микробиологических и паразитологических исследований. "
            "Пример: 'По санитарно-бактериологическим и паразитологическим показателям "
            "почвы относятся к категории «Чистая».'"
        ),
    )
    toxicology_results: Optional[str] = Field(
        None,
        description=(
            "Результаты токсикологических исследований и класс опасности. "
            "Пример: 'По результатам биотестирования почвогрунт относится к V классу опасности – "
            "практически неопасный.'"
        ),
    )
    radiation_results: Optional[str] = Field(
        None,
        description=(
            "Результаты радиационного обследования. "
            "Пример: 'Мощность амбиентной дозы гамма-излучения на территории ниже "
            "утвержденного норматива (не более 0,30 мкЗв/ч). Радиационных аномалий не обнаружено.'"
        ),
    )
    depth_recommendations: Optional[list[SoilDepthRecommendation]] = Field(
        None,
        description="Рекомендации по использованию почв по глубинам",
    )
    cipher: Optional[str] = Field(
        None,
        description="Шифр тома проектной документации. Например: '123-2026-ИЭИ' или '05/24-ИЭИ'"
    )
    _part_name: str = PrivateAttr(default=["ИЭИ"])


class LandUse(BaseModel):
    """2.1.5. Характер землепользования района проектирования"""

    site_condition: Optional[str] = Field(
        None,
        description=(
            "Описание текущего состояния участка. "
            "Пример: 'На участке проектирования объекты капитального строительства отсутствуют.'"
        ),
    )
    restrictions_text: Optional[str] = Field(
        None,
        description=(
            "Описание ограничений по данным справок от уполномоченных органов: "
            "ООПТ, объекты культурного наследия, ЗСО, скотомогильники, свалки и т.д. "
            "Каждое ограничение — отдельное предложение."
        ),
    )
    zouit_list: Optional[list[str]] = Field(
        None,
        description=(
            "Перечень ЗОУИТ (зон с особыми условиями использования территории), "
            "попадающих в границы участка. Заполняется при наличии."
        ),
        json_schema_extra={"vanish": True},
    )
    absence_items: Optional[list[str]] = Field(
        None,
        description=(
            "Перечень объектов/зон, отсутствующих на участке (для перечисления списком). "
            "Пример элементов: 'курорты и природно-лечебные ресурсы', "
            "'объекты государственной мелиоративной системы'"
        ),
        json_schema_extra={"vanish": True},
    )
    _part_name: str = PrivateAttr(default=["ПЗУ"])


class Technogenic(BaseModel):
    """2.1.6. Техногенное нарушение территории"""

    description: Optional[str] = Field(
        None,
        description=(
            "Описание техногенного состояния территории. "
            "Пример: 'В настоящее время территория участка представляет собой "
            "частично застроенную территорию с асфальтовым покрытием.'"
        ),
    )
    topsoil_not_fertile: bool = Field(
        False,
        description="Если True — поверхностный слой не является плодородным, снятие ПСП не предусмотрено.",
        json_schema_extra={"vanish": True},
    )
    topsoil_description: Optional[str] = Field(
        None,
        description=(
            "Описание состояния плодородного слоя и рекомендации. "
            "Пример: 'Поверхностный слой почв не является плодородным. "
            "Снятие плодородного слоя почвы (ПСП) не предусматривается.'"
        ),
    )
    _part_name: str = PrivateAttr(default=["ИГИ", "ИЭИ"])


class PhysicalFactors(BaseModel):
    """2.1.7. Физические факторы воздействия"""

    noise_description: Optional[str] = Field(
        None,
        description=(
            "Описание результатов измерений шума. "
            "Пример: 'По результатам измерений эквивалентный уровень звука составил 48 дБА, "
            "максимальный – 56 дБА при нормативных значениях 55/70 дБА (дневное время).'"
        ),
    )
    emf_description: Optional[str] = Field(
        None,
        description=(
            "Описание результатов измерений электромагнитных полей. "
            "Пример: 'Напряженность электрического поля промышленной частоты 50 Гц "
            "составила 0,01 кВ/м при ПДУ 5 кВ/м. Индукция магнитного поля – 0,1 мкТл при ПДУ 5 мкТл.'"
        ),
    )
    vibration_description: Optional[str] = Field(
        None,
        description=(
            "Описание результатов измерений вибрации (при наличии). "
            "Пример: 'Уровень общей вибрации составил 42 дБ при ПДУ 72 дБ.'"
        ),
        json_schema_extra={"vanish": True},
    )
    _part_name: str = PrivateAttr(default=["ИЭИ"])


class Measures(BaseModel):
    """2.5. Мероприятия по охране и рациональному использованию земельных ресурсов"""

    additional_items: Optional[list[str]] = Field(
        None,
        description=(
            "Дополнительные мероприятия сверх стандартных. "
            "Пример элементов: 'организация площадок временного хранения отходов с твёрдым покрытием', "
            "'проведение рекультивации нарушенных земель'"
        ),
        json_schema_extra={"vanish": True},
    )
    _part_name: str = PrivateAttr(default=["ИГИ", "ИЭИ", "ПЗ"])


class LandPlot(BaseModel):
    """Ссылка на модель LandPlot из models.py Главы 1 (минимальные поля для Главы 2)"""

    cadastral_number: Optional[str] = Field(
        None,
        description="Кадастровый номер земельного участка",
    )
    land_category: Optional[str] = Field(
        None,
        description="Категория земель (например: земли населенных пунктов).",
    )
    permitted_use: Optional[str] = Field(
        None,
        description="Вид разрешенного использования.",
    )
    area_sqm: Optional[float] = Field(
        None,
        description="Площадь земельного участка в кв.м",
    )
    _part_name: str = PrivateAttr(default=["ПЗУ"])


# if __name__ == "__main__":
#     from src.utils.utils import iter_models_from_module
#
#     for x in iter_models_from_module("src.ecology_chapters.chapter2.models"):
#         print(x.__name__)
