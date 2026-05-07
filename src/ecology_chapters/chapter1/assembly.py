# ─────────────────────────────────────────────
# Корневая модель
# ─────────────────────────────────────────────

from .models import *


class Chapter1Data(BaseModel):
    """
    Корневая модель данных для генерации главы 1
    'Общие сведения об объекте проектирования'.

    Использование:
        data = Chapter1Data(**json_data)
        rendered = template.render(**data.model_dump())
    """
    facility: Facility
    land_plot: LandPlot
    sanitary_zone: SanitaryZone
    structures: Structures
    ownership: Ownership
    surroundings: Surroundings
    general_plan: GeneralPlan
    nearest_objects: NearestObjects
    architecture: Architecture
    heat_supply: HeatSupply
    boilers: Boilers
    pumps: Pumps
    fuel: Fuel
    power_supply: PowerSupply
    water_treatment: WaterTreatment
    utility_networks: UtilityNetworks
    ventilation: Ventilation
