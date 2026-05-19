"""
Дебаг: какие pydantic-модели гонять в main2 при test_mode="filter".

None — все модели из chapter*.models
["Facility", "LandPlot"] — только перечисленные (имена классов)

chapter1: Facility, LandPlot, SanitaryZone, Structures, Ownership, Surroundings,
GeneralPlan, NearestObjects, Architecture, HeatSupply, Boilers, Pumps, Fuel,
PowerSupply, WaterTreatment, UtilityNetworks, Ventilation
"""

# ACTIVE_MODEL_NAMES: list[str] | None = None

# ACTIVE_MODEL_NAMES = ["Architecture", "Boilers", "NearestObjects", "Surroundings", "GeneralPlan"]
ACTIVE_MODEL_NAMES = ["Architecture"]
