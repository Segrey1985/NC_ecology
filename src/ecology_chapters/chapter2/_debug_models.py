"""
Дебаг: какие pydantic-модели гонять в main2 при test_mode="filter".
None — все модели из chapter*.models
"""

# ACTIVE_MODEL_NAMES: list[str] | None = None

ACTIVE_MODEL_NAMES = ["Geology", "Facility", "PhysicalFactors", "Technogenic"]
