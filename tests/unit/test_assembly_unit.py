from pydantic import BaseModel

from src.ecology_chapters.chapter1.assembly import Chapter1Data
from src.utils.utils import (
    assembly_results_to_docx_context,
    build_chapter_assembly_model,
    FIELD_TO_MODEL_ATTR,
    iter_models_from_module,
    pascal_to_snake,
    pick_assembly_model,
)


def test_pascal_to_snake():
    assert pascal_to_snake("Facility") == "facility"
    assert pascal_to_snake("LandPlot") == "land_plot"
    assert pascal_to_snake("NearestObjects") == "nearest_objects"


def test_chapter1_assembly_has_all_models():
    models = iter_models_from_module("src.ecology_chapters.chapter1.models")
    assert len(Chapter1Data.model_fields) == len(models)


def test_assemble_partial_results():
    assembly_model = build_chapter_assembly_model(
        "src.ecology_chapters.chapter1.models",
        model_name="Chapter1Data",
    )
    partial_results = {
        "architecture": {
            "building_type": "stationary",
            "floors": "1",
            "construction_type": "отдельно стоящая котельная, одноэтажное здание без подвала и технического этажа",
            "module_count": None,
            "axes": "1-6, А-Г",
            "dimensions": "30.0х18.0 м",
            "building_height_m": 8.25,
            "room_height_m": 6.5,
            "wall_thickness_mm": 100,
            "structural_scheme": "стальной каркас",
            "stack_count": 1,
            "stack_diameter_mm": 0,
            "stack_height_m": 17.85,
            "entrance_description": None,
        }
    }
    ctx = assembly_results_to_docx_context(assembly_model, partial_results)
    assert ctx["architecture"]["floors"] == "1"
    assert ctx["architecture"]["module_count"] == "{{ architecture.module_count }}"
    assert ctx["land_plot"]["area_sqm"] == "{{ land_plot.area_sqm }}"
    assert ctx["structures"]["structures"] == ["{{ structures.structures }}"]


def test_pick_assembly_model_finds_dynamic_chapter1():
    model = pick_assembly_model("src.ecology_chapters.chapter1.assembly")
    assert issubclass(model, BaseModel)
    assert model.__name__ == "Chapter1Data"


def test_build_chapter_assembly_model_sets_field_to_model():
    model = build_chapter_assembly_model(
        "src.ecology_chapters.chapter1.models",
        model_name="Chapter1Data",
    )
    field_to_model = getattr(model, FIELD_TO_MODEL_ATTR)
    assert field_to_model["architecture"].__name__ == "Architecture"
