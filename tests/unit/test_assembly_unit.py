from pydantic import BaseModel

from src.ecology_chapters.chapter1.assembly import Chapter1Data
from src.utils.utils import (
    filter_mode_payload_and_validate,
    filter_mode_assembly_to_docx_context,
    build_chapter_assembly_model,
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
    assembly = build_chapter_assembly_model(
        "src.ecology_chapters.chapter1.models",
        model_name="Chapter1Data",
    )
    data = filter_mode_payload_and_validate(
        assembly,
        {
            "Facility": {
                "type_nominative": "котельная",
                "type_genitive": "котельной",
                "gender": "f",
                "work_type": "строительство",
                "project_goal": "цель",
                "capacity_text": "1 МВт",
                "capacity_mw": 1.0,
                "capacity_gcal": 0.86,
                "address": "адрес",
                "heat_consumers": "потребители",
            }
        },
    )
    ctx = filter_mode_assembly_to_docx_context(assembly, data)
    assert ctx["facility"]["type_nominative"] == "котельная"
    assert ctx["land_plot"] == {}

    ctx_filter = filter_mode_assembly_to_docx_context(assembly, data, preserve_unfilled=True)
    assert ctx_filter["facility"]["type_nominative"] == "котельная"
    assert ctx_filter["land_plot"]["area_sqm"] == "{{ land_plot.area_sqm }}"
    assert ctx_filter["structures"]["structures"] == ["{{ structures.structures }}"]


def test_pick_assembly_model_finds_dynamic_chapter1():
    model = pick_assembly_model("src.ecology_chapters.chapter1.assembly")
    assert issubclass(model, BaseModel)
    assert model.__name__ == "Chapter1Data"
