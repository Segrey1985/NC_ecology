# ─────────────────────────────────────────────
# Корневая модель (автособирается из chapter2.models)
# ─────────────────────────────────────────────

from src.utils.utils import build_chapter_assembly_model

Chapter2Data = build_chapter_assembly_model(
    "src.ecology_chapters.chapter2.models",
    model_name="Chapter2Data",
)
