# ─────────────────────────────────────────────
# Корневая модель (автособирается из chapter1.models)
# ─────────────────────────────────────────────

from src.utils.utils import build_chapter_assembly_model

Chapter1Data = build_chapter_assembly_model(
    "src.ecology_chapters.chapter1.models",
    model_name="Chapter1Data",
)
