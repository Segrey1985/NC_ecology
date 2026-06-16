# ─────────────────────────────────────────────
# Корневая модель (автособирается из chapter6.models)
# ─────────────────────────────────────────────

from src.utils.utils import build_chapter_assembly_model

Chapter6Data = build_chapter_assembly_model(
    "src.ecology_chapters.chapter6.models",
    model_name="Chapter6Data",
)
