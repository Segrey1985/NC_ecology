import os
from pathlib import Path
from numpy import ndarray
from functools import lru_cache
from sentence_transformers import CrossEncoder

from src.utils.logger import logger
from config.config_file import cfg
from src.retrieval.qdrant import ProjectPart

# model_name = "BAAI/bge-reranker-base"
# model_name = "DiTy/cross-encoder-russian-msmarco"
# model_name = "BAAI/bge-reranker-v2-m3"
# model_name = "qilowoq/bge-reranker-v2-m3-en-ru"


def load_local_reranker(reranker_model: str):
    models_dir = cfg.BASE_DIR / "data" / "__local_models"
    model_name_str = reranker_model.replace("/", "_")
    model_path = models_dir / model_name_str
    device = cfg.DEVICE
    if os.path.exists(model_path):
        logger.debug(f"[reranker] Loading local model from {model_path}...")
        reranker = CrossEncoder(model_path.as_posix(), device=device)
    else:
        logger.debug(f"[reranker] Loading model from HuggingFace...")
        reranker = CrossEncoder(reranker_model, device=device)
        reranker.save(
            model_path
        )  # внутри метода save есть makedirs(..., exist_ok=True)
    logger.debug(f"[reranker] Model loading complete. Device: {device}")
    return reranker


@lru_cache(maxsize=2)
def get_reranker(reranker_model: str) -> CrossEncoder:
    return load_local_reranker(reranker_model)


def rerank_chunks(query: str, chunks: list[str]) -> list[str]:
    model = get_reranker(cfg.RERANKER_MODEL)
    logger.debug(f"[reranker] Re-ranking start.")
    pairs = [(query, chunk) for chunk in chunks]
    scores: ndarray = model.predict(pairs)
    reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    logger.debug(f"[reranker] Re-ranking complete.")
    return reranked


if __name__ == "__main__":
    project_part = ProjectPart(
        Path(
            r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim\2_ОК.17.24СТ-ПЗУ.pdf"
        )
    )
    project_part.run()

    chunks = project_part.chunks
    query = "Наименование объекта"

    reranked = rerank_chunks(query, chunks)
    for r in reranked:
        print(r)
