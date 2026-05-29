import os
import requests
import threading
from pathlib import Path
from numpy import ndarray
from functools import lru_cache
from sentence_transformers import CrossEncoder

from src.utils.logger import logger
from config.config_file import cfg, rerankers_list
from src.retrieval.qdrant import ProjectPart

# model_name = "BAAI/bge-reranker-base"
# model_name = "DiTy/cross-encoder-russian-msmarco"
# model_name = "BAAI/bge-reranker-v2-m3"
# model_name = "qilowoq/bge-reranker-v2-m3-en-ru"


def _load_local_reranker(reranker_model: str):
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


_reranker_init_lock = threading.Lock()


@lru_cache(maxsize=2)
def _get_local_reranker(reranker_model: str) -> CrossEncoder:
    with _reranker_init_lock:
        return _load_local_reranker(reranker_model)


def rerank_with_local_reranker(
    model_name: str,
    query: str,
    chunks: list[str],
    *,
    top_n: int | None = 5,
    batch_size: int = 32,
) -> list[dict]:

    if not chunks:
        return []

    model = _get_local_reranker(model_name)
    logger.debug(f"[reranker] Re-ranking start.")

    pairs = [(query, chunk) for chunk in chunks]
    scores: ndarray = model.predict(pairs, batch_size=batch_size)

    order = scores.argsort()[::-1]
    if top_n is not None:
        order = order[:top_n]
    
    reranked = [
        {
            "index": int(i),
            "text": chunks[i],
            "score": scores[i]
        }
        for i in order
    ]
    
    logger.debug(f"[local_reranker] Re-ranking complete.")

    return reranked


def rerank_with_api(
    model_name: str, query: str, chunks: list[str], *, top_n: int = 5
) -> list[dict]:
    response = requests.post(
        "https://api.aitunnel.ru/v1/rerank",
        headers={
            "Authorization": f"Bearer {cfg.ai_tunnel_api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model_name, "query": query, "documents": chunks, "top_n": top_n},
    )
    
    if response.status_code != 200:
        logger.error(f"{response.status_code}: {response.text}")
        return []
    
    response_json = response.json()
    
    if not response_json.get("results"):
        logger.error(f"[reranker] Отсутствует ключ 'results': {response}")
        return []
    
    reranked = [
        {
            "index": x["index"],
            "text": x["document"]["text"],
            "score": x["relevance_score"]
        }
        for x in response_json["results"]
    ]
    logger.debug(f"[api reranker] [{model_name}] Re-ranking complete.")
    return reranked


def rerank_chunks(
    query: str, chunks: list[str], reranker_model: str, *, top_n: int = 5
) -> list[dict]:
    """Делает rerank и возвращает список словарей с информацией о чанках с ключами index, text, score"""
    is_local = rerankers_list[reranker_model]["is_local"]
    if is_local:
        return rerank_with_local_reranker(model_name=reranker_model, query=query, chunks=chunks, top_n=top_n)
    else:
        return rerank_with_api(model_name=reranker_model, query=query, chunks=chunks, top_n=top_n)


if __name__ == "__main__":
    project_part = ProjectPart(
        Path(
            r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim\2_ОК.17.24СТ-ПЗУ.pdf"
        )
    )
    project_part.make_chunks()
    chunk_pairs = project_part.chunk_pairs
    
    query = "Наименование объекта строительства"

    reranked = rerank_chunks(query, list(map(lambda x: x["child_text"], chunk_pairs)), reranker_model="rerank-4-pro")
    for r in reranked:
        print(r)
