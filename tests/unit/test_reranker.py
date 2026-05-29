from pathlib import Path

from src.retrieval.qdrant import ProjectPart
from config.config_file import cfg

project_part = ProjectPart(
    file_path = Path(
        r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim\2_ОК.17.24СТ-ПЗУ.pdf"
    )
)
project_part.make_chunks()
chunks = project_part.make_chunks()
query = "Наименование объекта строительства"


def test_rerank_chunks():
    from src.retrieval.reranker import rerank_chunks

    reranked = rerank_chunks(query, chunks, cfg.RERANKER_MODEL, top_n=3)
    assert len(reranked) == 3


def test_local_rerank_chunks():
    from src.retrieval.reranker import rerank_with_local_reranker

    model_name = "qilowoq/bge-reranker-v2-m3-en-ru"
    reranked = rerank_with_local_reranker(
        model_name=model_name, query=query, chunks=chunks, top_n=3
    )
    assert len(reranked) == 3


def test_api_rerank_chunks():
    from src.retrieval.reranker import rerank_with_api

    model_name = "rerank-v3.5"
    reranked = rerank_with_api(
        model_name=model_name, query=query, chunks=chunks, top_n=3
    )
    assert len(reranked) == 3
