from numpy import ndarray
from pathlib import Path
from sentence_transformers import CrossEncoder

from config.config_file import cfg
from src.project_data.qdrant import ProjectPart, QdrantService, QdrantClient


# model_name = "BAAI/bge-reranker-base"
# model_name = "DiTy/cross-encoder-russian-msmarco"
# model_name = "BAAI/bge-reranker-v2-m3"
# model_name = "qilowoq/bge-reranker-v2-m3-en-ru"


def rerank_chunks(query: str, chunks: list[str]) -> list[str]:
    model = CrossEncoder(cfg.RERANKER_MODEL)
    pairs = [(query, chunk) for chunk in chunks]
    scores: ndarray = model.predict(pairs)
    reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return reranked


if __name__ == "__main__":
    project_part = ProjectPart(
        Path(
            r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim\2_ОК.17.24СТ-ПЗУ.pdf"
        )
    )
    project_part.run()
    
    chunks = project_part.chunks
    query = 'Наименование объекта'
    
    reranked = rerank_chunks(query, chunks)
    for r in reranked:
        print(r)
        
    