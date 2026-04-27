import json
import uuid
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config_file import cfg
from src.utils.logger import logger
from src.utils.utils import extract_text_with_miner_coords
from src.project_data.embeddings import load_local_embedder, init_openai_embedder

if cfg.EMBEDDINGS_LOCAL:
    embedder = load_local_embedder(cfg.EMBEDDINGS_MODEL_PATH)
else:
    embedder = init_openai_embedder()


# _______________ QdrantService _______________


class QdrantService:

    def __init__(self, client: QdrantClient, model: SentenceTransformer):
        self.client = client
        self.model = model
        self.vector_size = len(model.encode(["test"])[0])

    def create_collection(self, collection_name: str) -> None:
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection {collection_name}.")

    def add_points_to_collection(
        self, collection_name: str, points: list[PointStruct]
    ) -> None:
        self.client.upsert(collection_name=collection_name, wait=True, points=points)
    
    def run_query(self, query: str, collection_name: str, limit: int = 3):
        vector = self.model.encode([query])
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=vector[0],
            limit=limit,
        ).points
        return search_result

def build_qdrant_service() -> QdrantService:
    """Создает и возвращает QdrantService, умеющий создавать коллекции и добавлять туда точки"""
    client = QdrantClient(url="http://localhost:6333")
    return QdrantService(client=client, model=embedder)


# _______________ ProjectPart _______________


class ProjectPart:

    NAME_BY_NUMBER = {
        "1": "ПЗ",
        "2": "ПЗУ",
        "3": "АР",
        "4": "OK",
        "5": "ИОС",
        "5.1": "Система электроснабжения",
        "5.2": "Система водоснабжения",
        "5.3": "Система водоотведения",
        "5.4": "Отопление, вентиляция и кондиционирование воздуха, тепловые сети",
        "5.5": "Сети связи",
        "5.6": "Система газоснабжения",
    }

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.texts_by_page: Optional[list[str]] = None
        self.text: Optional[str] = None
        self.chunks: Optional[list[str]] = None
        self.vectors: Optional[list[list[float]]] = None
        self.payload: Optional[list[dict]] = None
        self.points: Optional[list[PointStruct]] = None

    def extract_text(self) -> None:
        self.texts_by_page = extract_text_with_miner_coords(self.file_path)
        self.text = " ".join(self.texts_by_page)

    def make_chunks(self) -> None:
        if not self.text:
            raise ValueError("Can't make chunks: ProjectPart.text is empty")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            is_separator_regex=True,
            # Используем Lookahead: (?=[А-ЯЁA-Z])
            # Это значит: "найти точку с пробелом, если за ними идет заглавная буква"
            separators=["\n\n", r"\. (?=[А-ЯЁA-Z])"],
            keep_separator="end",
        )
        chunks = text_splitter.split_text(self.text)
        self.chunks = chunks

    def calculate_vectors(self) -> None:
        if not self.chunks:
            raise ValueError(
                "Cannot calculate vectors: ProjectPart.chunks list is empty"
            )
        vectors = embedder.encode(self.chunks)
        vectors = [
            vector if isinstance(vector, list) else vector.tolist()
            for vector in vectors
        ]
        self.vectors = vectors

    def build_payload(self) -> None:
        base = {}
        self._payload_add_part(base)
        self.payload = self._payload_add_text(base)  # теперь list[dict]
    
    def _payload_add_part(self, payload) -> None:
        stem = self.file_path.stem
        part_ = stem.split("_")[0]
        parts_split_by_point = part_.split(".")
        if len(parts_split_by_point) == 1:
            part_number = parts_split_by_point[0]
        else:
            part_number = parts_split_by_point[0] + "." + parts_split_by_point[1]
        payload["part_number"] = part_number
        payload["part_name"] = self.NAME_BY_NUMBER[part_number]
    
    
    def _payload_add_text(self, base: dict) -> list[dict]:
        return [{**base, "text": chunk} for chunk in self.chunks]
    
    
    def calculate_points(self) -> None:
        if not self.vectors:
            raise ValueError("Cannot calculate points: ProjectPart.vectors list is empty")
        self.points = [
            PointStruct(id=uuid.uuid4().hex, vector=vector, payload=payload)
            for payload, vector in zip(self.payload, self.vectors)
        ]

    def run(self) -> None:
        self.extract_text()
        self.make_chunks()
        self.build_payload()
        self.calculate_vectors()
        self.calculate_points()

    def __repr__(self):
        return json.dumps(
            {"file": self.file_path.as_posix(), "payload": self.payload},
            ensure_ascii=False,
        )


def collect_project_parts(folder_path: Path) -> list[ProjectPart]:
    """Ищет все файлы .pdf в директории и превращает их в list[ProjectPart]"""
    project_parts = []
    for file in folder_path.iterdir():
        if file.suffix == ".pdf":
            project_parts.append(ProjectPart(file_path=file))
    return project_parts


if __name__ == "__main__":
    pass
            
        

