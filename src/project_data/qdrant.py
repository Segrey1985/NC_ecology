import json
from pathlib import Path
from typing import Optional
from numpy import ndarray

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config_file import cfg
from src.utils.logger import logger
from src.utils.utils import extract_text_with_miner_coords
from src.project_data.embeddings import load_embeddings

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


def build_qdrant_service() -> QdrantService:
    client = QdrantClient(url="http://localhost:6333")
    model = load_embeddings(cfg.EMBEDDINGS_MODEL_NAME)
    return QdrantService(client=client, model=model)


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
        self.payload = self._build_payload()
        self.texts_by_page: Optional[list[str]] = None
        self.text: Optional[str] = None
        self.chunks: Optional[list[str]] = None
        self.vectors: Optional[list[ndarray]] = None

    def _build_payload(self) -> dict:
        payload = {}
        self._payload_add_part(payload)
        return payload

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

    def extract_text(self) -> None:
        self.texts_by_page = extract_text_with_miner_coords(self.file_path)
        self.text = " ".join(self.texts_by_page)

    def make_chunks(self) -> None:
        if not self.text:
            raise ValueError("Cannot make chunks: ProjectPart.text is empty")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            is_separator_regex=True,
            # Используем Lookahead: (?=[А-ЯЁA-Z])
            # Это значит: "найти точку с пробелом, если за ними идет заглавная буква"
            separators=["\n\n", r"\. (?=[А-ЯЁA-Z])"],
            keep_separator='end'
        )
        chunks = text_splitter.split_text(self.text)
        self.chunks = chunks

    def calculate_vectors(self) -> None:
        if not self.chunks:
            raise ValueError("Cannot calculate vectors: ProjectPart.chunks list is empty")
        model = load_embeddings(cfg.EMBEDDINGS_MODEL_NAME)
        vectors = []
        for chunk in self.chunks:
            vector = model.encode(chunk)
            vectors.append(vector)
        self.vectors = vectors

    def __repr__(self):
        return json.dumps(
            {"file": self.file_path.as_posix(), "payload": self.payload},
            ensure_ascii=False,
        )


def collect_project_parts(folder_path: Path) -> list[ProjectPart]:
    project_parts = []
    for file in folder_path.iterdir():
        if file.suffix == ".pdf":
            project_parts.append(ProjectPart(file_path=file))
    return project_parts


if __name__ == "__main__":

    # test build_qdrant_service()

    # qdrant_service = build_qdrant_service()
    # COLLECTION_NAME = "project_data"
    # qdrant_service.create_collection(collection_name=COLLECTION_NAME)

    # test ProjectPart and collect_project_parts

    p = ProjectPart(
        Path(
            r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim\2_ОК.17.24СТ-ПЗУ.pdf"
        )
    )
    print(p)
    print(p.texts_by_page)
    p.extract_text()
    print(p.texts_by_page)
    print(len(p.texts_by_page))
    
    # p.make_chunks()
    # for ch in p.chunks:
    #     print(ch, end="\n\n==========================\n\n")
    
    # p.calculate_vectors()
    # print(p.vectors)

    # parts = collect_project_parts(Path(r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim"))
    # for p in parts:
    #     print(p)

    pass
