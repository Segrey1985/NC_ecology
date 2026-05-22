import json
import uuid
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from qdrant_client.models import Filter, FieldCondition, MatchAny

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config_file import cfg, embeddings_list, Config
from src.utils.logger import logger
from src.utils.utils import extract_text_with_miner_coords
from src.retrieval.embeddings import load_local_embedder, init_openai_embedder


@lru_cache(maxsize=None)
def init_embedder(model_name: str):
    embedding_model = embeddings_list.get(model_name)
    
    if not embedding_model:
        raise ValueError(f"{model_name} отсутствует в embeddings_list.")
    
    if embedding_model['is_local'] is True:
        embedder = load_local_embedder(model_name)
    else:
        embedder = init_openai_embedder(model_name)
    return embedder

# _______________ QdrantService _______________


class QdrantService:
    """Класс для создания коллекций, добавления в них точек и поиска похожих точек."""

    DISCIPLINE_BY_NUMBER = cfg.DISCIPLINE_BY_NUMBER

    def __init__(self, client: QdrantClient, model: SentenceTransformer):
        self.client = client
        self.model = model
        self.vector_size: int | None = None

    def create_collection(self, collection_name: str) -> None:
        if not self.client.collection_exists(collection_name):
            if self.vector_size is None:
                try:
                    self.vector_size = len(self.model.encode(["test"])[0])
                except Exception:
                    logger.exception(
                        "[qdrant] Failed to determine vector size from embedder."
                    )
                    raise RuntimeError(
                        "Не удалось определить размер эмбеддингов для создания коллекции Qdrant. "
                        "Проверьте настройки EMBEDDINGS (локальные/внешние) и доступность API."
                    )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection <{collection_name}> has been created.")

    def add_points_to_collection(
        self, collection_name: str, points: list[PointStruct], batch_size=32
    ) -> None:
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, wait=True, points=batch)

    def _calculate_vectors(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            raise ValueError("Cannot calculate vectors: chunks list is empty")

        vectors = self.model.encode(chunks)
        return [
            vector if isinstance(vector, list) else vector.tolist()
            for vector in vectors
        ]

    def _build_payload(self, project_part: "ProjectPart") -> list[dict]:
        if not project_part.chunks:
            raise ValueError(
                "Cannot build payload: ProjectPart.chunks list is empty"
            )

        base = self._build_part_payload(project_part.file_path)
        return [{**base, "text": chunk} for chunk in project_part.chunks]

    def _build_part_payload(self, file_path: Path) -> dict:
        stem = file_path.stem
        part_ = stem.split("_")[0]
        parts_split_by_point = part_.split(".")
        if len(parts_split_by_point) == 1:
            part_number = parts_split_by_point[0]
        else:
            part_number = parts_split_by_point[0] + "." + parts_split_by_point[1]

        return {
            "part_number": part_number,
            "part_name": self.DISCIPLINE_BY_NUMBER.get(
                part_number, self.DISCIPLINE_BY_NUMBER["прочее"]
            ),
        }

    def calculate_points(self, project_part: "ProjectPart") -> list[PointStruct]:
        """Сборная функция. Вычисляет эмбеддинги, добавляет payload, формирует и возвращает список PointStruct."""
        logger.debug(f"Calculating points for <{project_part.file_path}> ...")
        vectors = self._calculate_vectors(project_part.chunks)
        payload = self._build_payload(project_part)
        return [
            PointStruct(id=uuid.uuid4().hex, vector=vector, payload=payload_item)
            for payload_item, vector in zip(payload, vectors)
        ]
    
    # part_names = ['ПЗ', 'ПЗУ', 'АР', 'КР', 'ИОС', 'Система электроснабжения', 'Система водоснабжения', 'Система водоотведения',
    #  'Отопление, вентиляция и кондиционирование воздуха, тепловые сети', 'Сети связи', 'Система газоснабжения']
    
    def run_query(
        self,
        query: str,
        collection_name: str,
        limit: int = 3,
        part_names: list[str] | None = None,
    ) -> list[ScoredPoint]:
        """
        Выполняет векторный поиск по коллекции.

        Примечание: если эмбеддинги не удаётся получить (например, из-за проблем с внешним API),
        возвращаем пустой результат. Это позволяет пайплайну корректно отработать дальше (с пустым контекстом)
        и показать причину.
        """
        try:
            vector = self.model.encode([query])
            parts_filter = None
            if part_names:
                parts_filter = Filter(
                    must=[
                        FieldCondition(
                            key="part_name",
                            match=MatchAny(any=part_names)
                        )
                    ]
                )
            
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=vector[0],
                limit=limit,
                query_filter=parts_filter,
            ).points
            return search_result
        except UnexpectedResponse as exc:
            if "vector dimension error" in str(exc).lower():
                logger.error(f"[qdrant] Vector dimension error. collection={collection_name}")
            raise
        except Exception:
            logger.exception(
                "[qdrant] Failed to encode/query. Returning empty search result. "
                f"collection={collection_name}, query={query!r}"
            )
            return []


def build_qdrant_service(runtime_cfg: Config) -> QdrantService:
    """Создает и возвращает QdrantService, умеющий создавать коллекции и добавлять туда точки"""
    client = QdrantClient(url=runtime_cfg.QDRANT_URL)
    embedder = init_embedder(runtime_cfg.EMBEDDINGS_MODEL_NAME)
    return QdrantService(client=client, model=embedder)


# _______________ ProjectPart _______________


class ProjectPart:
    """Класс описывающий смежный раздел проектной документации (АР, КР, ИОС ...)"""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.chunks: Optional[list[str]] = None

    def extract_text(self) -> str:
        texts_by_page = extract_text_with_miner_coords(self.file_path)
        return " ".join(texts_by_page)

    def make_chunks(
        self,
        text: str | None = None,
        chunk_size=750,
        chunk_overlap=150,
    ) -> None:
        text = text or self.extract_text()
        if not text:
            raise ValueError("Can't make chunks: extracted text is empty")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
            # Используем Lookahead: (?=[А-ЯЁA-Z])
            # "найти точку с пробелом, если за ними идет заглавная буква"
            separators=["\n\n", r"[\.\:\;] (?=[А-ЯЁA-Z])", "\n", " ", ""],
            keep_separator="end",
        )
        chunks = text_splitter.split_text(text)
        # пропускаем пустые и очень короткие чанки
        filtered_chunks = list(filter(lambda x: x and len(x.strip()) > 20, chunks))
        self.chunks = filtered_chunks

    def __repr__(self):
        return json.dumps(
            {"file": self.file_path.as_posix(), "chunks": self.chunks},
            ensure_ascii=False,
        )


# _______________ вспомогательные функции _______________


def _collect_project_parts(folder_path: Path) -> list[ProjectPart]:
    """Ищет все файлы .pdf в директории и превращает их в list[ProjectPart]"""
    project_parts = []
    for file in folder_path.iterdir():
        if file.suffix == ".pdf":
            project_parts.append(ProjectPart(file_path=file))
    return project_parts


def create_project_parts(project_parts_path: Path) -> list[ProjectPart]:
    """Собирает все .pdf файлы в директории, превращает их в list[ProjectPart] и нарезает на чанки."""
    project_parts: list[ProjectPart] = _collect_project_parts(project_parts_path)

    def _make_chunks(part: ProjectPart) -> None:
        part.make_chunks()
        logger.debug(f"{part.file_path.name} chunks complete.")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_make_chunks, part) for part in project_parts]
        for f in as_completed(futures):
            f.result()  # пробрасываем исключения

    return project_parts


def create_collection(qdrant_service: QdrantService, collection_name: str) -> None:
    """Создает коллекцию collection_name в указанном qdrant_service"""
    qdrant_service.create_collection(collection_name=collection_name)


def fill_collection(
    qdrant_service: QdrantService,
    collection_name: str,
    project_parts: list[ProjectPart],
) -> None:
    """Параллельно считает PointStruct и добавляет готовые части в коллекцию."""

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(qdrant_service.calculate_points, part): part
            for part in project_parts
        }
        for future in as_completed(futures):
            project_part = futures[future]
            points = future.result()  # пробрасываем исключения
            qdrant_service.add_points_to_collection(
                collection_name=collection_name,
                points=points,
            )
            logger.debug(f"Project part '{project_part.file_path.name}' was added to Qdrant.")


if __name__ == "__main__":

    pp = ProjectPart(
        file_path=cfg.BASE_DIR / "data" / "IN" / "project1" / "1_ОК.17.24СТ-ПЗ.pdf"
    )
    print(pp)
    pp.make_chunks()

    for chunk in pp.chunks:
        print(chunk)
        print(f"---------------  {len(chunk)}  ---------------")
