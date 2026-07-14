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
        """Создать Qdrant коллекцию"""
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
        """Добавить в коллекцию список рассчитанных точек PointStruct"""
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, wait=True, points=batch)

    def _calculate_vectors(self, chunks: list[str]) -> list[list[float]]:
        """Превращает список чанков в список эмбеддингов"""
        if not chunks:
            raise ValueError("Cannot calculate vectors: chunks list is empty")

        vectors = self.model.encode(chunks)
        return [
            vector if isinstance(vector, list) else vector.tolist()
            for vector in vectors
        ]

    def _build_payload(self, project_part: "ProjectPart") -> list[dict]:
        """Создание payload для всех чанков экземпляра ProjectPart"""
        if not project_part.chunk_pairs:
            raise ValueError(
                "Cannot build payload: ProjectPart.chunk_pairs list is empty"
            )

        part_number_and_part_name: dict = self._build_part_payload(project_part.file_path)
        return [
            {
                **part_number_and_part_name,
                "text": pair["child_text"],
                "parent_text": pair["parent_text"],
                "parent_id": pair["parent_id"]
            }
            for pair in project_part.chunk_pairs
        ]

    def _build_part_payload(self, file_path: Path) -> dict:
        """Вспомогательная функция для генерации ключей part_number, part_name"""
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
        """
        Сборная функция.
        На основе project_part вычисляет эмбеддинги, добавляет payload, формирует и возвращает список PointStruct.
        """
        logger.debug(f"Calculating points for <{project_part.file_path}> ...")
        # Считаем эмбеддинги только по дочерним текстам
        child_texts = [pair["child_text"] for pair in project_part.chunk_pairs]
        vectors = self._calculate_vectors(child_texts)
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
        self.chunk_pairs: list[dict] = []

    def extract_text(self) -> str:
        texts_by_page = extract_text_with_miner_coords(self.file_path)
        return " ".join(texts_by_page)

    def make_chunks(
        self,
        text: str | None = None,
        child_size=600,
        child_overlap=150,
        parent_size=3000,
        parent_overlap=500,
    ) -> None:

        text = text or self.extract_text()
        if not text:
            raise ValueError("Can't make chunks: extracted text is empty")
            
        # 1. Сплиттер для Родительских чанков
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            is_separator_regex=True,
            separators=["\n\n", r"[\.\:\;] (?=[А-ЯЁA-Z])", "\n", " ", ""],
            keep_separator="end",
        )
        
        # 2. Сплиттер для Дочерних чанков
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            is_separator_regex=True,
            separators=["\n\n", r"[\.\:\;] (?=[А-ЯЁA-Z])", "\n", " ", ""],
            keep_separator="end",
        )
        
        parent_chunks = parent_splitter.split_text(text)
        self.chunk_pairs = []
        
        # 3. Двухуровневое разделение
        for p_text in parent_chunks:
            if not p_text or len(p_text.strip()) <= 20:
                continue
                
            # Генерируем уникальный ID для родительского чанка
            parent_id = str(uuid.uuid4())
            
            # Делим текущего родителя на детей
            child_chunks = child_splitter.split_text(p_text)
            
            for c_text in child_chunks:
                if c_text and len(c_text.strip()) > 20:
                    self.chunk_pairs.append({
                        "child_text": c_text,
                        "parent_text": p_text,
                        "parent_id": parent_id
                    })
    
    
    def __repr__(self):
        return json.dumps(
            {"file": self.file_path.as_posix(), "chunks_count": len(self.chunk_pairs)},
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
        file_path=cfg.BASE_DIR / "data" / "IN" / "project1" / "trim" /"1_ОК.17.24СТ-ПЗ.pdf"
    )

    pp.make_chunks()

    # for chunk in pp.chunk_pairs:
    #     print(f"{chunk["child_text"]=}")
    #     print(f"{chunk["parent_text"]=}")
    #     print(f"{chunk["parent_id"]=}")
    #     print(f"------------------------------")
    
    from config.config_file import cfg
    
    qdrant_service = build_qdrant_service(cfg)
    create_collection(qdrant_service, "parent1")
    fill_collection(qdrant_service, "parent1", [pp])
    