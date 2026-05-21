from pathlib import Path
from src.retrieval.qdrant import ProjectPart, build_qdrant_service
from src.retrieval.embeddings import init_openai_embedder
from config.config_file import build_runtime_config

def test_qdrant_service():
    project_part = ProjectPart(
        Path(
            r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim\2_ОК.17.24СТ-ПЗУ.pdf"
        ),
        embedder=init_openai_embedder("text-embedding-3-small")
    )
    project_part.run()

    qdrant_service = build_qdrant_service(runtime_cfg=build_runtime_config('on'))
    COLLECTION_NAME = "test_data"
    qdrant_service.create_collection(collection_name=COLLECTION_NAME)

    qdrant_service.add_points_to_collection(
        collection_name=COLLECTION_NAME,
        points=project_part.points,
    )


def test_recreate_collection():
    """build_qdrant_service and recreate collection"""
    qdrant_service = build_qdrant_service(runtime_cfg=build_runtime_config('on'))
    COLLECTION_NAME = "test_data"
    if qdrant_service.client.collection_exists(COLLECTION_NAME):
        qdrant_service.client.delete_collection(COLLECTION_NAME)
    qdrant_service.create_collection(collection_name=COLLECTION_NAME)


def test_print_points():
    """print ProjectPart point"""
    project_part = ProjectPart(
        Path(
            r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim\2_ОК.17.24СТ-ПЗУ.pdf"
        ),
        embedder=init_openai_embedder("text-embedding-3-small")
    )
    project_part.run()
    points, chunks = project_part.points, project_part.chunks
    for i, x in enumerate(points):
        print(x)
        if i == 2:
            break


def test_run_query():
    """create collection and find relevant points"""
    project_part = ProjectPart(
        Path(
            r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim\2_ОК.17.24СТ-ПЗУ.pdf"
        ),
        embedder=init_openai_embedder("text-embedding-3-small")
    )
    project_part.run()

    qdrant_service = build_qdrant_service(runtime_cfg=build_runtime_config('on'))
    COLLECTION_NAME = "test_data"
    if qdrant_service.client.collection_exists(COLLECTION_NAME):
        qdrant_service.client.delete_collection(COLLECTION_NAME)
    qdrant_service.create_collection(collection_name=COLLECTION_NAME)
    qdrant_service.add_points_to_collection(
        collection_name=COLLECTION_NAME,
        points=project_part.points,
    )
    result = qdrant_service.run_query(
        query="Адрес объекта", collection_name=COLLECTION_NAME, limit=5
    )
    for r in result:
        print(r)
        print()
