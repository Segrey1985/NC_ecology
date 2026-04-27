from pathlib import Path
from src.project_data.qdrant import ProjectPart, build_qdrant_service


def test_qdrant_service():
    project_part = ProjectPart(
        Path(
            r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim\2_ОК.17.24СТ-ПЗУ.pdf"
        )
    )
    project_part.run()
    
    qdrant_service = build_qdrant_service()
    COLLECTION_NAME = "test_collection"
    qdrant_service.create_collection(collection_name=COLLECTION_NAME)
    
    qdrant_service.add_points_to_collection(
        collection_name=COLLECTION_NAME,
        points=project_part.points,
    )