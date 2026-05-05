import uuid
from pathlib import Path

from src.project_data.qdrant import ProjectPart, build_qdrant_service


def test_qdrant_minibase_from_trim_first3_and_cleanup():
    base = Path(__file__).resolve().parents[2]
    trim_dir = base / "data" / "IN" / "project1" / "trim"

    pdfs = sorted(trim_dir.glob("*.pdf"))[:3]
    assert len(pdfs) == 3, f"Ожидалось 3 pdf в {trim_dir}, найдено: {len(pdfs)}"

    qdrant_service = build_qdrant_service()
    collection_name = f"trim_first3_{uuid.uuid4().hex}"

    try:
        if qdrant_service.client.collection_exists(collection_name):
            qdrant_service.client.delete_collection(collection_name)

        qdrant_service.create_collection(collection_name=collection_name)

        for pdf_path in pdfs:
            part = ProjectPart(pdf_path)
            part.run()
            qdrant_service.add_points_to_collection(
                collection_name=collection_name,
                points=part.points,
            )

        results = qdrant_service.run_query(
            query="Как называется проект?",
            collection_name=collection_name,
            limit=5,
        )
        assert len(results) > 0

    finally:
        if qdrant_service.client.collection_exists(collection_name):
            qdrant_service.client.delete_collection(collection_name)
