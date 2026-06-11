import json
import uuid
import threading
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.agent_base import GraphResources, init_graph
from src.agents.agent_base_aux import init_graph as init_aux_graph
from src.ecology_chapters.chapter0.calculations import add_calculated_placeholders

from config.langfuse_client import langfuse_config
from config.config_file import build_runtime_config, TestMode
from src.utils.logger import (
    add_output_log_file,
    logger,
    logger_file_format,
    print_and_save_thread_logs,
)
from src.utils.utils import print_chunk, is_valid_uuid4_hex, extract_project_parts_pdfs
from src.templates.docx_template_engine import fill_docx_template


def _load_placeholders(placeholders_path: Path, table_placeholders_path: Path | None):

    if table_placeholders_path:
        with open(
            table_placeholders_path, "r", encoding="utf-8"
        ) as table_placeholders_file:
            table_placeholders = json.load(table_placeholders_file)
    else:
        table_placeholders = {}

    with open(placeholders_path, "r", encoding="utf-8") as placeholders_file:
        placeholders = json.load(placeholders_file)

    placeholders = {
        k: v for k, v in placeholders.items() if k not in table_placeholders
    }
    return placeholders, table_placeholders


def _run_graph(
    graph, placeholder_info: dict, verbose: bool = True
) -> str:

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)

    final_content = ""
    for chunk in graph.stream(
        input={
            "for_rag_search": placeholder_info["for_rag_search"],
            "examples": placeholder_info["examples"],
            "question": placeholder_info["question"],
            "meta": placeholder_info.get("meta", {}),
        },
        stream_mode="updates",
        config=config,
    ):
        if verbose:
            print_chunk(chunk)

        if "answer_node" in chunk:
            final_output = chunk["answer_node"]
            final_content = final_output["answer"]

    return final_content


def _log_thread():
    log_lines = []
    thread_ident = threading.get_ident()

    def _only_this_thread(record) -> bool:
        return record["thread"].id == thread_ident

    def _capture_sink(message) -> None:
        log_lines.append(message.strip())

    handler_id = logger.add(
        _capture_sink,
        filter=_only_this_thread,
        format=logger_file_format,
        colorize=False,
    )
    return handler_id, log_lines


def thread_run_graph_for_placeholder(
    graph,
    placeholder: str,
    placeholder_info: dict,
    verbose: bool,
) -> dict:
    handler_id, log_lines = _log_thread()

    try:
        final_content = _run_graph(
            graph=graph,
            placeholder_info=placeholder_info,
            verbose=verbose,
        )
        return {
            "placeholder": placeholder,
            "result": json.loads(final_content).get("answer", "__empty__"),
            "logs_lines": log_lines,
        }
    finally:
        logger.remove(handler_id)


def add_auxiliary_placeholders(placeholders_chapter_0: dict, table_placeholders: dict, runtime_cfg):
    """Добавляет дополнительные плейсхолдеры после получения базовых сведений о проекте"""
    
    aux_graph, aux_resources = init_aux_graph(runtime_cfg=runtime_cfg)
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)
    result = aux_graph.invoke(
        input={
            "chapter_0": json.dumps(placeholders_chapter_0, ensure_ascii=False),
            "table_placeholders": table_placeholders
        },
        config=config,
    )
    
    aux_answer: dict = json.loads(result['answer'])
    main_answer: dict = placeholders_chapter_0
    
    # проверка на отсутствие одинаковых ключей
    aux_answer_set = set(aux_answer.keys())
    main_answer_set = set(main_answer.keys())
    similar_keys = aux_answer_set.intersection(main_answer_set)
    
    if len(similar_keys) != 0:
        raise AssertionError(
            f"Требуется устранить одинаковые ключи в agent_base и agent_base_aux: {similar_keys}")
    
    placeholders_chapter_0.update(aux_answer)


def main(
    template_docx_path: Path | None,
    placeholders_path: Path,
    table_placeholders_path: Path | None,
    project_parts_zip: bytes | Path | None,
    output_path: Path | None,
    collection_name: str = "main",
    verbose: bool = True,
    test_mode: TestMode = "on",
    max_workers: int | None = None,
    **kwargs
) -> dict:
    
    resources: GraphResources | None = None
    output_log_handler_id: int | None = None
    project_parts_tmp: tempfile.TemporaryDirectory | None = None
    if output_path:
        output_log_handler_id = add_output_log_file(output_path)

    try:
        project_parts_path: Path | None = None
        if project_parts_zip is not None:
            project_parts_tmp = tempfile.TemporaryDirectory(prefix="project_parts_")
            project_parts_path = Path(project_parts_tmp.name)
            extract_project_parts_pdfs(project_parts_zip, project_parts_path)

        runtime_cfg = build_runtime_config(test_mode)

        graph, resources = init_graph(
            collection_name=collection_name,
            project_parts_path=project_parts_path,
            runtime_cfg=runtime_cfg,
        )

        placeholders, table_placeholders = _load_placeholders(
            placeholders_path, table_placeholders_path
        )

        if test_mode == "mock":
            placeholders_output = json.load(
                open(
                    "data/mock/chapter0.json",
                    encoding="utf-8",
                )
            )
        else:
            placeholders_output = {}
            total_results: list[dict] = []

            for key, value in table_placeholders.items():
                placeholders_output[key] = value

            placeholder_items = list(placeholders.items())
            if test_mode == "on":
                placeholder_items = placeholder_items[:1]

            if max_workers is None:
                max_workers = min(4, max(1, len(placeholder_items)))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        thread_run_graph_for_placeholder,
                        graph=graph,
                        placeholder=placeholder,
                        placeholder_info=placeholder_info,
                        verbose=verbose,
                    )
                    for placeholder, placeholder_info in placeholder_items
                ]

                for future in as_completed(futures):
                    dct = future.result()
                    placeholders_output[dct["placeholder"]] = dct["result"]
                    total_results.append(dct)

            print_and_save_thread_logs(output_path, total_results, "placeholder")

            add_auxiliary_placeholders(
                placeholders_chapter_0=placeholders_output,
                table_placeholders=table_placeholders,
                runtime_cfg=runtime_cfg
            )

            add_calculated_placeholders(placeholders=placeholders_output)

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

            placeholders_out_path = output_path / "placeholders.json"
            with open(placeholders_out_path, "w", encoding="utf-8") as f:
                json.dump(placeholders_output, f, ensure_ascii=False, indent=4)

            if template_docx_path:
                result_template_out_path = output_path / "result_template.docx"
                fill_docx_template(
                    template_path=template_docx_path,
                    data=placeholders_output,
                    output_docx_path=result_template_out_path,
                )

        return placeholders_output

    finally:
        if project_parts_tmp is not None:
            project_parts_tmp.cleanup()
        if output_log_handler_id is not None:
            logger.remove(output_log_handler_id)
        if "save_db" not in kwargs:
            qdrant_service = getattr(resources, "qdrant_service", None)
            if (
                qdrant_service
                and qdrant_service.client.collection_exists(collection_name)
                and is_valid_uuid4_hex(collection_name)
            ):
                qdrant_service.client.delete_collection(collection_name)
                logger.info(
                    f"collection <{collection_name}> name is valid uuid and was deleted"
                )


if __name__ == "__main__":

    base = Path(__file__).parent
    input_dir = base / "src" / "ecology_chapters" / "chapter0"
    main(
        template_docx_path=input_dir / "template.docx",
        placeholders_path=input_dir / "placeholders.json",
        table_placeholders_path=input_dir / "table_placeholders.json",
        project_parts_zip=Path(r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\project_parts.zip"),
        output_path=base / "data" / "OUT" / "project1",
        collection_name="main_base_test_off_parent",
        test_mode="off",
    )
