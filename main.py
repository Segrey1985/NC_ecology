import json
import uuid
import traceback
import threading
import tempfile
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.agent import GraphResources, init_graph
from config.config_file import build_runtime_config, TestMode
from config.langfuse_client import langfuse_config
from src.utils.logger import (
    add_output_log_file,
    logger,
    logger_file_format,
    print_and_save_thread_logs,
)
from src.utils.utils import (
    print_chunk,
    is_valid_uuid4_hex,
    extract_project_parts_pdfs,
    iter_models_from_module,
    iter_chapter_models,
    pick_assembly_model,
    assembly_results_to_docx_context,
    pascal_to_snake,
    build_input_query,
)
from src.templates.docx_template_engine import fill_docx_template


def _run_graph(
    graph,
    input_query: str,
    output_model: type[BaseModel],
    chapter_module_path: str,
    verbose: bool = True,
) -> str:

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "output_model": output_model,
        },
        "metadata": {
            "output_model": output_model.__name__,
            "chapter_module_path": chapter_module_path,
        },
    }
    config.update(langfuse_config)

    final_content = ""
    for chunk in graph.stream(
        input={
            "input_query": input_query,
        },
        stream_mode="updates",
        config=config,
    ):
        if verbose:
            print_chunk(chunk, logger)

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


def thread_run_graph_for_model(
    graph: CompiledStateGraph, model: type[BaseModel], chapter_module_path: str, verbose: bool
) -> dict:
    handler_id, log_lines = _log_thread()

    try:
        final_content = _run_graph(
            graph,
            input_query=build_input_query(model),
            output_model=model,
            chapter_module_path=chapter_module_path,
            verbose=verbose,
        )
        return {
            "model": model,
            "model_name": model.__name__,
            "result": json.loads(final_content) if final_content else {},
            "logs_lines": log_lines,
        }
    finally:
        logger.remove(handler_id)


def main(
    template_docx_path: Path | None,
    project_parts_zip: bytes | Path | None,
    table_placeholders_path: Path | None,
    output_path: Path,
    chapter_module_path: str,
    collection_name: str = "main",
    verbose: bool = True,
    test_mode: TestMode = "on",
    max_workers: int | None = None,
    **kwargs
):
    
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

        chapter_name = chapter_module_path.split('.')[-1]
        runtime_cfg = build_runtime_config(test_mode)

        # init graph

        graph, resources = init_graph(
            collection_name=collection_name,
            project_parts_path=project_parts_path,
            runtime_cfg=runtime_cfg,
        )

        total_results: list[dict] = []

        if test_mode == "mock":
            results = json.load(
                open(
                    f"data/mock/{chapter_name}.json",
                    encoding="utf-8",
                )
            )
        else:

            # get models from module

            models_module_path = chapter_module_path + ".models"
            if test_mode == "filter":
                models = iter_chapter_models(chapter_module_path)
                if not models:
                    raise RuntimeError(
                        f"Не нашёл pydantic-моделей в модуле `{models_module_path}` "
                        f"(см. `{chapter_module_path}.debug_models`)."
                    )
            else:
                models = iter_models_from_module(models_module_path)
                if not models:
                    raise RuntimeError(
                        f"Не нашёл pydantic-моделей в модуле `{models_module_path}`."
                    )
                if test_mode == "on":
                    models = models[:1]

            # run thread_run_graph_for_model in ThreadPoolExecutor

            if max_workers is None:
                max_workers = min(4, max(1, len(models)))

            results: dict[str, object] = {
                pascal_to_snake(model.__name__): None for model in models
            }

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        thread_run_graph_for_model,
                        graph=graph,
                        model=model,
                        chapter_module_path=chapter_module_path,
                        verbose=verbose,
                    )
                    for model in models
                ]

                for future in as_completed(futures):
                    dct = future.result()
                    results[pascal_to_snake(dct["model_name"])] = dct["result"]
                    total_results.append(dct)

        # print results

        print_and_save_thread_logs(output_path, total_results, "model_name")

        # export results

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

            assembly_module_path = chapter_module_path + ".assembly"
            assembly_model = pick_assembly_model(assembly_module_path)
            data_dict = assembly_results_to_docx_context(
                assembly_model=assembly_model,
                results_dict=results,
            )

            if table_placeholders_path:
                try:
                    with open(table_placeholders_path, "r", encoding="utf-8") as f:
                        table_placeholders: dict = json.load(f)
                        logger.info(f"Плейсхолдеры дополнены из {table_placeholders_path}:\n{list(table_placeholders)}")
                        data_dict.update(table_placeholders)
                except Exception:
                    logger.error(traceback.format_exc())

            results_out_path = output_path / f"{chapter_name}_output.json"
            with open(results_out_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)

            if template_docx_path:
                result_template_out_path = (
                    output_path / f"{chapter_module_path.split('.')[-1]}.docx"
                )
                fill_docx_template(
                    template_path=template_docx_path,
                    data=data_dict,
                    output_docx_path=result_template_out_path,
                )
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
    logger.info("\n\n ЗАВЕРШЕНО \n\n")


if __name__ == "__main__":

    base = Path(__file__).parent
    chapter_n = 6
    main(
        template_docx_path=Path(f"src/ecology_chapters/chapter{chapter_n}/template.docx"),
        project_parts_zip=None,
        table_placeholders_path=Path(rf"C:\Users\maxfi\PycharmProjects\NC_ecology\src\ecology_chapters\chapter{chapter_n}\table_placeholders.json"),
        output_path=base / "data" / "OUT" / "project1",
        chapter_module_path=f"src.ecology_chapters.chapter{chapter_n}",
        collection_name="full",
        test_mode="off",
        max_workers=8,
    )
