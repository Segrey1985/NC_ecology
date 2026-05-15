import json
import uuid
import threading
from pathlib import Path
from typing import Literal

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.agent2 import init_graph_2, PARAMS_2
from config.langfuse_client import langfuse_config
from src.utils.logger import logger
from src.utils.utils import (
    print_chunk,
    is_valid_uuid4_hex,
    iter_models_from_module,
    iter_chapter_models,
    pick_assembly_model,
    filter_payload_and_validate,
    assembly_to_docx_context,
    build_input_query,
)
from src.templates.docx_template_engine import fill_docx_template


def _run_graph(
    graph,
    input_query: str,
    output_model: type[BaseModel],
    verbose: bool = True,
) -> str:

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)

    final_content = ""
    for chunk in graph.stream(
        input={
            "input_query": input_query,
            "output_model": output_model,
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

    handler_id = logger.add(_capture_sink, filter=_only_this_thread, colorize=True)
    return handler_id, log_lines


def thread_run_graph_for_model(
    graph: CompiledStateGraph, model: type[BaseModel], verbose: bool
) -> dict:
    handler_id, log_lines = _log_thread()

    try:
        final_content = _run_graph(
            graph,
            input_query=build_input_query(model),
            output_model=model,
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
    project_parts_path: Path | None,
    output_path: Path,
    chapter_module_path: str,
    collection_name: str = "main",
    verbose: bool = True,
    test_mode: Literal["on", "off", "mock", "filter"] = "on",
    max_workers: int | None = None,
):
    try:
        # init graph

        graph = init_graph_2(
            collection_name=collection_name, project_parts_path=project_parts_path
        )

        total_results: list[dict] = []

        if test_mode == "mock":
            results = json.load(
                open(
                    "data/mock/chapter1_models_output.json",
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

            results: dict[str, object] = {model.__name__: None for model in models}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        thread_run_graph_for_model,
                        graph=graph,
                        model=model,
                        verbose=verbose,
                    )
                    for model in models
                ]

                for future in as_completed(futures):
                    dct = future.result()
                    results[dct["model_name"]] = dct["result"]
                    total_results.append(dct)

        # print results

        for t in total_results:
            model_name = t["model_name"]
            log_lines = t["logs_lines"]
            print(f"\n--- Логи потока {model_name} ({len(log_lines)} записей) ---")
            for line in log_lines:
                print(line)
            print("--- конец логов потока ---\n")

        # export results

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

            results_out_path = output_path / "chapter1_models_output.json"
            with open(results_out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            if template_docx_path:
                assembly_module_path = chapter_module_path + ".assembly"
                assembly_model = pick_assembly_model(assembly_module_path)
                if test_mode == "filter":
                    data = filter_payload_and_validate(assembly_model, results)
                    data_dict = assembly_to_docx_context(assembly_model, data)
                else:
                    data = assembly_model.model_validate(results)
                    data_dict = data.model_dump(mode="json")
                result_template_out_path = (
                    output_path / f"{chapter_module_path.split('.')[-1]}.docx"
                )
                fill_docx_template(
                    template_path=template_docx_path,
                    data=data_dict,
                    output_docx_path=result_template_out_path,
                )
    finally:
        qdrant_service = PARAMS_2.qdrant_service
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
    main(
        template_docx_path=base
        / "data"
        / "IN"
        / "project1"
        / "schemas"
        / "1_Общие_сведения"
        / "chapter1_template.docx",
        project_parts_path=None,
        output_path=base / "data" / "OUT" / "project1",
        chapter_module_path="src.ecology_chapters.chapter1",
        collection_name="all",
        test_mode="filter",
        max_workers=8,
    )
