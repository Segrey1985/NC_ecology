import json
import uuid
import threading
from pathlib import Path
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.agent_base import GraphResources, init_graph
from config.langfuse_client import langfuse_config
from config.config_file import build_runtime_config
from src.utils.logger import logger
from src.utils.utils import print_chunk, is_valid_uuid4_hex
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
    graph, for_rag_search: str, examples: list[str], question: str, verbose: bool = True
) -> str:

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)

    final_content = ""
    for chunk in graph.stream(
        input={
            "for_rag_search": for_rag_search,
            "examples": examples,
            "question": question,
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
            for_rag_search=placeholder_info["for_rag_search"],
            examples=placeholder_info["examples"],
            question=placeholder_info["question"],
            verbose=verbose,
        )
        return {
            "placeholder": placeholder,
            "result": json.loads(final_content).get("answer", "__empty__"),
            "logs_lines": log_lines,
        }
    finally:
        logger.remove(handler_id)


def main(
    template_docx_path: Path | None,
    placeholders_path: Path,
    table_placeholders_path: Path | None,
    project_parts_path: Path | None,
    output_path: Path,
    collection_name: str = "main",
    verbose: bool = True,
    test_mode: Literal["on", "off", "mock"] = "on",
    max_workers: int | None = None,
):
    resources: GraphResources | None = None
    try:
        runtime_cfg = build_runtime_config(test_mode)
        
        graph, resources = init_graph(
            collection_name=collection_name, project_parts_path=project_parts_path, runtime_cfg=runtime_cfg
        )
    
        placeholders, table_placeholders = _load_placeholders(
            placeholders_path, table_placeholders_path
        )
    
        if test_mode == "mock":
            placeholders_output = json.load(
                open(
                    "data/mock/placeholders_output.json",
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

            for t in total_results:
                placeholder = t["placeholder"]
                log_lines = t["logs_lines"]
                print(f"\n--- Логи потока {placeholder} ({len(log_lines)} записей) ---")
                for line in log_lines:
                    print(line)
                print("--- конец логов потока ---\n")
    
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
    finally:
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
    input_dir = base / "data" / "IN" / "project1" / "schemas" / "0_Аннотация_и_Введение"
    main(
        template_docx_path=input_dir / "template.docx",
        placeholders_path=input_dir / "placeholders.json",
        table_placeholders_path=input_dir / "table_placeholders.json",
        project_parts_path=Path(r"C:\Users\maxfi\PycharmProjects\NC_ecology\data\IN\project1\trim"),
        output_path=base / "data" / "OUT" / "project1",
        collection_name="main_base_test_off",
        test_mode="off",
    )
