import json
import uuid
import inspect
import importlib
from pathlib import Path
from typing import Literal

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.agent2 import init_graph_2, PARAMS_2
from config.langfuse_client import langfuse_config
from src.utils.logger import logger
from src.utils.utils import print_chunk, is_valid_uuid4_hex
from src.templates.docx_template_engine import fill_docx_template


def _run_graph(
    graph,
    input_for_rag_search: str,
    input_for_agent_prompt: str,
    output_model: type[BaseModel],
    verbose: bool = True,
) -> str:

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)

    final_content = ""
    for chunk in graph.stream(
        input={
            "input_query": input_for_rag_search,
            "input_for_agent_prompt": input_for_agent_prompt,
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


def _iter_models_from_module(module_path: str) -> list[type[BaseModel]]:
    """
    Берём только pydantic-модели, объявленные именно в модуле `module_path`,
    чтобы не тащить импортированные классы (например, из `inner`).
    """
    module = importlib.import_module(module_path)
    out: list[type[BaseModel]] = []
    for _name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, BaseModel):
            continue
        if obj is BaseModel:
            continue
        if getattr(obj, "__module__", "") != module.__name__:
            continue
        out.append(obj)

    out.sort(key=lambda cls: cls.__name__)
    return out


def _pick_assembly_model(assembly_module_path: str) -> type[BaseModel]:
    """
    Достаём единственную корневую модель сборки из `<chapter>.assembly`.
    В `assembly.py` должен быть ровно один локально объявленный класс-наследник BaseModel.
    """
    module = importlib.import_module(assembly_module_path)

    candidates: list[type[BaseModel]] = []
    for _name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, BaseModel):
            continue
        if obj is BaseModel:
            continue
        if getattr(obj, "__module__", "") != module.__name__:
            continue
        candidates.append(obj)

    if len(candidates) != 1:
        raise RuntimeError(
            f"В модуле `{assembly_module_path}` ожидается ровно 1 BaseModel-класс, "
            f"найдено: {len(candidates)}."
        )

    return candidates[0]


def _build_inputs_for_model(model: type[BaseModel]) -> tuple[str, str]:
    doc = (inspect.getdoc(model) or "").strip()
    fields = getattr(model, "model_fields", {}) or {}

    field_lines: list[str] = []
    for f_name, f_info in fields.items():
        desc = getattr(f_info, "description", None)
        if desc:
            field_lines.append(f"- {f_name}: {desc}")
        else:
            field_lines.append(f"- {f_name}")

    input_for_rag_search = f"{model.__name__} {doc}".strip() or model.__name__
    input_for_agent_prompt = (
        f"Заполни модель `{model.__name__}`.\n"
        f"Описание: {doc or '—'}\n"
        "Поля:\n" + "\n".join(field_lines)
    )
    return input_for_rag_search, input_for_agent_prompt


def thread_run_graph_for_model(graph: CompiledStateGraph, model: type[BaseModel], verbose: bool):
    input_for_rag_search, input_for_agent_prompt = _build_inputs_for_model(model)

    final_content = _run_graph(
        graph,
        input_for_rag_search=input_for_rag_search,
        input_for_agent_prompt=input_for_agent_prompt,
        output_model=model,
        verbose=verbose,
    )

    return (
        model.__name__,
        json.loads(final_content) if final_content else {},
    )


def main(
    template_docx_path: Path | None,
    project_parts_path: Path | None,
    output_path: Path,
    chapter_module_path: str,
    collection_name: str = "main",
    verbose: bool = True,
    test_mode: Literal["on", "off", "mock"] = "on",
    max_workers: int | None = None,
):
    
    # init graph
    graph = init_graph_2(
        collection_name=collection_name, project_parts_path=project_parts_path
    )
    
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
        models = _iter_models_from_module(models_module_path)
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
                    thread_run_graph_for_model, graph=graph, model=model, verbose=verbose
                )
                for model in models
            ]
    
            for future in as_completed(futures):
                model_name, result = future.result()
                results[model_name] = result

    qdrant_service = PARAMS_2.qdrant_service
    if qdrant_service.client.collection_exists(collection_name) and is_valid_uuid4_hex(
        collection_name
    ):
        qdrant_service.client.delete_collection(collection_name)
        logger.info(
            f"collection <{collection_name}> name is valid uuid and was deleted"
        )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        results_out_path = output_path / "chapter1_models_output.json"
        with open(results_out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        if template_docx_path:
            assembly_module_path = chapter_module_path + ".assembly"
            assembly_model = _pick_assembly_model(assembly_module_path)
            data = assembly_model.model_validate(results)
            data_dict = data.model_dump(mode="json")
            result_template_out_path = output_path / f"{chapter_module_path.split('.')[-1]}.docx"
            fill_docx_template(
                template_path=template_docx_path,
                data=data_dict,
                output_docx_path=result_template_out_path,
            )

    logger.info("\n\n ЗАВЕРШЕНО \n\n")


if __name__ == "__main__":

    base = Path(__file__).parent
    main(
        template_docx_path=base / "data" / "IN" / "project1" / "schemas" / "1_Общие_сведения" / "chapter1_template.docx",
        project_parts_path=None,
        output_path=base / "data" / "OUT" / "project1",
        chapter_module_path="src.ecology_chapters.chapter1",
        collection_name="main",
        test_mode="off",
        max_workers=8,
    )
