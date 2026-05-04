import json
import uuid
from pprint import pprint
from pathlib import Path
from langchain_core.messages import HumanMessage

from agent import init_graph
from config.langfuse_client import langfuse_config
from src.utils.utils import print_chunk
from src.templates.template_engine import fill_template

__placeholders_example = {
    "НАИМЕНОВАНИЕ_ПРОЕКТА": "Наименование проекта",
    "ТИП_РАБОТ": " тип строительных работ для склонения в тексте (например, 'строительства', 'реконструкции', 'технического перевооружения').",
}


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
    graph, input_for_rag_search, input_for_agent_prompt, verbose: bool = True
) -> str:

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)

    final_content = ""
    for chunk in graph.stream(
        input={
            "input_query": input_for_rag_search,
            "input_for_agent_prompt": input_for_agent_prompt
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


def main(
    template_docx_path: Path | None,
    placeholders_path: Path,
    table_placeholders_path: Path | None,
    project_parts_path: Path | None,
    output_path: Path,
    collection_name: str = "main",
    verbose: bool = True,
    test_mode: bool = False,
):
    placeholders, table_placeholders = _load_placeholders(
        placeholders_path, table_placeholders_path
    )

    graph = init_graph(
        collection_name=collection_name, project_parts_path=project_parts_path
    )

    placeholders_output = {}
    for placeholder, placeholder_info in placeholders.items():
        input_for_rag_search = placeholder_info["for_rag_search"]
        input_for_agent_prompt = placeholder_info["for_agent_prompt"]
        final_content = _run_graph(
            graph, input_for_rag_search, input_for_agent_prompt, verbose=verbose
        )
        placeholders_output[placeholder] = json.loads(final_content).get(
            "answer", "__empty__"
        )
        if test_mode:
            break

    ## mock
    # placeholders_output = json.load(
    #     open(
    #         "data/mock/placeholders_output.json",
    #         encoding="utf-8",
    #     )
    # )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        placeholders_out_path = output_path / "placeholders.json"
        with open(placeholders_out_path, "w", encoding="utf-8") as f:
            json.dump(placeholders_output, f, ensure_ascii=False, indent=4)

        if template_docx_path:
            result_template_out_path = output_path / "result_template.docx"
            fill_template(
                template_path=template_docx_path,
                data=placeholders_output,
                output_docx_path=result_template_out_path,
            )


if __name__ == "__main__":

    base = Path(__file__).parent
    main(
        template_docx_path=base
        / "data"
        / "IN"
        / "templates"
        / "Анализ_и_введение.docx",
        placeholders_path=base / "data" / "IN" / "templates" / "Анализ_и_введение.json",
        table_placeholders_path=None,
        project_parts_path=None,
        output_path=base / "data" / "OUT" / "project1",
        collection_name="main",
        test_mode=False,
    )
