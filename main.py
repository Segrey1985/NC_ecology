import json
import uuid
from pprint import pprint
from pathlib import Path
from langchain_core.messages import HumanMessage

from agent import graph
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


def _run_graph(detailed_placeholder, verbose=True) -> str:

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)

    input_messages = [HumanMessage(detailed_placeholder)]

    final_content = ""
    for chunk in graph.stream(
        input={
            "messages": input_messages,
            "input_query": input_messages[0].content,
        },
        stream_mode="updates",
        config=config,
    ):
        if verbose:
            print_chunk(chunk)

        if "structured_output_node" in chunk:
            final_output = chunk["structured_output_node"]
            final_content = final_output["messages"][-1].content

    return final_content


def main(
    template_docx_path: Path | None,
    placeholders_path: Path,
    table_placeholders_path: Path | None,
    project_parts_path: Path | None,
    output_path: Path,
):
    placeholders, table_placeholders = _load_placeholders(
        placeholders_path, table_placeholders_path
    )

    placeholders_output = {}
    for placeholder, detailed_placeholder in placeholders.items():
        final_content = _run_graph(detailed_placeholder, verbose=True)
        placeholders_output[placeholder] = json.loads(final_content).get("answer", "__empty__")
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
        template_docx_path=base / "data" / "IN" / "templates" / "Анализ_и_введение.docx",
        placeholders_path=base / "data" / "IN" / "templates" / "Анализ_и_введение.json",
        table_placeholders_path=None,
        project_parts_path=None,
        output_path=base / "data" / "OUT" / "project1",
    )
