import json
import uuid
from pathlib import Path
from langchain_core.messages import HumanMessage

from agent import graph
from config.langfuse_client import langfuse_config
from src.utils.utils import print_chunk

__placeholders_example = {
    "НАИМЕНОВАНИЕ_ПРОЕКТА": "Наименование проекта",
    "ТИП_РАБОТ": " тип строительных работ для склонения в тексте (например, 'строительства', 'реконструкции', 'технического перевооружения')."
}
    

def main(
    template_docx_path: Path | None,
    placeholders_path: Path,
    table_placeholders_path: Path | None,
    project_parts_path: Path | None,
):
    
    output = {}
    
    if table_placeholders_path:
        with open(table_placeholders_path, "r", encoding="utf-8") as table_placeholders_file:
            table_placeholders = json.load(table_placeholders_file)
    else:
        table_placeholders = {}
        
    with open(placeholders_path, "r", encoding="utf-8") as placeholders_file:
        placeholders = json.load(placeholders_file)
        
    placeholders = {k: v for k, v in placeholders.items() if k not in table_placeholders}
    
    final_content = ""
    for placeholder, detailed_placeholder in placeholders.items():
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        config.update(langfuse_config)
        
        input_messages = [HumanMessage(detailed_placeholder)]
        
        for chunk in graph.stream(
            input={
                "messages": input_messages,
                "input_query": input_messages[0].content,
            },
            stream_mode="updates",
            config=config
        ):
            print_chunk(chunk)
            
            if "structured_output_node" in chunk:
                final_output = chunk["structured_output_node"]
                final_content = final_output["messages"][-1].content
        
        output[placeholder] = json.loads(final_content).get("answer", "__empty__")
        # break
    
    return output

if __name__ == "__main__":
    
    project_path = Path(__file__).parent
    output = main(
        template_docx_path = None,
        placeholders_path = project_path / "data" / "IN" / "templates" / "Анализ_и_введение.json" ,
        table_placeholders_path = None,
        project_parts_path = None,
    )
    
    print(output)