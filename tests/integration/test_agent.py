import uuid
from langchain_core.messages import HumanMessage

from config.langfuse_client import langfuse_config
from agent import init_graph

def test_agent():
    graph = init_graph(collection_name="test_data", project_parts_path=None)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)
    input_query = 'Проектируемые электросети'
    input_for_agent_prompt = 'Краткое описание проектируемых электросетей и их параметров'
    for chunk in graph.stream(
        input={
            "input_query": input_query,
            "input_for_agent_prompt": input_for_agent_prompt
        },
        stream_mode="updates",
        config=config
    ):
        print(chunk)
