import uuid
from langchain_core.messages import HumanMessage

from config.langfuse_client import langfuse_config
from agent import init_graph

def test_agent():
    graph = init_graph(collection_name="test_data", project_parts_path=None)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)
    input_messages = [HumanMessage('Проектируемые электросети')]
    for chunk in graph.stream(
        input={
            "messages": input_messages,
            "input_query": input_messages[0].content,
        },
        stream_mode="updates",
        config=config
    ):
        print(chunk)
