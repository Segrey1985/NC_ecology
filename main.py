import uuid
import json
from pathlib import Path
from operator import add
from typing import Optional, Annotated
from concurrent.futures import ThreadPoolExecutor, as_completed

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, BaseTool
from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    RemoveMessage,
)

from src.utils.logger import logger
from config.config_file import cfg
from src.models import LlmModel
from src.pydantic_models import RelatedDisciplinesSearch
from src.project_data.qdrant import QdrantService, collect_project_parts, build_qdrant_service
from src.project_data.reranker import rerank_chunks


def create_and_fill_collection(collection_name: str) -> QdrantService:
    qdrant_service = build_qdrant_service()
    if not qdrant_service.client.collection_exists(collection_name):
        
        project_parts = collect_project_parts(Path("data/IN/project1/trim"))
        
        for project_part in project_parts:
            project_part.run()
        
        qdrant_service.create_collection(collection_name=collection_name)
        for project_part in project_parts:
            qdrant_service.add_points_to_collection(
                collection_name=collection_name,
                points=project_part.points,
            )
    
    return qdrant_service


COLLECTION_NAME = "main"
qdrant_service = create_and_fill_collection(COLLECTION_NAME)
llm = LlmModel(model_type="ai_tunnel", model_name=cfg.MODEL_NAME).create()


class AgentState(MessagesState):
    pass


@tool(args_schema=RelatedDisciplinesSearch)
def search_related_disciplines(query: str):
    """ найти релевантные документы """
    relevant_points = qdrant_service.run_query(query, collection_name=COLLECTION_NAME, limit=30)
    texts = [point.payload['text'] for point in relevant_points]
    
    reranked = rerank_chunks(query, texts)[0:5]
    return reranked


tools_list = [search_related_disciplines]
tool_node = ToolNode(tools_list)

def agent_node(state: AgentState) -> AgentState:
    system_message = SystemMessage(
        "Ты помощник по поиску данных по строительному проекту. "
        "Для поиска можешь пользоваться search_related_disciplines. "
        "Этот инструмент может искать по запросу релевантные части текста по смежным разделам."
    )
    messages = [system_message] + state["messages"]
    llm_with_tools = llm.bind_tools(tools_list)
    response = llm_with_tools.invoke(messages)
    if response.tool_calls:
        logger.info(
            f"LLM запросил {len(response.tool_calls)} инструментов: "
            f"{[tc['name'] for tc in response.tool_calls]}"
        )
        
    return {
        "messages": [response],
    }


builder = StateGraph(AgentState)
builder.add_node("tools", tool_node)
builder.add_node("agent_node", agent_node)
builder.add_edge(START, "agent_node")
builder.add_conditional_edges("agent_node", tools_condition)
builder.add_edge("tools", "agent_node")

graph = builder.compile()


if __name__ == "__main__":
    create_and_fill_collection(COLLECTION_NAME)
    
    input_messages = [HumanMessage('Сведения о прочностных и деформационных характеристиках грунта в основании объекта капитального строительства')]
    
    for chunk in graph.stream(
            {"messages": input_messages}, stream_mode="updates"
    ):
        print(chunk)
        

