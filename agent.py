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
from langgraph.graph import MessagesState, StateGraph, START, END
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
from config.langfuse_client import langfuse_config
from src.models import LlmModel
from src.utils.utils import print_chunk
from src.pydantic_models import RelatedDisciplinesSearch, StructuredResponse
from src.project_data.qdrant import (
    QdrantService,
    collect_project_parts,
    build_qdrant_service,
)
from src.project_data.reranker import rerank_chunks


class GraphParams:
    """Класс для хранения текущего состояния ресурсов графа."""

    def __init__(self):
        self.collection_name: Optional[str] = None
        self.qdrant_service: Optional[QdrantService] = None
        self.llm = None


PARAMS = GraphParams()  # глобальный объект параметров


class AgentState(MessagesState):
    input_query: str


# --- Вспомогательные функции ---


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


# --- Tools ---


@tool(args_schema=RelatedDisciplinesSearch)
def search_in_related_disciplines(query: str):
    """Найти релевантные части текста в документах смежных разделов."""

    qdrant_service = PARAMS.qdrant_service
    collection_name = PARAMS.collection_name

    relevant_points = qdrant_service.run_query(
        query, collection_name=collection_name, limit=30
    )
    texts = [point.payload["text"] for point in relevant_points]
    reranked = rerank_chunks(query, texts)[0:5]
    return reranked


tools_list = [search_in_related_disciplines]


# --- Nodes ---


def agent_node(state: AgentState) -> AgentState:
    llm = PARAMS.llm
    system_message = SystemMessage(
        "Ты помощник по поиску данных по строительному проекту. "
        "Для поиска информации можешь пользоваться search_in_related_disciplines."
    )
    messages = [system_message] + state["messages"]
    llm_with_tools = llm.bind_tools(tools_list)
    response = llm_with_tools.invoke(messages)
    if response.tool_calls:
        logger.info(
            f"LLM запросил {len(response.tool_calls)} инструментов: "
            f"{[tc['name'] for tc in response.tool_calls]}"
        )

    return {"messages": [response]}


def structured_output_node(state: AgentState) -> AgentState:
    llm = PARAMS.llm
    system_message = SystemMessage(
        "Верни ответ в формате:\n"
        "- answer: краткий ответ на вопрос\n"
        "- explanation: пояснение или обоснование из контекста (если есть)"
    )
    input_message = state["input_query"]
    last_llm_message = "Ответ отсутствует"

    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_llm_message = msg.content
            break

    messages = [
        system_message,
        HumanMessage(
            content=f"Запрос пользователя:\n{input_message}\n\nОтвет модели:\n{last_llm_message}"
        ),
    ]
    response = llm.with_structured_output(StructuredResponse).invoke(messages)

    return {
        "messages": [AIMessage(content=response.model_dump_json())],
    }


# --- Инициализация графа ---


def init_graph(collection_name: str = "main"):
    """
    Инициализирует параметры и собирает граф.
    """
    # Обновляем глобальные параметры
    PARAMS.collection_name = collection_name
    PARAMS.qdrant_service = create_and_fill_collection(collection_name)
    PARAMS.llm = LlmModel(model_type="ai_tunnel", model_name=cfg.MODEL_NAME).create()

    builder = StateGraph(AgentState)
    builder.add_node("tools", ToolNode(tools_list))
    builder.add_node("agent_node", agent_node)
    builder.add_node("structured_output_node", structured_output_node)
    builder.add_edge(START, "agent_node")
    builder.add_conditional_edges(
        "agent_node", tools_condition, {"tools": "tools", END: "structured_output_node"}
    )
    builder.add_edge("tools", "agent_node")
    builder.add_edge("structured_output_node", END)

    return builder.compile()


if __name__ == "__main__":

    graph = init_graph(collection_name="main")

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)

    input_query = "Проектируемые электросети"
    input_messages = [HumanMessage(input_query)]

    for chunk in graph.stream(
        input={
            "messages": input_messages,
            "input_query": input_query,
        },
        stream_mode="updates",
        config=config,
    ):
        print_chunk(chunk)
