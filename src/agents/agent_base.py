import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field

from src.utils.logger import logger
from config.config_file import cfg, Config
from config.langfuse_client import langfuse_config
from src.models import LlmModel
from src.utils.utils import print_chunk, format_rag_context
from src.pydantic_models import StructuredResponse
from src.retrieval.qdrant import (
    QdrantService,
    build_qdrant_service,
    create_project_parts,
    create_collection,
    fill_collection
)
from src.retrieval.reranker import rerank_chunks


@dataclass(frozen=True)
class GraphResources:
    """Ресурсы экземпляра графа"""
    collection_name: str
    qdrant_service: QdrantService
    llm: object
    runtime_cfg: Config


class AgentState(TypedDict):
    # input
    input_query: str

    # rag
    rag_query: str
    rag_context: str

    # agent_node
    input_for_agent_prompt: str
    answer: str

    # check_node
    check_decision: Literal["OK", "REWRITE"]
    check_reason: str

    # rewrite_node
    rewrite_count: int


# --- Tools ---


def search_in_related_disciplines(query: str, resources: GraphResources) -> list[str]:
    """Найти релевантные части текста в документах смежных разделов."""
    qdrant_service = resources.qdrant_service
    collection_name = resources.collection_name
    if qdrant_service is None or collection_name is None:
        raise RuntimeError("Qdrant не инициализирован. Сначала вызовите init_graph().")

    relevant_points = qdrant_service.run_query(
        query, collection_name=collection_name, limit=50
    )
    texts = [point.payload["text"] for point in relevant_points]
    reranked = rerank_chunks(
        query, texts, reranker_model=resources.runtime_cfg.RERANKER_MODEL, top_n=5
    )
    return [chunk for chunk, _score in reranked]


# --- Nodes ---


def rag_search_node(state: AgentState, resources: GraphResources) -> AgentState:
    rag_query = state.get("rag_query") or state["input_query"]
    chunks = search_in_related_disciplines(rag_query, resources)
    rag_context = format_rag_context(chunks)
    logger.info(f"RAG search completed for query: {rag_query}")
    return {
        "rag_query": rag_query,
        "rag_context": rag_context,
    }


def answer_node(state: AgentState, resources: GraphResources) -> AgentState:
    llm = resources.llm
    system_message = SystemMessage(
        "Ты помощник по поиску данных по строительному проекту. "
        "Отвечай только на основе переданного RAG-контекста. "
        "Если в контексте нет данных для уверенного ответа, так и укажи."
        "Правила:\n"
        "1. Запрещено повторять вопрос в ответе.\n"
        "2. Запрещено использовать вводные конструкции (например, 'Согласно документу...', 'Основанием является...').\n"
        "3. Выводи только конкретный факт или фрагмент текста."
    )
    messages = [
        system_message,
        HumanMessage(
            content=(
                f"Запрос пользователя:\n{state['input_for_agent_prompt']}\n\n"
                f"RAG-контекст:\n{state['rag_context']}"
            )
        ),
    ]
    response = llm.with_structured_output(StructuredResponse, strict=True).invoke(messages)
    response_json = response.model_dump_json()

    return {
        "answer": response_json,
    }


class AnswerCheck(BaseModel):
    decision: Literal["OK", "REWRITE"] = Field(
        ...,
        description="OK, если ответ достаточно обоснован контекстом; иначе REWRITE.",
    )
    reason: str = Field(..., description="Краткая причина решения.")
    rewrite_focus: Optional[str] = Field(
        None,
        description="Что нужно уточнить при переписывании RAG-запроса.",
    )


def check_node(state: AgentState, resources: GraphResources) -> AgentState:
    llm = resources.llm
    system_message = SystemMessage(
        "Проверь, отвечает ли ответ на запрос пользователя и достаточно ли он "
        "подтвержден RAG-контекстом. Верни OK, если ответ можно использовать. "
        "Верни REWRITE, если нужен более точный RAG-запрос "
        "или если было найдено недостаточно данных для корректного ответа."
    )
    messages = [
        system_message,
        HumanMessage(
            content=(
                f"Запрос пользователя:\n{state['input_for_agent_prompt']}\n\n"
                f"RAG-контекст:\n{state['rag_context']}\n\n"
                f"Ответ:\n{state['answer']}"
            )
        ),
    ]
    check = llm.with_structured_output(AnswerCheck, strict=True).invoke(messages)

    return {
        "check_decision": check.decision,
        "check_reason": check.reason,
    }


def rewrite_query_node(state: AgentState, resources: GraphResources) -> AgentState:
    llm = resources.llm
    system_message = SystemMessage(
        "Перепиши запрос для RAG-поиска так, чтобы следующий поиск нашел "
        "контекст, которого не хватило для ответа. Верни только текст запроса."
    )
    messages = [
        system_message,
        HumanMessage(
            content=(
                f"Исходный запрос пользователя:\n{state['input_query']}\n\n"
                f"Предыдущий RAG-запрос:\n{state['rag_query']}\n\n"
                f"Причина повторного поиска:\n{state['check_reason']}\n\n"
                f"Предыдущий ответ:\n{state['answer']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    rewritten_query = str(response.content).strip() or state["input_query"]
    logger.info(f"RAG query rewritten: {rewritten_query}")

    return {
        "rag_query": rewritten_query,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


def route_after_check(state: AgentState) -> str:
    MAX_REWRITES = 2
    if (
        state.get("check_decision") == "REWRITE"
        and state.get("rewrite_count", 0) < MAX_REWRITES
    ):
        return "rewrite_query_node"
    return END


# --- Инициализация графа ---


def init_graph(
    collection_name: str, project_parts_path: Path | None, runtime_cfg: Config | None = None
):
    """
    Инициализирует параметры и собирает граф.
    """
    
    runtime_cfg = runtime_cfg or cfg

    qdrant_service = build_qdrant_service(runtime_cfg)

    # Создаем и заполняем новую коллекцию, при необходимости
    if not qdrant_service.client.collection_exists(collection_name):
        if not project_parts_path:
            raise ValueError(
                f"Коллекция {collection_name} не существует. "
                f"Требуется создание коллекции из project_parts_path. "
                f"Аргумент project_parts_path не передан."
            )
        logger.info(f"Создаю новую коллекцию <{collection_name}>")
        project_parts = create_project_parts(
            project_parts_path=project_parts_path, embedder=qdrant_service.model
        )
        create_collection(qdrant_service, collection_name)
        fill_collection(qdrant_service, collection_name, project_parts)
    else:
        logger.info(f"Найдена существующая коллекция <{collection_name}>")

    llm = LlmModel(model_type="ai_tunnel", model_name=runtime_cfg.MODEL_NAME).create()
    
    resources = GraphResources(
        collection_name=collection_name,
        qdrant_service=qdrant_service,
        llm=llm,
        runtime_cfg=runtime_cfg,
    )

    builder = StateGraph(AgentState)
    builder.add_node(
        "rag_search_node", lambda state: rag_search_node(state, resources)
    )
    builder.add_node("answer_node", lambda state: answer_node(state, resources))
    builder.add_node("check_node", lambda state: check_node(state, resources))
    builder.add_node(
        "rewrite_query_node", lambda state: rewrite_query_node(state, resources)
    )
    builder.add_edge(START, "rag_search_node")
    builder.add_edge("rag_search_node", "answer_node")
    builder.add_edge("answer_node", "check_node")
    builder.add_conditional_edges(
        "check_node",
        route_after_check,
        {"rewrite_query_node": "rewrite_query_node", END: END},
    )
    builder.add_edge("rewrite_query_node", "rag_search_node")

    return builder.compile(), resources


if __name__ == "__main__":

    graph, _resources = init_graph(
        collection_name="main", project_parts_path=Path("../../data/IN/project1/trim")
    )

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)

    input_query = "Проектируемые электросети"
    input_for_agent_prompt = (
        "Краткое описание проектируемых электросетей и их параметров"
    )

    for chunk in graph.stream(
        input={
            "input_query": input_query,
            "input_for_agent_prompt": input_for_agent_prompt,
        },
        stream_mode="updates",
        config=config,
    ):
        print_chunk(chunk)
