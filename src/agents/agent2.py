import json
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config.config_file import cfg
from src.models import LlmModel
from src.retrieval.qdrant import (
    QdrantService,
    build_qdrant_service,
    create_collection,
    create_project_parts,
    fill_collection,
)
from src.retrieval.reranker import rerank_chunks
from src.utils.logger import logger
from src.utils.utils import format_rag_context


class GraphParams:
    """Класс для хранения текущего состояния ресурсов графа."""

    def __init__(self):
        self.collection_name: Optional[str] = None
        self.qdrant_service: Optional[QdrantService] = None
        self.llm = None


PARAMS_2 = GraphParams()  # глобальный объект параметров (для agent_2)


class Agent2State(TypedDict):
    # input
    input_query: str

    # rag
    rag_query: str
    rag_context: str

    # agent_node
    input_for_agent_prompt: str
    output_model: BaseModel
    answer: str

    # optional loop
    check_decision: Literal["OK", "REWRITE"]
    check_reason: str
    rewrite_count: int


def _search_in_related_disciplines(query: str) -> list[str]:
    qdrant_service = PARAMS_2.qdrant_service
    collection_name = PARAMS_2.collection_name
    if qdrant_service is None or collection_name is None:
        raise RuntimeError("Qdrant не инициализирован. Сначала вызовите init_graph_2().")

    relevant_points = qdrant_service.run_query(query, collection_name=collection_name, limit=30)
    texts = [point.payload["text"] for point in relevant_points]
    reranked = rerank_chunks(query, texts)[0:5]
    return [chunk for chunk, _score in reranked]


def rag_search_node(state: Agent2State) -> Agent2State:
    rag_query = state.get("rag_query") or state["input_query"]
    chunks = _search_in_related_disciplines(rag_query)
    rag_context = format_rag_context(chunks)
    logger.info(f"[agent_2] RAG search completed for query: {rag_query}")
    return {"rag_query": rag_query, "rag_context": rag_context}


def answer_node(state: Agent2State) -> Agent2State:
    llm = PARAMS_2.llm

    system_message = SystemMessage(
        "Ты помощник по извлечению данных по строительному проекту.\n"
        "Заполни JSON строго по переданной схеме и только на основе RAG-контекста.\n"
        "Если значения нет в контексте, ставь строку '__empty__'.\n"
        "Не добавляй поля, которых нет в схеме."
    )
    messages = [
        system_message,
        HumanMessage(
            content=(
                f"Задача:\n{state['input_for_agent_prompt']}\n\n"
                f"RAG-контекст:\n{state['rag_context']}"
            )
        ),
    ]
    response = llm.with_structured_output(state['output_model']).invoke(messages)
    response_json = response.model_dump_json()

    return {
        "answer": response_json
    }


class AnswerCheck(BaseModel):
    decision: Literal["OK", "REWRITE"] = Field(
        ...,
        description="OK, если контекста хватило; иначе REWRITE.",
    )
    reason: str = Field(..., description="Краткая причина решения.")


def check_node(state: Agent2State) -> Agent2State:
    """
    Дешёвая эвристика: если слишком много '__empty__', пробуем переписать RAG-запрос.
    """
    rewrite_count = state.get("rewrite_count", 0)
    try:
        data = json.loads(state.get("answer") or "{}")
    except Exception:
        return {
            "check_decision": "REWRITE" if rewrite_count < 2 else "OK",
            "check_reason": "Ответ не является валидным JSON.",
        }

    missing = _collect_empty_fields(data)
    if len(missing) >= 3 and rewrite_count < 2:
        return {
            "check_decision": "REWRITE",
            "check_reason": "Много пустых полей: " + ", ".join(missing[:10]),
        }

    return {"check_decision": "OK", "check_reason": "Контекст достаточен."}


def rewrite_query_node(state: Agent2State) -> Agent2State:
    llm = PARAMS_2.llm
    system_message = SystemMessage(
        "Перепиши запрос для RAG-поиска так, чтобы найти недостающий контекст. "
        "Верни только текст запроса."
    )
    messages = [
        system_message,
        HumanMessage(
            content=(
                f"Исходный запрос:\n{state['input_query']}\n\n"
                f"Предыдущий RAG-запрос:\n{state['rag_query']}\n\n"
                f"Причина:\n{state.get('check_reason', '')}\n\n"
                "Сделай запрос более конкретным, упомяни недостающие поля и возможные синонимы."
            )
        ),
    ]
    response = llm.invoke(messages)
    rewritten = str(getattr(response, "content", response)).strip() or state["input_query"]
    logger.info(f"[agent_2] RAG query rewritten: {rewritten}")
    return {"rag_query": rewritten, "rewrite_count": state.get("rewrite_count", 0) + 1}


def route_after_check(state: Agent2State) -> str:
    if state.get("check_decision") == "REWRITE" and state.get("rewrite_count", 0) < 2:
        return "rewrite_query_node"
    return END


def init_graph_2(collection_name: str, project_parts_path: Path | None):
    """
    Инициализирует параметры и собирает граф для работы с минисхемами JSON Schema.
    """
    PARAMS_2.collection_name = collection_name
    PARAMS_2.qdrant_service = build_qdrant_service()

    if not PARAMS_2.qdrant_service.client.collection_exists(collection_name):
        if not project_parts_path:
            raise ValueError(
                f"Коллекция {collection_name} не существует. "
                f"Требуется создание коллекции из project_parts_path. "
                f"Аргумент project_parts_path не передан."
            )
        logger.info(f"[agent_2] Создаю новую коллекцию <{collection_name}>")
        project_parts = create_project_parts(project_parts_path)
        create_collection(PARAMS_2.qdrant_service, collection_name)
        fill_collection(PARAMS_2.qdrant_service, collection_name, project_parts)
    else:
        logger.info(f"[agent_2] Найдена существующая коллекция <{collection_name}>")

    PARAMS_2.llm = LlmModel(model_type="ai_tunnel", model_name=cfg.MODEL_NAME).create()

    builder = StateGraph(Agent2State)
    builder.add_node("rag_search_node", rag_search_node)
    builder.add_node("answer_node", answer_node)
    builder.add_node("check_node", check_node)
    builder.add_node("rewrite_query_node", rewrite_query_node)

    builder.add_edge(START, "rag_search_node")
    builder.add_edge("rag_search_node", "answer_node")
    builder.add_edge("answer_node", "check_node")
    builder.add_conditional_edges(
        "check_node",
        route_after_check,
        {"rewrite_query_node": "rewrite_query_node", END: END},
    )
    builder.add_edge("rewrite_query_node", "rag_search_node")

    return builder.compile()

