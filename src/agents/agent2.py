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
from src.utils.validators import validate_and_dump_json_str


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
    # В цикле снаружи передаётся pydantic-модель (класс), который нужно заполнить.
    output_model: type[BaseModel]
    answer: str

    # optional loop
    check_decision: Literal["OK", "REWRITE"]
    check_reason: str
    rewrite_focus: str
    rewrite_count: int


def _search_in_related_disciplines(query: str) -> list[str]:
    qdrant_service = PARAMS_2.qdrant_service
    collection_name = PARAMS_2.collection_name
    if qdrant_service is None or collection_name is None:
        raise RuntimeError(
            "Qdrant не инициализирован. Сначала вызовите init_graph_2()."
        )

    relevant_points = qdrant_service.run_query(
        query, collection_name=collection_name, limit=30
    )
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
        "Если значения нет в контексте, не выдумывай. Используй только допустимые схемой "
        "пустые значения (например, null для Optional-полей или пустые списки там, где это уместно).\n"
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
    
    try:
        response = llm.with_structured_output(state["output_model"], strict=True).invoke(messages)
        response_json = response.model_dump_json()
        return {"answer": response_json}
    
    except Exception:
        # structured_output может падать на несовпадении типов (pydantic ValidationError).
        # fallback: просим вернуть "сырой" JSON. Если и это не удаётся — возвращаем пустой объект,
        # чтобы граф продолжил работу и check_node мог инициировать rewrite.
        logger.exception("Structured output validation failed in answer_node. trying fallback №1")
        
        fallback_system = SystemMessage(
            "Сформируй ОДИН валидный JSON-объект строго по указанной схеме.\n"
            "Требования:\n"
            "- Соблюдай типы (int/float/bool/string/null/array/object).\n"
            "- Не добавляй лишних полей.\n"
            "- Если значения нет в контексте: используй null (для Optional) или пустой список, если поле list.\n"
            "Верни только JSON, без пояснений и без markdown."
        )
        
        try:
            raw = llm.invoke(
                [
                    fallback_system,
                    HumanMessage(
                        content=(
                            f"Схема (Pydantic модель): {state['output_model'].__name__}\n"
                            f"Задача:\n{state['input_for_agent_prompt']}\n\n"
                            f"RAG-контекст:\n{state.get('rag_context', '')}"
                        )
                    ),
                ]
            )
            response_json = validate_and_dump_json_str(state['output_model'], str(getattr(raw, "content", raw)))
            return {"answer": response_json}
        except Exception:
            logger.exception(
                "Structured output validation failed in answer_node in fallback №1. "
                "Doing fallback №2 and return empty dict"
            )
            return {"answer": "{}"}


class AnswerCheck(BaseModel):
    decision: Literal["OK", "REWRITE"] = Field(
        ...,
        description="OK, если контекста хватило; иначе REWRITE.",
    )
    reason: str = Field(..., description="Краткая причина решения.")
    rewrite_focus: Optional[str] = Field(
        None,
        description="Что именно нужно найти/уточнить при переписывании RAG-запроса.",
    )


def check_node(state: Agent2State) -> Agent2State:
    llm = PARAMS_2.llm

    system_message = SystemMessage(
        "Проверь, можно ли считать ответ корректным заполнением схемы на основе RAG-контекста.\n"
        "Верни OK, если ответ можно использовать.\n"
        "Верни REWRITE, если RAG-контекст не даёт достаточно данных/есть явные пробелы и нужно "
        "переформулировать запрос для поиска.\n"
    )
    messages = [
        system_message,
        HumanMessage(
            content=(
                f"Задача (что нужно извлечь):\n{state['input_for_agent_prompt']}\n\n"
                f"RAG-контекст:\n{state.get('rag_context', '')}\n\n"
                f"Ответ (JSON):\n{state.get('answer', '')}"
            )
        ),
    ]

    check = llm.with_structured_output(AnswerCheck).invoke(messages)
    return {
        "check_decision": check.decision,
        "check_reason": check.reason,
        "rewrite_focus": check.rewrite_focus or "",
    }


def rewrite_query_node(state: Agent2State) -> Agent2State:
    llm = PARAMS_2.llm
    system_message = SystemMessage(
        "Перепиши запрос для RAG-поиска так, чтобы следующий поиск нашёл контекст, "
        "которого не хватило для заполнения схемы. Верни только текст запроса."
    )
    messages = [
        system_message,
        HumanMessage(
            content=(
                f"Исходный запрос:\n{state.get('input_query', '')}\n\n"
                f"Предыдущий RAG-запрос:\n{state['rag_query']}\n\n"
                f"Причина повторного поиска:\n{state.get('check_reason', '')}\n\n"
                f"Фокус переписывания (если указан):\n{state.get('rewrite_focus', '')}\n\n"
                f"Задача (что нужно извлечь):\n{state.get('input_for_agent_prompt', '')}\n\n"
                f"Предыдущий ответ (JSON):\n{state.get('answer', '')}\n\n"
                "Сделай запрос более конкретным: используй термины из задачи, "
                "возможные синонимы и формулировки из предметной области."
            )
        ),
    ]
    response = llm.invoke(messages)
    rewritten = (
        str(getattr(response, "content", response)).strip() or state["input_query"]
    )
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
