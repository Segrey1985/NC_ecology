import uuid
from pathlib import Path
from typing import Literal, Optional, TypedDict

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from pydantic import BaseModel, Field

from src.utils.logger import logger
from config.config_file import cfg
from config.langfuse_client import langfuse_config
from src.models import LlmModel
from src.utils.utils import print_chunk
from src.pydantic_models import StructuredResponse
from src.project_data.qdrant import (
    QdrantService,
    collect_project_parts,
    build_qdrant_service,
    ProjectPart,
)
from src.project_data.reranker import rerank_chunks


class GraphParams:
    """Класс для хранения текущего состояния ресурсов графа."""

    def __init__(self):
        self.collection_name: Optional[str] = None
        self.qdrant_service: Optional[QdrantService] = None
        self.llm = None


PARAMS = GraphParams()  # глобальный объект параметров


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


# --- Вспомогательные функции ---


def create_project_parts(project_parts_path: Path) -> list[ProjectPart]:
    project_parts = collect_project_parts(project_parts_path)
    for project_part in project_parts:
        project_part.run()
    return project_parts


def create_collection(qdrant_service: QdrantService, collection_name: str) -> None:
    qdrant_service.create_collection(collection_name=collection_name)


def fill_collection(
    qdrant_service: QdrantService,
    collection_name: str,
    project_parts: list[ProjectPart],
) -> None:
    for project_part in project_parts:
        qdrant_service.add_points_to_collection(
            collection_name=collection_name,
            points=project_part.points,
        )


# --- Tools ---


def search_in_related_disciplines(query: str) -> list[str]:
    """Найти релевантные части текста в документах смежных разделов."""
    qdrant_service = PARAMS.qdrant_service
    collection_name = PARAMS.collection_name
    if qdrant_service is None or collection_name is None:
        raise RuntimeError("Qdrant не инициализирован. Сначала вызовите init_graph().")

    relevant_points = qdrant_service.run_query(
        query, collection_name=collection_name, limit=30
    )
    texts = [point.payload["text"] for point in relevant_points]
    reranked = rerank_chunks(query, texts)[0:5]
    return [chunk for chunk, _score in reranked]


def format_rag_context(chunks: list[str]) -> str:
    if not chunks:
        return "Релевантный контекст не найден."
    return "\n\n".join(f"[{idx}] {chunk}" for idx, chunk in enumerate(chunks, start=1))


# --- Nodes ---


def rag_search_node(state: AgentState) -> AgentState:
    rag_query = state.get("rag_query") or state["input_query"]
    chunks = search_in_related_disciplines(rag_query)
    rag_context = format_rag_context(chunks)
    logger.info(f"RAG search completed for query: {rag_query}")
    return {
        "rag_query": rag_query,
        "rag_context": rag_context,
    }


def answer_node(state: AgentState) -> AgentState:
    llm = PARAMS.llm
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
    response = llm.with_structured_output(StructuredResponse).invoke(messages)
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


def check_node(state: AgentState) -> AgentState:
    llm = PARAMS.llm
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
    check = llm.with_structured_output(AnswerCheck).invoke(messages)

    return {
        "check_decision": check.decision,
        "check_reason": check.reason,
    }


def rewrite_query_node(state: AgentState) -> AgentState:
    llm = PARAMS.llm
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


def init_graph(collection_name: str, project_parts_path: Path | None):
    """
    Инициализирует параметры и собирает граф.
    """

    # Обновляем глобальные параметры
    PARAMS.collection_name = collection_name
    PARAMS.qdrant_service = build_qdrant_service()

    # Создаем и заполняем новую коллекцию, при необходимости
    if not PARAMS.qdrant_service.client.collection_exists(collection_name):
        if not project_parts_path:
            raise ValueError(
                f"Коллекция {collection_name} не существует. "
                f"Требуется создание коллекции из project_parts_path. "
                f"Аргумент project_parts_path не передан."
            )
        logger.info(f"Создаю новую коллекцию <{collection_name}>")
        project_parts = create_project_parts(project_parts_path)
        create_collection(PARAMS.qdrant_service, collection_name)
        fill_collection(PARAMS.qdrant_service, collection_name, project_parts)
    else:
        logger.info(f"Найдена существующая коллекция <{collection_name}>")

    PARAMS.llm = LlmModel(model_type="ai_tunnel", model_name=cfg.MODEL_NAME).create()

    builder = StateGraph(AgentState)
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


if __name__ == "__main__":

    graph = init_graph(
        collection_name="main", project_parts_path=Path("data/IN/project1/trim")
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
