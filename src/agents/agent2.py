import json
import uuid
from pathlib import Path
from operator import add
from typing import Any, Literal, Optional, TypedDict, Annotated

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
from src.ecology_chapters.chapter1.rag_map import get_part_names_for_model


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
    rag_prompt: str
    rag_context: str
    rag_contexts: Annotated[list[str], add]
    reranker_prompt: str

    # agent_node (output_model — класс pydantic-схемы для structured output)
    answer: str
    output_model: type[BaseModel]

    # optional loop (rewrite_focus из check при REWRITE)
    check_decision: Literal["OK", "REWRITE"]
    check_reason: str
    rewrite_focus: str
    rewrite_count: int


def _rag_search_and_rerank(
    rag_prompt: str,
    reranker_prompt: str,
    output_model: type[BaseModel] | None,
) -> list[str]:
    qdrant_service = PARAMS_2.qdrant_service
    collection_name = PARAMS_2.collection_name
    if qdrant_service is None or collection_name is None:
        raise RuntimeError(
            "Qdrant не инициализирован. Сначала вызовите init_graph_2()."
        )
    part_names = get_part_names_for_model(output_model)
    relevant_points = qdrant_service.run_query(
        rag_prompt,
        collection_name=collection_name,
        limit=50,
        part_names=part_names,
    )
    texts = [point.payload["text"] for point in relevant_points]
    reranked = rerank_chunks(reranker_prompt, texts, top_n=5)
    return [chunk for chunk, _score in reranked]


class RetrievalPrompts(BaseModel):
    """Промпты для retrieval-этапа: dense retrieval и cross-encoder reranking."""

    rag_prompt: str = Field(
        ...,
        description=(
            "Семантический запрос для dense/vector поиска в Qdrant. "
            "Должен содержать предметные термины, синонимы, сокращения, "
            "типовые формулировки из проектной документации, СП, ГОСТ и других "
            "источников, близкие к ожидаемому тексту в документах. "
            "Оптимизируется под recall и поиск максимально релевантных чанков."
        ),
    )

    reranker_prompt: str = Field(
        ...,
        description=(
            "Короткая и точная формулировка целевой информации для cross-encoder reranker. "
            "Должна описывать, какая именно информация должна присутствовать во фрагменте. "
            "Оптимизируется под semantic precision и точную оценку релевантности. "
            "Без длинных инструкций, перечислений синонимов и служебного текста."
        ),
    )


def generate_retrieval_prompts_node(state: Agent2State) -> Agent2State:
    
    input_query = state["input_query"]
    prev_rewrite_count = state.get("rewrite_count", 0)
    
    out = {}
    
    if state.get("check_decision") == "REWRITE" and prev_rewrite_count < 2:
        rewrite_count = prev_rewrite_count + 1
        out["rewrite_count"] = rewrite_count
    else:
        rewrite_count = prev_rewrite_count

    system_message = SystemMessage(
        "Ты готовишь два разных текста для RAG по проектной документации (строительство, экология).\n"
        "1) rag_prompt — для семантического (векторного) поиска по чанкам: насыщен ключевыми словами, "
        "синонимами, типовыми аббревиатурами разделов, чтобы поиск нашёл кандидатов шире.\n"
        "2) reranker_prompt — для cross-encoder: одна ясная формулировка «что искать во фрагменте», "
        "по смыслу близкая к проверке релевантности пары (запрос, абзац документа).\n"
        "Оба текста на русском. Не дублируй дословно длинные куски задачи — извлеки суть под поиск."
    )

    extra = ""
    if rewrite_count > 0:
        extra = (
            f"\n\nЭто попытка повторного поиска (номер {rewrite_count}).\n"
            f"Причина REWRITE: {state.get('check_reason', '')}\n"
            f"Фокус повторного поиска: {state.get('rewrite_focus', '')}\n"
            f"Предыдущий JSON-ответ: {state.get('answer', '')}\n"
            f"Предыдущий rag_prompt: {state.get('rag_prompt', '')}\n"
            f"Предыдущий reranker_prompt: {state.get('reranker_prompt', '')}\n"
            "Сгенерируй новые rag_prompt и reranker_prompt, чтобы закрыть пробелы."
        )

    human = HumanMessage(
        content=(
            f"Запрос пользователя:\n{input_query}\n"
            f"{extra}"
        )
    )
    
    llm = PARAMS_2.llm
    prompts = llm.with_structured_output(RetrievalPrompts, strict=True).invoke(
        [system_message, human]
    )
    out["rag_prompt"] = prompts.rag_prompt.strip() or input_query
    out["reranker_prompt"] = prompts.reranker_prompt.strip() or input_query

    return out


def rag_search_node(state: Agent2State) -> Agent2State:
    rag_prompt = state["rag_prompt"]
    reranker_prompt = state["reranker_prompt"]
    chunks = _rag_search_and_rerank(
        rag_prompt, reranker_prompt, state.get("output_model")
    )
    rag_context = format_rag_context(chunks)
    logger.info(
        f"[agent_2] RAG search completed (rag_prompt / reranker_prompt lengths: "
        f"{len(rag_prompt)} / {len(reranker_prompt)})"
    )
    return {"rag_context": rag_context}


def answer_node(state: Agent2State) -> Agent2State:

    def update_previous_answer_with_new_answer(previous_answer: str | None, new_answer: str) -> str:
        """Возвращает прошлый ответ, обновленный текущим ответом"""
        
        def _select_fields_to_update(previous_answer: str) -> list[str]:
            dct = json.loads(previous_answer)
            fields_to_update = [k for k, v in dct.items() if bool(v) is False]
            return fields_to_update
        
        if previous_answer:
            updated_answer = json.loads(previous_answer)  # начинаем обновлять прошлый ответ
            new_answer = json.loads(new_answer)
            fields_to_update = _select_fields_to_update(previous_answer)
            for field in fields_to_update:
                updated_answer[field] = new_answer[field]
            return json.dumps(updated_answer, ensure_ascii=False)
        else:
            return new_answer
    
    llm = PARAMS_2.llm
    input_query = state["input_query"]
    rag_context = state["rag_context"]
    output_model = state["output_model"]
    previous_answer = state.get("answer")

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
                f"Задача:\n{input_query}\n\n"
                f"RAG-контекст:\n{rag_context}"
            )
        ),
    ]
    
    try:
        current_response = llm.with_structured_output(output_model, strict=True).invoke(messages)
        new_answer = current_response.model_dump_json()
        updated_answer = update_previous_answer_with_new_answer(previous_answer, new_answer)
        return {"answer": updated_answer, "rag_contexts": [rag_context]}
    
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
                            f"Схема (Pydantic модель): {output_model.__name__}\n"
                            f"Задача:\n{input_query}\n\n"
                            f"RAG-контекст:\n{rag_context}"
                        )
                    ),
                ]
            )
            response_json = validate_and_dump_json_str(
                state['output_model'], str(getattr(raw, "content", raw))
            )
            new_answer = response_json
            updated_answer = update_previous_answer_with_new_answer(previous_answer, new_answer)
            return {"answer": updated_answer, "rag_contexts": [rag_context]}
        except Exception:
            logger.exception(
                "Structured output validation failed in answer_node in fallback №1. "
                "Doing fallback №2 and return empty dict"
            )
            return {"answer": "{}", "rag_contexts": [rag_context]}


class AnswerCheck(BaseModel):
    decision: Literal["OK", "REWRITE"] = Field(
        ...,
        description="OK, если контекста хватило; иначе REWRITE.",
    )
    reason: str = Field(
        ...,
        description="Краткая причина решения."
    )
    rewrite_focus: Optional[str] = Field(
        None,
        description="Что именно нужно найти/уточнить при следующей генерации retrieval-промптов.",
    )


def check_node(state: Agent2State) -> Agent2State:
    llm = PARAMS_2.llm
    input_query = state["input_query"]
    rag_contexts = state["rag_contexts"]
    answer = state["answer"]
    
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
                f"Задача (что нужно извлечь):\n{input_query}\n\n"
                f"RAG-контекст:\n{'\n'.join(rag_contexts)}\n\n"
                f"Ответ (JSON):\n{answer}"
            )
        ),
    ]

    check = llm.with_structured_output(AnswerCheck, strict=True).invoke(messages)
    return {
        "check_decision": check.decision,
        "check_reason": check.reason,
        "rewrite_focus": check.rewrite_focus or "",
        "output_model": state["output_model"]  # для логов
    }


def route_after_check(state: Agent2State) -> str:
    if state.get("check_decision") == "REWRITE" and state.get("rewrite_count", 0) < 2:
        return "generate_retrieval_prompts_node"
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
    builder.add_node("generate_retrieval_prompts_node", generate_retrieval_prompts_node)
    builder.add_node("rag_search_node", rag_search_node)
    builder.add_node("answer_node", answer_node)
    builder.add_node("check_node", check_node)

    builder.add_edge(START, "generate_retrieval_prompts_node")
    builder.add_edge("generate_retrieval_prompts_node", "rag_search_node")
    builder.add_edge("rag_search_node", "answer_node")
    builder.add_edge("answer_node", "check_node")
    builder.add_conditional_edges(
        "check_node",
        route_after_check,
        {"generate_retrieval_prompts_node": "generate_retrieval_prompts_node", END: END},
    )

    return builder.compile()
