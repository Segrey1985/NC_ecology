import json
from pathlib import Path
from operator import add
from typing import Literal, Optional, TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config.config_file import cfg, Config
from src.models import LlmModel
from src.retrieval.qdrant import (
    QdrantService,
    build_qdrant_service,
    create_collection,
    create_project_parts,
    fill_collection,
)
from src.retrieval.retrieval_expansion import (
    chunks_to_texts,
    merge_retrieval_results,
    search_by_multi_rag_queries,
)
from src.retrieval.reranker_expansion import rerank_with_expanded_queries
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
        self.runtime_cfg: Config | None = None


PARAMS_2 = GraphParams()  # глобальный объект параметров (для agent_2)


class Agent2State(TypedDict):
    # input
    input_query: str

    # rag
    rag_prompts: list[str]
    rag_context: str
    rag_contexts: Annotated[list[str], add]
    reranker_prompts: list[str]

    # agent_node (output_model — класс pydantic-схемы для structured output)
    answer: str
    output_model: type[BaseModel]

    # optional loop (rewrite_focus и fields_to_rewrite из check при REWRITE)
    check_decision: Literal["OK", "REWRITE"]
    check_reason: str
    rewrite_focus: str
    fields_to_rewrite: list[str]
    verified_fields: Annotated[list[str], add]
    rewrite_count: int


def _rag_search_and_rerank(
    rag_prompts: list[str],
    reranker_prompts: list[str],
    output_model: type[BaseModel] | None,
) -> list[str]:
    qdrant_service = PARAMS_2.qdrant_service
    collection_name = PARAMS_2.collection_name
    if qdrant_service is None or collection_name is None:
        raise RuntimeError(
            "Qdrant не инициализирован. Сначала вызовите init_graph_2()."
        )

    queries = [q.strip() for q in rag_prompts if q and q.strip()]
    if not queries:
        return []

    query_points_tuple = search_by_multi_rag_queries(
        queries,
        qdrant_service,
        collection_name,
        limit=50,
        part_names=get_part_names_for_model(output_model),
    )
    merged = merge_retrieval_results(query_points_tuple)
    texts = chunks_to_texts(merged)
    if not texts:
        return []
    
    reranked = rerank_with_expanded_queries(reranker_prompts, texts, PARAMS_2.runtime_cfg.RERANKER_MODEL, top_n=5)
    return [chunk for chunk, _score in reranked]


class RagPrompt(BaseModel):
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


class RerankPrompt(BaseModel):
    reranker_prompt: str = Field(
        ...,
        description=(
            "Короткая формулировка целевой информации для cross-encoder reranker. "
            "Описывает, какая именно информация должна присутствовать во фрагменте. "
            "Без длинных перечислений синонимов и служебного текста."
        ),
    )


class RetrievalPrompts(BaseModel):
    """Промпты для retrieval-этапа: dense retrieval и cross-encoder reranking."""

    rag_prompts: list[RagPrompt] = Field(
        ...,
        min_length=3,
        max_length=3,
        description=(
            "Ровно 3 разных семантических запроса для dense-поиска в Qdrant. "
            "Каждый — отдельный угол: синонимы, аббревиатуры, формулировки из ПД/СП. "
            "Запросы не должны дублировать друг друга дословно."
        ),
    )

    reranker_prompts: list[RerankPrompt] = Field(
        ...,
        min_length=3,
        max_length=3,
        description=(
            "Ровно 3 разных коротких запроса для cross-encoder reranker. "
            "Каждый описывает отдельный аспект целевой информации во фрагменте; "
            "формулировки не дублируют друг друга дословно."
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
        "Ты готовишь промпты для RAG по проектной документации (строительство, экология).\n"
        "Твоя задача — извлечь из запроса пользователя ключевые смыслы для эффективного RAG-поиска.\n"
        "Отвечай на русском языке, используя терминологию отрасли."
    )

    extra = ""
    if rewrite_count > 0:
        extra = (
            f"\n\nЭто попытка повторного поиска (номер {rewrite_count}).\n\n"
            f"Причина REWRITE: {state.get('check_reason', '')}\n\n"
            f"Фокус повторного поиска: {state.get('rewrite_focus', '')}\n\n"
            f"Поля требующие повторного поиска: {state.get('fields_to_rewrite', [])}\n\n"
            # f"Предыдущий JSON-ответ: {state.get('answer', '')}\n\n"
            # f"Предыдущие rag_prompts: {state.get('rag_prompts', [])}\n\n"
            # f"Предыдущие reranker_prompts: {state.get('reranker_prompts', [])}\n\n"
            "Сгенерируй новые rag_prompts и reranker_prompts, чтобы закрыть пробелы."
        )

    human = HumanMessage(
        content=(
            f"Запрос пользователя:\n{input_query}"
            f"{extra}"
        )
    )
    
    llm = PARAMS_2.llm
    prompts = llm.with_structured_output(RetrievalPrompts, strict=True).invoke(
        [system_message, human]
    )
    expanded = [
        item.rag_prompt.strip()
        for item in prompts.rag_prompts
        if item.rag_prompt.strip()
    ]
    if len(expanded) != 3:
        logger.warning("len(RetrievalPrompts.rag_prompts) != 3: %s", expanded)
    while len(expanded) < 3:
        expanded.append(input_query)
    out["rag_prompts"] = expanded[:3]

    rerank_expanded = [
        item.reranker_prompt.strip()
        for item in prompts.reranker_prompts
        if item.reranker_prompt.strip()
    ]
    if len(rerank_expanded) != 3:
        logger.warning("len(RetrievalPrompts.reranker_prompts) != 3: %s", rerank_expanded)
    while len(rerank_expanded) < 3:
        rerank_expanded.append(input_query)
    out["reranker_prompts"] = rerank_expanded[:3]

    return out


def rag_search_node(state: Agent2State) -> Agent2State:
    rag_prompts = state["rag_prompts"]
    reranker_prompts = state["reranker_prompts"]
    chunks = _rag_search_and_rerank(
        rag_prompts, reranker_prompts, state.get("output_model")
    )
    rag_context = format_rag_context(chunks)
    logger.info(
        f"[agent_2] RAG search completed "
        f"(rag_prompts={len(rag_prompts)}, reranker_prompts={len(reranker_prompts)})"
    )
    return {"rag_context": rag_context}


def answer_node(state: Agent2State) -> Agent2State:

    def update_previous_answer_with_new_answer(
        previous_answer: str | None,
        new_answer: str,
        fields_to_rewrite: list[str],
    ) -> str:
        """Возвращает прошлый ответ, обновленный текущим ответом."""
        
        # если есть прошлый ответ - обновляем его
        if previous_answer:
            updated_answer = json.loads(previous_answer)
            new_answer_dict = json.loads(new_answer)
            
            # если есть прошлый ответ, но отсутствует список полей для обновления - обновляем все
            if not fields_to_rewrite:
                logger.warning(
                    "[update answer] (есть previous_answer; нет fields_to_rewrite): перезапишем все поля ответа"
                )
                return new_answer
            
            for field in fields_to_rewrite:
                updated_answer[field] = new_answer_dict[field]
            return json.dumps(updated_answer, ensure_ascii=False)
        return new_answer

    llm = PARAMS_2.llm
    input_query = state["input_query"]
    rag_context = state["rag_context"]
    output_model = state["output_model"]
    previous_answer = state.get("answer")
    fields_to_rewrite = state.get("fields_to_rewrite", [])

    system_message = SystemMessage(
        "Ты помощник по извлечению данных по строительному проекту.\n"
        "Заполни JSON строго по переданной схеме и только на основе RAG-контекста.\n"
        "Если значения нет в контексте, не выдумывай. Используй только допустимые схемой "
        "пустые значения (например, null для Optional-полей или пустые списки там, где это уместно).\n"
        "Не добавляй поля, которых нет в схеме."
        "Не используй 'на основе контекста', 'в rag контексте найдено', 'в представленном материале найдено'"
        " и прочие конструкции, упоминающие источник текста."
    )
    messages = [
        system_message,
        HumanMessage(
            content=(
                f"Задача: заполни json-схему на основе предложенного контекста:\n"
                f"RAG-контекст:\n{rag_context}"
            )
        ),
    ]
    
    try:
        current_response = llm.with_structured_output(output_model, strict=True).invoke(messages)
        new_answer = current_response.model_dump_json()
        updated_answer = update_previous_answer_with_new_answer(
            previous_answer, new_answer, fields_to_rewrite
        )
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
            updated_answer = update_previous_answer_with_new_answer(
                previous_answer, new_answer, fields_to_rewrite
            )
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
    fields_to_rewrite: list[str] = Field(
        default_factory=list,
        description=(
            "Имена полей из JSON-ответа, которые нужно перезаписать при следующем извлечении. "
            "Пустой список при decision=OK. При decision=REWRITE — перечисли поля с ошибочными, "
            "неподтверждёнными контекстом или отсутствующими значениями."
        ),
    )


def check_node(state: Agent2State) -> Agent2State:
    llm = PARAMS_2.llm
    input_query = state["input_query"]
    rag_contexts = state["rag_contexts"]
    answer = state["answer"]
    
    output_model = state["output_model"]
    
    # все поля схемы
    schema_field_names = list(output_model.model_fields.keys())
    
    # "хорошие" поля по которым не было вопросов
    old_verified_fields = state.get("verified_fields", [])

    # поля требующие проверки
    _fields_need_to_check = list(set(schema_field_names) - set(old_verified_fields))
    if sorted(_fields_need_to_check) == sorted(schema_field_names):
        _fields_need_to_check = 'все поля.'

    system_message = SystemMessage(
        "Проверь, можно ли считать ответ корректным заполнением схемы на основе RAG-контекста.\n"
        "Проверяй только __поля_требующие_проверки__.\n"
        "Верни OK, если ответ можно использовать; fields_to_rewrite при этом — пустой список.\n"
        "Верни REWRITE, если RAG-контекст не даёт достаточно данных/есть явные пробелы и нужно "
        "переформулировать запрос для поиска.\n"
        "При REWRITE укажи в fields_to_rewrite имена полей JSON, которые нужно перезаписать "
        "(некорректные, неподтверждённые контекстом, пустые при наличии данных в задаче).\n"
        "Используй только имена полей из переданного списка допустимых полей схемы."
    )
    messages = [
        system_message,
        HumanMessage(
            content=(
                f"Задача (что нужно извлечь):\n{input_query}\n\n"
                f"__поля_требующие_проверки__: {_fields_need_to_check}\n\n"
                f"Допустимые поля схемы: {schema_field_names}\n\n"
                f"RAG-контекст:\n{'\n'.join(rag_contexts)}\n\n"
                f"Ответ (JSON):\n{answer}"
            )
        ),
    ]

    response = llm.with_structured_output(AnswerCheck, strict=True).invoke(messages)
    
    # берем поля для перезаписи и исключаем из них "хорошие" поля
    fields_to_rewrite = [field for field in response.fields_to_rewrite
                         if ((field in schema_field_names) and (field not in old_verified_fields))]
    
    # добавляем новые "хорошие" поля = все поля - поля для перезаписи
    verified_fields = [field for field in list(set(schema_field_names) - set(fields_to_rewrite))
                       if field not in old_verified_fields]

    return {
        "check_decision": response.decision,
        "check_reason": response.reason,
        "rewrite_focus": response.rewrite_focus or "",
        "fields_to_rewrite": fields_to_rewrite,
        "verified_fields": verified_fields,  # add без [] т.к. verified_fields: list
        "output_model": output_model,
    }


def route_after_check(state: Agent2State) -> str:
    if state.get("check_decision") == "REWRITE" and state.get("rewrite_count", 0) < 2:
        return "generate_retrieval_prompts_node"
    return END


def init_graph_2(collection_name: str, project_parts_path: Path | None, runtime_cfg: Config | None):
    """
    Инициализирует параметры и собирает граф для работы с минисхемами JSON Schema.
    """
    
    runtime_cfg = runtime_cfg or cfg
    
    PARAMS_2.collection_name = collection_name
    PARAMS_2.qdrant_service = build_qdrant_service(runtime_cfg)
    PARAMS_2.runtime_cfg = runtime_cfg

    if not PARAMS_2.qdrant_service.client.collection_exists(collection_name):
        if not project_parts_path:
            raise ValueError(
                f"Коллекция {collection_name} не существует. "
                f"Требуется создание коллекции из project_parts_path. "
                f"Аргумент project_parts_path не передан."
            )
        logger.info(f"[agent_2] Создаю новую коллекцию <{collection_name}>")
        project_parts = create_project_parts(
            project_parts_path=project_parts_path, embedder=PARAMS_2.qdrant_service.model
        )
        create_collection(PARAMS_2.qdrant_service, collection_name)
        fill_collection(PARAMS_2.qdrant_service, collection_name, project_parts)
    else:
        logger.info(f"[agent_2] Найдена существующая коллекция <{collection_name}>")

    PARAMS_2.llm = LlmModel(model_type="ai_tunnel", model_name=runtime_cfg.MODEL_NAME).create()

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
