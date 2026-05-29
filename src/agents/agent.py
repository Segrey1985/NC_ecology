import json
from pathlib import Path
from operator import add
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Literal, Optional, TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_core.language_models import BaseChatModel

from config.config_file import cfg, Config
from src.llm import LlmModel
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
from src.utils.utils import get_part_names_for_model


@dataclass(frozen=True)
class GraphResources:
    """Ресурсы экземпляра графа"""
    collection_name: str
    qdrant_service: QdrantService
    llm: BaseChatModel
    runtime_cfg: Config


class Agent2State(TypedDict):
    # input
    input_query: str

    # rag
    rag_prompts: list[str]
    rag_context: str
    rag_contexts: Annotated[list[str], add]
    reranker_prompts: list[str]

    # agent_node
    answer: str

    # optional loop (rewrite_focus и fields_to_rewrite из check при REWRITE)
    check_decision: Literal["OK", "REWRITE"]
    check_reason: str
    rewrite_focus: str
    fields_to_rewrite: list[str]
    verified_fields: Annotated[list[str], add]
    rewrite_count: int


def _rag_search_and_rerank(
    resources: GraphResources,
    rag_prompts: list[str],
    reranker_prompts: list[str],
    output_model: type[BaseModel] | None,
    chapter_module_path: str,
) -> list[str]:

    qdrant_service = resources.qdrant_service
    collection_name = resources.collection_name
    reranker_model = resources.runtime_cfg.RERANKER_MODEL

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
        part_names=get_part_names_for_model(output_model, chapter_module_path),
    )
    merged = merge_retrieval_results(query_points_tuple)
    texts = chunks_to_texts(merged)
    if not texts:
        return []

    ranked_child_chunks = rerank_with_expanded_queries(
        reranker_prompts, texts, reranker_model, top_n=5
    )

    if not ranked_child_chunks:
        logger.warning("? reranker returned no results ?; fallback to retrieval results")
        return [chunk.parent_text for chunk in merged[:5]]

    parent_texts: list[str] = []
    for child_chunk in ranked_child_chunks:
        index = child_chunk["index"]
        if 0 <= index < len(merged):
            parent_texts.append(merged[index].parent_text)
        else:
            logger.warning("[agent_2] reranker returned invalid index: %r", child_chunk)
    return parent_texts


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


def _get_output_model(config: RunnableConfig) -> type[BaseModel]:
    output_model = config.get("configurable", {}).get("output_model")
    if output_model is None:
        raise ValueError("RunnableConfig.configurable.output_model не передан.")
    return output_model


def _get_chapter_module_path(config: RunnableConfig) -> str:
    pth = config.get("metadata", {}).get("chapter_module_path")
    if not pth:
        raise RuntimeError("В config['metadata'] отсутствует ключ chapter_module_path.")
    return pth

def generate_retrieval_prompts_node(
    state: Agent2State, resources: GraphResources
) -> Agent2State:
    
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
    
    llm = resources.llm
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


def rag_search_node(
    state: Agent2State, config: RunnableConfig, resources: GraphResources
) -> Agent2State:
    rag_prompts = state["rag_prompts"]
    reranker_prompts = state["reranker_prompts"]
    chunks = _rag_search_and_rerank(
        resources, rag_prompts, reranker_prompts, _get_output_model(config), _get_chapter_module_path(config)
    )
    rag_context = format_rag_context(chunks)
    logger.info(
        f"[agent_2] RAG search completed "
        f"(rag_prompts={len(rag_prompts)}, reranker_prompts={len(reranker_prompts)})"
    )
    return {"rag_context": rag_context}


def answer_node(
    state: Agent2State, config: RunnableConfig, resources: GraphResources
) -> Agent2State:

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

    llm = resources.llm
    input_query = state["input_query"]
    rag_context = state["rag_context"]
    output_model = _get_output_model(config)
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
                output_model, str(getattr(raw, "content", raw))
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


def check_node(
    state: Agent2State, config: RunnableConfig, resources: GraphResources
) -> Agent2State:
    llm = resources.llm
    input_query = state["input_query"]
    rag_contexts = state["rag_contexts"]
    answer = state["answer"]
    
    output_model = _get_output_model(config)
    
    # все поля схемы
    schema_field_names = list(output_model.model_fields.keys())
    
    # "хорошие" поля по которым не было вопросов
    old_verified_fields = state.get("verified_fields", [])

    # поля требующие проверки
    _fields_need_to_check = list(set(schema_field_names) - set(old_verified_fields))
    if sorted(_fields_need_to_check) == sorted(schema_field_names):
        _fields_need_to_check = 'все поля.'

    system_message = """
Проверь, можно ли считать ответ корректным заполнением схемы на основе RAG-контекста.
Проверяй только __поля_требующие_проверки__.
Удостоверься, что __поля_требующие_проверки__ в ответе не противоречат остальным полям.
Используй только имена полей из переданного списка допустимых полей схемы.

1) Верни OK, если ответ можно использовать; fields_to_rewrite при этом — пустой список.

2) Верни REWRITE, если RAG-контекст не даёт достаточно данных, есть явные пробелы и нужно
переформулировать запрос для поиска.
При REWRITE укажи в fields_to_rewrite имена полей, которые нужно перезаписать
(некорректные, неподтверждённые контекстом, пустые при наличии данных в задаче).
    """.strip()

    system_message = SystemMessage(system_message)
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
    }


def route_after_check(state: Agent2State) -> str:
    if state.get("check_decision") == "REWRITE" and state.get("rewrite_count", 0) < 2:
        return "generate_retrieval_prompts_node"
    return END


def init_graph(
    collection_name: str, project_parts_path: Path | None, runtime_cfg: Config | None = None
):
    """
    Инициализирует параметры и собирает граф для работы с минисхемами JSON Schema.
    """
    
    runtime_cfg = runtime_cfg or cfg
    
    qdrant_service = build_qdrant_service(runtime_cfg)

    if not qdrant_service.client.collection_exists(collection_name):
        if not project_parts_path:
            raise ValueError(
                f"Коллекция {collection_name} не существует. "
                f"Требуется создание коллекции из project_parts_path. "
                f"Аргумент project_parts_path не передан."
            )
        logger.info(f"Creating new collection <{collection_name}>")
        create_collection(qdrant_service, collection_name)
        project_parts = create_project_parts(project_parts_path=project_parts_path)
        fill_collection(qdrant_service, collection_name, project_parts)
    else:
        logger.info(f"Found existing collection: <{collection_name}>")

    llm = LlmModel(model_type="ai_tunnel", model_name=runtime_cfg.MODEL_NAME).create()
    
    resources = GraphResources(
        collection_name=collection_name,
        qdrant_service=qdrant_service,
        llm=llm,
        runtime_cfg=runtime_cfg,
    )

    builder = StateGraph(Agent2State)
    builder.add_node(
        "generate_retrieval_prompts_node",
        lambda state: generate_retrieval_prompts_node(state, resources),
    )
    builder.add_node(
        "rag_search_node",
        lambda state, config: rag_search_node(state, config, resources),
    )
    builder.add_node(
        "answer_node",
        lambda state, config: answer_node(state, config, resources),
    )
    builder.add_node(
        "check_node",
        lambda state, config: check_node(state, config, resources),
    )

    builder.add_edge(START, "generate_retrieval_prompts_node")
    builder.add_edge("generate_retrieval_prompts_node", "rag_search_node")
    builder.add_edge("rag_search_node", "answer_node")
    builder.add_edge("answer_node", "check_node")
    builder.add_conditional_edges(
        "check_node",
        route_after_check,
        {"generate_retrieval_prompts_node": "generate_retrieval_prompts_node", END: END},
    )

    return builder.compile(), resources
