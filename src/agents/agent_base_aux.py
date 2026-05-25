import json
import uuid
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.language_models import BaseChatModel

from src.utils.logger import logger
from config.config_file import cfg, Config
from config.langfuse_client import langfuse_config
from src.llm import LlmModel
from src.utils.utils import print_chunk
from src.pydantic_models.aux_models import AuxSchema

# --- state & resources ---


class AgentState(TypedDict):
    
    # строка в формате словаря
    chapter_0: str
    
    # ответ
    answer: str


@dataclass(frozen=True)
class GraphResources:
    """Ресурсы экземпляра графа"""
    llm: BaseChatModel
    runtime_cfg: Config


# --- tools ---


def _build_agent_prompt(chapter_0: str):
    return "\n\n".join(f"KEY: {k}\nVALUE: {v}" for k, v in json.loads(chapter_0).items())


# --- nodes ---


def aux_node(state: AgentState, resources: GraphResources) -> AgentState:
    
    # resources
    llm = resources.llm
    
    # state
    chapter_0: str = state.get("chapter_0", "")
    
    # code
    system_prompt = (
        "На основе извлеченной структуры данных по проекту, "
        "извлеки дополнительные поля."
    )
    human_prompt: str = _build_agent_prompt(chapter_0)
    
    messages = [SystemMessage(system_prompt), HumanMessage(human_prompt)]
    response = llm.with_structured_output(AuxSchema, strict=True).invoke(messages)
    print()
    return {"answer": response.model_dump_json()}
    


# --- инициализация графа ---


def init_graph(runtime_cfg: Config | None = None):
    """Инициализирует параметры и собирает граф."""
    
    runtime_cfg = runtime_cfg or cfg

    llm = LlmModel(model_type="ai_tunnel", model_name=runtime_cfg.MODEL_NAME).create()
    
    resources = GraphResources(
        llm=llm,
        runtime_cfg=runtime_cfg,
    )

    builder = StateGraph(AgentState)
    builder.add_node("aux_node", lambda state: aux_node(state, resources))
    builder.add_edge(START, "aux_node")
    builder.add_edge("aux_node", END)

    return builder.compile(), resources


if __name__ == "__main__":

    graph, _resources = init_graph(runtime_cfg=cfg)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    config.update(langfuse_config)
    
    input_chapter_0: str = """{
    "МОЩНОСТЬ_ОБЪЕКТА": "48,0 МВт",
    "КАДАСТРОВЫЙ_НОМЕР": "78:42:0018501:3309",
    "АДРЕС_ОБЪЕКТА": "Санкт-Петербург, посёлок Шушары, территория предприятия «Шушары», участок 22 (Пулковский)",
    "НАИМЕНОВАНИЕ_ПРОЕКТА": "Строительство автоматизированной газовой котельной",
    "ТИП_РАБОТ": "строительства",
    "РАЗРАБОТЧИК_РАЗДЕЛА": "ООО «СК»",
    "ОСНОВАНИЕ_ДЛЯ_ПРОЕКТИРОВАНИЯ": "договор № ЛЭ 01-09/10 от 15 ноября 2023 г. между Заказчиком – ООО «ЛСР.»",
    "НАЗНАЧЕНИЕ_ОБЪЕКТА": "обеспечение отоплением, вентиляцией и ГВС строящихся многоквартирных жилых домов и общественно-деловой застройки"
}""".strip()

    for chunk in graph.stream(
        input={
            "chapter_0": input_chapter_0,
        },
        stream_mode="updates",
        config=config,
    ):
        print_chunk(chunk)
