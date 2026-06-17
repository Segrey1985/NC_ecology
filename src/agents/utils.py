from pydantic import BaseModel, create_model
from langchain_core.runnables import RunnableConfig


def merge_unique_chunks(left: list[str], right: list[str]) -> list[str]:
    return list(dict.fromkeys(left + right))


def get_output_model(config: RunnableConfig) -> type[BaseModel]:
    output_model = config.get("configurable", {}).get("output_model")
    if output_model is None:
        raise ValueError("RunnableConfig.configurable.output_model не передан.")
    return output_model


def get_chapter_module_path(config: RunnableConfig) -> str:
    pth = config.get("metadata", {}).get("chapter_module_path")
    if not pth:
        raise RuntimeError("В config['metadata'] отсутствует ключ chapter_module_path.")
    return pth


def _build_partial_output_model(
    output_model: type[BaseModel],
    field_names: list[str],
) -> type[BaseModel]:
    """Подмодель только с полями, которые нужно перегенерировать."""
    
    field_definitions = {}
    for name in field_names:
        if name in output_model.model_fields:
            value_type = output_model.model_fields[name].annotation
            value_field = output_model.model_fields[name]
            field_definitions[name] = (value_type, value_field)
    
    return create_model(
        f"{output_model.__name__}Partial",
        __config__=output_model.model_config,
        **field_definitions,
    )


def select_generation_model(
    output_model: type[BaseModel],
    previous_answer: str | None,
    fields_to_rewrite: list[str],
) -> type[BaseModel]:
    """Полная схема при первом проходе; при rewrite — только поля из fields_to_rewrite."""
    if not previous_answer:
        return output_model
    
    valid_fields = [field for field in fields_to_rewrite if field in output_model.model_fields]
    if not valid_fields:
        return output_model
    
    return _build_partial_output_model(output_model, valid_fields)