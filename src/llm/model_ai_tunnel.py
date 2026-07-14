import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import langchain_anthropic.chat_models as _anthropic_module
from dotenv import load_dotenv
from config.config_file import cfg


def _clean_schema_for_claude(schema: Any) -> Any:
    """
    Рекурсивно очищает JSON Schema от полей, не поддерживаемых Claude:
    - Удаляет кастомные ключи (vanish и любые другие нестандартные)
    - Заменяет minItems > 1 на minItems: 1
    """
    STANDARD_SCHEMA_KEYS = {
        "type", "properties", "required", "additionalProperties",
        "items", "anyOf", "allOf", "oneOf", "not", "enum",
        "const", "title", "description", "default", "examples",
        "format", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf", "minLength", "maxLength", "pattern",
        "minItems", "maxItems", "uniqueItems", "minProperties",
        "maxProperties", "if", "then", "else", "definitions",
        "$ref", "$defs", "$schema", "$id", "nullable",
    }
    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            # Удаляем нестандартные ключи
            if k not in STANDARD_SCHEMA_KEYS:
                continue
            if k == "minItems":
                cleaned[k] = min(int(v), 1)
                continue
            cleaned[k] = _clean_schema_for_claude(v)
        return cleaned
    elif isinstance(schema, list):
        return [_clean_schema_for_claude(item) for item in schema]
    return schema


# Патчим convert_to_anthropic_tool на уровне модуля
_original_convert = _anthropic_module.convert_to_anthropic_tool


def _patched_convert_to_anthropic_tool(tool, *, strict=None):
    result = _original_convert(tool, strict=strict)
    if "input_schema" in result:
        result["input_schema"] = _clean_schema_for_claude(result["input_schema"])
    return result


_anthropic_module.convert_to_anthropic_tool = _patched_convert_to_anthropic_tool


def create_model(model_name: str):
    load_dotenv()
    if "claude" in model_name.lower():
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        model = ChatAnthropic(
            model=model_name,
            api_key=anthropic_key,
            base_url="https://api.stormlab.tech",
            timeout=120,
            temperature=cfg.TEMPERATURE if cfg.TEMPERATURE is not None else 0.7,
            max_tokens=8192,
        )
    else:
        api_key = os.getenv("AI_TUNNEL_API_KEY")
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.aitunnel.ru/v1/",
            timeout=30,
            temperature=cfg.TEMPERATURE,
            use_responses_api=True,
            output_version="responses/v1",
        )
    return model


if __name__ == "__main__":
    model = create_model(cfg.MODEL_NAME)
    print(model.invoke("Hello World"))
