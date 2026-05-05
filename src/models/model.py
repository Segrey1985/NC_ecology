from typing import Literal
import importlib
from langchain_core.language_models import BaseChatModel

MODEL_REGISTRY = {
    "ollama": "src.models.model_ollama:create_model",
    "ai_tunnel": "src.models.model_ai_tunnel:create_model",
}


class LlmModel:
    def __init__(self, model_type: Literal["ollama", "ai_tunnel"], model_name: str):
        self.model_type = model_type
        self.model_name = model_name

    def create(self) -> BaseChatModel:
        try:
            path = MODEL_REGISTRY[self.model_type]
        except KeyError:
            raise ValueError(f"Unknown model: {self.model_type}")

        module_path, attr_name = path.split(":")

        module = importlib.import_module(module_path)
        create_model_func = getattr(module, attr_name)
        return create_model_func(self.model_name)


if __name__ == "__main__":
    # llm_model = LlmModel(model_type="ollama", model_name="qwen3.5:35b").create()
    # print(llm_model.invoke("Привет"))

    llm_model = LlmModel(model_type="ai_tunnel", model_name="gpt-5-nano").create()
    print(llm_model.invoke("Привет"))
