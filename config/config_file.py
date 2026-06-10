import warnings
import torch
import pathlib
from pathlib import Path
from typing import Literal
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logger import logger


warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
)

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

models = {
    # clade
    "claude": "claude-sonnet-4.6",
    
    # gemini
    "gemini3.1pro": "gemini-3.1-pro-preview",
    "flash": "gemini-3-flash-preview",
    
    # gpt
    "gpt5.1": "gpt-5.1",
    "gpt54mini": "gpt-5.4-mini",
    "nano": "gpt-5.4-nano",
    
    # other
    "glm5": "glm-5",
}

embeddings_list = {
    "Qwen/Qwen3-Embedding-8B": {"is_local": True, "length": 4096},
    "text-embedding-3-small": {"is_local": False, "length": 1536},
    "text-embedding-3-large": {"is_local": False, "length": 3072},
}

rerankers_list = {
    "qilowoq/bge-reranker-v2-m3-en-ru": {"is_local": True},
    "rerank-4-pro": {"is_local": False},
    "rerank-v3.5": {"is_local": False}
}


class Config(BaseSettings):
    HF_TOKEN: SecretStr
    AI_TUNNEL_API_KEY: SecretStr
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    MODEL_NAME: str = models["gpt54mini"]
    TEMPERATURE: float | None = None
    BASE_DIR: Path = BASE_DIR
    EMBEDDINGS_MODEL_NAME: str = "text-embedding-3-large"
    EMBEDDINGS_LOCAL: bool = False
    QDRANT_URL: str = "http://localhost:6333"
    RERANKER_MODEL: str = "rerank-4-pro"
    USE_LANGFUSE: bool = True
    DEVICE: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def hf_token(self) -> str:
        return self.HF_TOKEN.get_secret_value()

    @property
    def ai_tunnel_api_key(self) -> str:
        return self.AI_TUNNEL_API_KEY.get_secret_value()
    
    # `key` - начало наименования файла до символа "_", `value` - значение part_name для записи в qdrant
    DISCIPLINE_BY_NUMBER: dict[str, str] = {
        "ИГИ": "ИГИ",
        "ИЭИ": "ИЭИ",
        "1": "ПЗ",
        "2": "ПЗУ",
        "3": "АР",
        "4": "КР",
        "5": "ИОС",
        "5.1": "Система электроснабжения",
        "5.2": "Система водоснабжения",
        "5.3": "Система водоотведения",
        "5.4": "ОВИК",
        "5.5": "Сети связи",
        "5.6": "Система газоснабжения",
        "6": "ТР",
        "6.1": "ТМ",
        "6.2": "ТП",
        "7": "ПОС",
        "8": "ООС",
        "9": "ПБ",
        "9.1": "ПБ",
        "9.2": "ПБ",
        "10": "ТБЭ",
        "прочее": "прочее"
    }


cfg = Config()

TestMode = Literal["off", "on", "mock", "filter"]


def build_runtime_config(test_mode: TestMode) -> Config:
    """Копия и доп. настройка параметров cfg"""
    runtime_config = cfg.model_copy(deep=True)
    
    if test_mode != "off":
        runtime_config.RERANKER_MODEL = "rerank-v3.5"
        runtime_config.EMBEDDINGS_MODEL_NAME = "text-embedding-3-small"
    
    return runtime_config


logger.info("\n\n\n\n\n---- START ----\n\n\n\n\n")
logger.info("---- cfg: ----")
logger.info('\n'.join(["\n"] + [f"{k}={v}" for k, v in cfg.model_dump().items()]))
