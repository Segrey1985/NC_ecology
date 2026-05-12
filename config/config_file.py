import warnings
import torch
import pathlib
from pathlib import Path
from typing import Literal
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
    
    # other
    "glm5": "glm-5",
}


class Config(BaseSettings):
    HF_TOKEN: str
    AI_TUNNEL_API_KEY: str
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MODEL_NAME: str = models["gpt54mini"]
    TEMPERATURE: float | None = None
    BASE_DIR: Path = BASE_DIR
    EMBEDDINGS_MODEL_NAME: str = "Qwen/Qwen3-Embedding-8B"
    EMBEDDINGS_LOCAL: bool = False
    QDRANT_URL: str = "http://localhost:6333"
    RERANKER_MODEL: str = "qilowoq/bge-reranker-v2-m3-en-ru"
    USE_LANGFUSE: bool = True
    DEVICE: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

    DISCIPLINE_BY_NUMBER: dict[str, str] = {
        "1": "ПЗ",
        "2": "ПЗУ",
        "3": "АР",
        "4": "OK",
        "5": "ИОС",
        "5.1": "Система электроснабжения",
        "5.2": "Система водоснабжения",
        "5.3": "Система водоотведения",
        "5.4": "Отопление, вентиляция и кондиционирование воздуха, тепловые сети",
        "5.5": "Сети связи",
        "5.6": "Система газоснабжения",
    }


cfg = Config()

logger.info("\n\n\n\n\n---START----\n\n\n\n\n")
logger.info(f"{cfg.DEVICE=}")

if __name__ == "__main__":
    print(cfg)
