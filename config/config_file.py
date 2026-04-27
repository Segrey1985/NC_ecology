import pathlib
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    HF_TOKEN: str
    AI_TUNNEL_API_KEY: str
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    BASE_DIR: Path = BASE_DIR
    EMBEDDINGS_MODEL_NAME: str = "Qwen/Qwen3-Embedding-8B"
    EMBEDDINGS_LOCAL: bool = False


cfg = Config()

if __name__ == "__main__":
    print(cfg)
