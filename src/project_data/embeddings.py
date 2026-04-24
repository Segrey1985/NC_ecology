import os
from sentence_transformers import SentenceTransformer

from config.config_file import cfg
from src.utils.logger import logger


def load_embeddings(model_name: str):
    models_dir = cfg.BASE_DIR / "data" / "__local_models"
    model_name_str = model_name.replace("/", "_")
    model_path = models_dir / model_name_str
    if os.path.exists(model_path):
        logger.debug(f"Loading local model from {model_path}...")
        embedder = SentenceTransformer(model_path.as_posix(), device="cuda")
    else:
        logger.debug(f"Loading model from HuggingFace...")
        embedder = SentenceTransformer(model_name, device="cuda")
        embedder.save(model_path)
    logger.debug(f"Loading model is finished.")
    return embedder


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-Embedding-8B"
    embedder = load_embeddings(model_name)
