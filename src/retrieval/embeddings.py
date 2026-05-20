import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config.config_file import cfg
from src.utils.logger import logger


def load_local_embedder(model_name: str):
    models_dir = cfg.BASE_DIR / "data" / "__local_models"
    model_name_str = model_name.replace("/", "_")
    model_path = models_dir / model_name_str
    device = cfg.DEVICE
    if os.path.exists(model_path):
        logger.debug(f"[embedder] [{model_name}] Loading local model from {model_path}...")
        embedder = SentenceTransformer(model_path.as_posix(), device=device)
    else:
        logger.debug(f"[embedder] [{model_name}] Loading model from HuggingFace...")
        embedder = SentenceTransformer(model_name, device=device)
        embedder.save(
            model_path
        )  # внутри метода save есть makedirs(..., exist_ok=True)
    logger.debug(f"[embedder] [{model_name}] Model loading complete.")
    return embedder


def load_openai_embeddings_client() -> OpenAI:
    client = OpenAI(
        api_key=cfg.AI_TUNNEL_API_KEY, base_url="https://api.aitunnel.ru/v1/"
    )
    return client


class OpenAIEmbedder:
    def __init__(self, model):
        self.client = load_openai_embeddings_client()
        self.model = model

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([x.embedding for x in response.data])

        return all_embeddings


def init_openai_embedder(model_name: str):
    logger.debug(f"[embedder] [{model_name}] init OpenAI embedder...")
    return OpenAIEmbedder(model=model_name)


if __name__ == "__main__":

    model_name = "Qwen/Qwen3-Embedding-8B"
    embedder = load_local_embedder(model_name)
    vectors = embedder.encode(["раз", "два"], batch_size=2)
    print(vectors)

    openai_embedder = init_openai_embedder("text-embedding-3-small")
    vectors = openai_embedder.encode(["раз", "два"])
    print(vectors)

    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    model_name = "Qwen/Qwen3-Embedding-8B"
    embedder = load_local_embedder(model_name)
    embeddings = embedder.encode(sentences, batch_size=32)
    similarities = embedder.similarity(embeddings, embeddings)
    print(similarities.shape)

    openai_embedder = init_openai_embedder("text-embedding-3-small")
    embeddings = openai_embedder.encode(sentences, batch_size=32)
    print(type(embeddings))
    print(embeddings)
