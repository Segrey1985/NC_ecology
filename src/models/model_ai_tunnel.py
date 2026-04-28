import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from config.config_file import cfg


def create_model(model_name):
    load_dotenv()
    api_key = os.getenv("AI_TUNNEL_API_KEY")
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.aitunnel.ru/v1/",
        timeout=30,
        temperature=cfg.TEMPERATURE,
    )
    return model


if __name__ == "__main__":
    model = create_model(cfg.MODEL_NAME)
    print(model.invoke("Hello World"))
