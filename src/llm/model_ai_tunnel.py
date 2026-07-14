import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from config.config_file import cfg

def create_model(model_name: str):
    load_dotenv()
    # Определяем провайдера по имени модели
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
