from langfuse import get_client
from langfuse.langchain import CallbackHandler

from config.config_file import cfg

from dotenv import load_dotenv

load_dotenv()

# Initialize Langfuse client
langfuse = get_client()

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

langfuse_config = {"callbacks": [langfuse_handler]} if cfg.USE_LANGFUSE else {}
