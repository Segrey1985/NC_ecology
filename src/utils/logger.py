from loguru import logger
import sys

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{name}</level>:<black>{function}</black>:<black>{line}</black> | <level>{message}</level>"
)
logger.remove()
logger.add(sys.stdout, format=logger_format, colorize=True)


def loguru_timer(func):
    def wrapper(*args, **kwargs):
        logger.info(f"START {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"END {func.__name__}")
        return result

    return wrapper

if __name__ == "__main__":
    logger.success(f"START {__name__}")
    logger.info(f"START {__name__}")
    logger.debug(f"START {__name__}")
    logger.warning(f"START {__name__}")
    logger.error(f"START {__name__}")
