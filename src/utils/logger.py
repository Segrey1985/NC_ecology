from pathlib import Path

from loguru import logger
import sys

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{name}</level>:{function}:{line} | <level>{message}</level>"
)
logger_file_format = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level: <8} | "
    "{name}:{function}:{line} | {message}"
)
logger.remove()
logger.add(sys.stdout, format=logger_format, colorize=True)


def add_output_log_file(output_path: Path) -> int:
    output_path.mkdir(parents=True, exist_ok=True)
    return logger.add(
        output_path / "run.log",
        format=logger_file_format,
        encoding="utf-8",
    )


def print_and_save_thread_logs(
    output_path: Path | None,
    thread_results: list[dict],
    name_key: str,
) -> None:
    for t in thread_results:
        name = t[name_key]
        log_lines = t["logs_lines"]
        print(f"\n--- Логи потока {name} ({len(log_lines)} записей) ---")
        for line in log_lines:
            print(line)
        print("--- конец логов потока ---\n")

    if not output_path or not thread_results:
        return

    logs_dir = output_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for t in thread_results:
        log_lines = t["logs_lines"]
        log_path = logs_dir / f"{t[name_key]}.log"
        log_path.write_text(
            "\n".join(log_lines) + ("\n" if log_lines else ""),
            encoding="utf-8",
        )


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
