import json
from pathlib import Path
from docxtpl import DocxTemplate

from config.config_file import cfg
from src.utils.logger import logger


def fill_docx_template(
    template_path: Path,
    data: Path | dict,
    output_docx_path: Path,
) -> None:
    """
    Заполнение шаблона docx с помощью json/dict и сохранение копии
    :param template_path: Путь к шаблону.
    :param data: Путь к json файлу или словарь
    :param output_docx_path:  Путь к сохраненному объекту.
    """
    if not isinstance(data, dict):
        try:
            with open(data, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Путь к json файлу не найден: {data}")

    doc = DocxTemplate(template_path)
    doc.render(data)
    doc.save(output_docx_path)
    logger.debug(f"Шаблон заполнен. Копия сохранена в {output_docx_path}")


if __name__ == "__main__":
    pass

