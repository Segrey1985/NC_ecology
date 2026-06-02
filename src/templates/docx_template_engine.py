import json
from pathlib import Path
from docxtpl import DocxTemplate
from jinja2 import Environment, ChainableUndefined

from config.config_file import cfg
from src.utils.logger import logger


class PlaceholderUndefined(ChainableUndefined):
    """ Класс плейсхолдера отсутствующий в data, но присутствующий в template """
    
    
    def __init__(self, *args, path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path or self._undefined_name
    
    
    def __getattr__(self, name):
        return type(self)(name=name, path=f"{self._path}.{name}")
    
    
    def __str__(self):
        return "{{" + "undefined " + self._path + "}}"


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
    doc.render(data, jinja_env=Environment(undefined=PlaceholderUndefined))
    doc.save(output_docx_path)
    logger.debug(f"Шаблон заполнен. Копия сохранена в {output_docx_path}")


if __name__ == "__main__":
    template = Path(r"C:\Users\maxfi\PycharmProjects\NC_ecology\src\ecology_chapters\chapter1\template.docx")
    fill_docx_template(
        template_path=template,
        data={},
        output_docx_path=template.parent / (template.name + "_out" + template.suffix),
    )

