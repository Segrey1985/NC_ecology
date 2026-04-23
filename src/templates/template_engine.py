import json
from pathlib import Path
from docxtpl import DocxTemplate

from src.utils.logger import logger


def fill_template(
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


template_path = r"C:\Users\maxfi\Desktop\ПМООС\MANUS\template_chapters_1-2\Универсальный шаблон для глав _Аннотация_ и _Введение_\Анализ_и_введение.docx"
fill_template(Path(template_path), Path('data.json'), output_docx_path=Path("result.docx"))