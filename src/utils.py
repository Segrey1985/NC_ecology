from io import BytesIO
from typing import Optional
import PyPDF2

# ___ PDF ___


def count_pages(file_path: str) -> int | None:
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return len(reader.pages)
    except Exception:
        return None


def extract_pages(
    input_pdf: str | bytes,
    pages_to_keep: list[int],
    output_pdf_path: Optional[str] = None,
) -> Optional[bytes]:
    """Извлечение страниц из PDF. Если output_pdf_path не задан, возвращает байты."""

    # Проверяем, что было передано: путь к файлу или байты
    if isinstance(input_pdf, bytes):
        input_pdf_file = BytesIO(input_pdf)
    else:
        input_pdf_file = open(input_pdf, "rb")

    try:
        reader = PyPDF2.PdfReader(input_pdf_file)
        writer = PyPDF2.PdfWriter()
        valid_pages = [x + 1 for x in range(len(reader.pages))]
        valid_pages_to_keep = [x for x in pages_to_keep if x in valid_pages]

        # Извлекаем указанные страницы
        for page_num in valid_pages_to_keep:
            # Нумерация страниц в PyPDF2 начинается с 0
            writer.add_page(reader.pages[page_num - 1])

        if output_pdf_path:
            # Записываем результат в новый PDF файл
            with open(output_pdf_path, "wb") as output_pdf_file:
                writer.write(output_pdf_file)
        else:
            # Создаем байтовый буфер для хранения результата
            output_buffer = BytesIO()
            writer.write(output_buffer)

            # Возвращаем байты PDF-файла
            return output_buffer.getvalue()

    finally:
        # Закрываем файл, если он был открыт
        if not isinstance(input_pdf, bytes):
            input_pdf_file.close()
