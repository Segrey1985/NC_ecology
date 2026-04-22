import glob
import pathlib
from io import BytesIO
from typing import Optional
import PyPDF2
from pdfminer.layout import LTTextBox, LTTextLine
from pdfminer.high_level import extract_pages as extract_pages_miner

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
            
            
def extract_text_with_miner_coords(
    pdf_path,
    y_tolerance: float = 5,
    page_numbers: list[int] | None = None,
):
    """
    Извлекает текст из PDF-файла с координатами строк, учитывая запас смещения по y.

    :param pdf_path: Путь к PDF-файлу (строка)
    :param y_tolerance: Запас смещения по y для группировки строк (по умолчанию 10)
    :param page_numbers: Список номеров страниц (0-based) для анализа. Если None — анализировать все.
    :return: Список строк, где каждая строка — это текст страницы с учетом структуры
    """
    text_with_coords = []

    # Извлекаем страницы из PDF
    pages_iter = (
        extract_pages_miner(pdf_path)
        if page_numbers is None
        else extract_pages_miner(pdf_path, page_numbers=page_numbers)
    )

    for page_layout in pages_iter:
        lines = []
        # Проходим по элементам страницы
        for element in page_layout:
            if isinstance(element, LTTextBox):
                # Обрабатываем текстовые блоки
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        # Получаем координаты и текст строки
                        x0, y0, x1, y1 = text_line.bbox
                        text = text_line.get_text().strip()
                        if text:  # Пропускаем пустые строки
                            lines.append((text, (x0, y0, x1, y1)))

        # Сортируем строки по y1 сверху вниз
        lines.sort(key=lambda x: -x[1][1])

        # Группируем строки по y-координате с учетом запаса смещения
        grouped_lines = []
        current_group = []
        prev_y = None

        for line in lines:
            text, (x0, y0, x1, y1) = line
            if prev_y is None or abs(y1 - prev_y) <= y_tolerance:
                current_group.append(line)
            else:
                # Сортируем текущую группу по x0 (слева направо)
                current_group.sort(key=lambda x: x[1][0])
                grouped_lines.append(current_group)
                current_group = [line]
            prev_y = y1

        # Добавляем последнюю группу
        if current_group:
            current_group.sort(key=lambda x: x[1][0])
            grouped_lines.append(current_group)

        # Объединяем текст из групп
        page_text = ""
        for group in grouped_lines:
            group_text = " ".join([text for text, _ in group])
            page_text += group_text + "\n"

        text_with_coords.append(page_text)

    return text_with_coords


def find_page_index_by_first_text(input_pdf: str | bytes, text: str) -> int | None:
    """
    Возвращает индекс страницы (нумерация с 0), где первой текстовой (буквенной) информацией является `text`.

    Правило:
    - из начала текста страницы игнорируются цифры, пробелы и прочие знаки;
    - как только встречается первый символ-буква, оставшийся текст (с нормализованными пробелами)
      должен начинаться с `text`.
    """

    if not text:
        return None

    if isinstance(input_pdf, bytes):
        input_pdf_file = BytesIO(input_pdf)
    else:
        input_pdf_file = open(input_pdf, "rb")

    try:
        reader = PyPDF2.PdfReader(input_pdf_file)


        for page_idx, page in enumerate(reader.pages):
            page_content: list[str] = extract_text_with_miner_coords(
                input_pdf_file,
                page_numbers=[page_idx],
            )
            page_text = page_content[0]
            # Нормализуем переносы/табуляции, чтобы startswith работал предсказуемо.
            page_text = " ".join(page_text.replace("\t", " ").split())

            # Ищем первый буквенный символ (Unicode: кириллица/латиница и т.п.).
            first_alpha_pos: int | None = None
            for i, ch in enumerate(page_text):
                if ch.isalpha():
                    first_alpha_pos = i
                    break

            if first_alpha_pos is None:
                continue

            if page_text[first_alpha_pos:].startswith(text):
                return page_idx

        return None
    finally:
        if not isinstance(input_pdf, bytes):
            input_pdf_file.close()



if __name__ == "__main__":
    
    # найти номер страницы файла, где заголовком будет text
    
    # text = "АННОТАЦИЯ"
    # print(
    #     find_page_index_by_first_text(
    #         r"C:\Users\maxfi\Desktop\ПМООС\ПМООСы\trim\ОК.02.24 СТ-ООС.pdf",
    #         text=text
    #     )
    # )


    # обрезать исходные pdf
    
    # for f in glob.glob(r"C:\Users\maxfi\Desktop\ПМООС\ПМООСы\*.pdf"):
    #     bytes_ = extract_pages(f, pages_to_keep=list(range(20)))
    #     pth = pathlib.Path(f)
    #     new_name = pth.parent / "trim" /pth.name
    #     with open(new_name, "wb") as new_file:
    #         new_file.write(bytes_)
    
    pass