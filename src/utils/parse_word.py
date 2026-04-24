from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(parent):
    """
    Итерирует по Paragraph и Table в правильном порядке
    """
    for child in parent.element.body:
        if child.tag.endswith("p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("tbl"):
            yield Table(child, parent)


def extract_clean_text_from_docx(path: str) -> str:
    doc = Document(path)

    result = []
    seen = set()

    def add_text(text: str):
        text = text.strip()
        if not text:
            return
        if text in seen:
            return
        seen.add(text)
        result.append(text)

    for block in iter_block_items(doc):
        # --- Paragraph ---
        if isinstance(block, Paragraph):
            style_name = block.style.name.lower() if block.style else ""

            if any(x in style_name for x in ["header", "footer"]):
                continue

            add_text(block.text)

        # --- Table ---
        elif isinstance(block, Table):
            for row in block.rows:
                row_text = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_text.append(cell_text)

                if row_text:
                    combined = " | ".join(row_text)
                    add_text(combined)

    return "\n".join(result)


if __name__ == "__main__":
    print(
        extract_clean_text_from_docx(r"C:\Users\maxfi\Desktop\ПМООС\ГИП\word_test.docx")
    )
