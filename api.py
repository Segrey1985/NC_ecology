import io
import json
import tempfile
import zipfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from main import main as run_main


app = FastAPI(title="NC_ecology API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    placeholders: UploadFile = File(..., description="JSON с плейсхолдерами"),
    template_docx: UploadFile = File(..., description="DOCX шаблон"),
    table_placeholders: UploadFile | None = File(
        None, description="JSON с табличными плейсхолдерами (опционально)"
    ),
    collection_name: str = Form("main"),
):
    
    # проверка типов приложенных файлов
    
    if placeholders.content_type not in {"application/json", "text/json"}:
        raise HTTPException(status_code=400, detail="`placeholders` должен быть JSON")

    if template_docx and template_docx.content_type not in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    }:
        raise HTTPException(status_code=400, detail="`template_docx` должен быть DOCX")

    if table_placeholders and table_placeholders.content_type not in {
        "application/json",
        "text/json",
    }:
        raise HTTPException(
            status_code=400, detail="`table_placeholders` должен быть JSON"
        )

    with tempfile.TemporaryDirectory(prefix="nc_ecology_") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "in"
        output_dir = tmp_dir / "out"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        placeholders_path = input_dir / "placeholders.json"
        placeholders_bytes = await placeholders.read()
        try:
            json.loads(placeholders_bytes.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Некорректный JSON: {e}") from e
        placeholders_path.write_bytes(placeholders_bytes)

        table_placeholders_path: Path | None = None
        if table_placeholders:
            table_placeholders_path = input_dir / "table_placeholders.json"
            table_bytes = await table_placeholders.read()
            try:
                json.loads(table_bytes.decode("utf-8"))
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Некорректный JSON: {e}"
                ) from e
            table_placeholders_path.write_bytes(table_bytes)

        template_docx_path: Path | None = None
        if template_docx:
            template_docx_path = input_dir / "template.docx"
            template_docx_path.write_bytes(await template_docx.read())

        run_main(
            template_docx_path=template_docx_path,
            placeholders_path=placeholders_path,
            table_placeholders_path=table_placeholders_path,
            project_parts_path=None,
            output_path=output_dir,
            collection_name=collection_name,
            verbose=False,
            test_mode=False,
        )
        
        zip_buf = io.BytesIO()
        
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    # сохраняем относительный путь внутри архива
                    arc_name = file_path.relative_to(output_dir)
                    zf.write(file_path, arcname=arc_name)
        
        zip_buf.seek(0)
        
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=result.zip"},
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0" ,port=8000)