from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

# Делает `import src...` работоспособным при запуске pytest из корня репозитория.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_project_parts_zip(pdf_paths: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdf_path in pdf_paths:
            zf.write(pdf_path, arcname=pdf_path.name)
    return buf.getvalue()
