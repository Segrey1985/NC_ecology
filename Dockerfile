FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_NO_DEV=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential python3-venv unar \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv venv /opt/venv --system-site-packages
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-package torch

COPY . /app

CMD ["python", "-m", "api.api_prod"]
