FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential python3-venv \
  && rm -rf /var/lib/apt/lists/*

# создаём виртуальное окружение
RUN python -m venv /opt/venv --system-site-packages
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
  && python -m pip install -r /app/requirements.txt

COPY . /app

CMD ["python", "api.py"]
