FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements_cpu.txt /app/requirements_cpu.txt

RUN python -m pip install --upgrade pip \
  && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0\
  && python -m pip install -r /app/requirements_cpu.txt

COPY . /app

CMD ["python", "api.py"]
