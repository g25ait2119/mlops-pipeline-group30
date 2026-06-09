FROM python:3.11-slim

ARG HF_MODEL_NAME=computervisionpro/convnextv2-real-fake
ENV HF_MODEL_NAME=${HF_MODEL_NAME}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libjpeg62-turbo libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch==2.4.0 transformers==4.44.2 huggingface-hub==0.24.6 numpy==1.26.4 Pillow==10.4.0 requests==2.32.3

COPY src ./src
COPY id2label.json ./id2label.json

RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "src/inference.py"]
