# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN useradd --create-home --uid 10001 llmlab

COPY requirements.txt .
RUN pip install -r requirements.txt

# Optional: build with real-model support → docker build --build-arg WITH_ML=1 .
ARG WITH_ML=0
COPY requirements-ml.txt .
RUN if [ "$WITH_ML" = "1" ]; then pip install -r requirements-ml.txt; fi

COPY app ./app
COPY frontend ./frontend
COPY migrations ./migrations
COPY alembic.ini ./
COPY data/samples ./data/samples
RUN mkdir -p /app/data/adapters /app/data/uploads && chown -R llmlab:llmlab /app/data

USER llmlab
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
