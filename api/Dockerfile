FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OCR_JOB_DIR=/data/jobs

WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27" \
        "pydantic-settings>=2.4" "python-multipart>=0.0.9" "pymupdf>=1.24" "pillow>=10.4"

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/jobs && chown appuser /data/jobs
USER appuser
VOLUME /data/jobs

EXPOSE 8000
# Background jobs live in this process: keep a single uvicorn worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
