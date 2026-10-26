import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.domain.ports import OcrEngine
from app.infrastructure.azure_document_intelligence import AzureConfig, AzureDocumentIntelligence
from app.infrastructure.image_loader import load_image
from app.infrastructure.job_store import FileJobStore
from app.infrastructure.pdf_renderer import PyMuPdfRenderer
from app.infrastructure.vllm_engine import GenerationConfig, VllmOcrEngine
from app.infrastructure.webhook import WebhookNotifier
from app.services.job_service import JobService
from app.services.ocr_service import OcrService


def build_engine(settings: Settings, client: httpx.AsyncClient) -> OcrEngine:
    return VllmOcrEngine(
        client,
        GenerationConfig(
            model_name=settings.model_name,
            max_tokens=settings.max_tokens,
            ngram_size=settings.ngram_size,
            single_image_window=settings.single_image_window,
            multi_image_window=settings.multi_image_window,
        ),
    )


def build_azure_ocr(settings: Settings, client: httpx.AsyncClient) -> AzureDocumentIntelligence | None:
    if not settings.azure_enabled:
        return None
    return AzureDocumentIntelligence(
        client,
        AzureConfig(
            endpoint=settings.azure_di_endpoint,
            api_key=settings.azure_di_api_key,
            model_id=settings.azure_di_model_id,
            api_version=settings.azure_di_api_version,
            timeout_seconds=settings.azure_di_timeout_seconds,
        ),
    )


def build_ocr_service(settings: Settings, engine: OcrEngine) -> OcrService:
    """Composition root: wires concrete adapters into the service."""
    return OcrService(
        engine=engine,
        pdf_renderer=PyMuPdfRenderer(dpi=settings.pdf_dpi, max_pages=settings.max_pdf_pages),
        image_loader=load_image,
        page_concurrency=settings.page_concurrency,
    )


def build_job_service(
    settings: Settings, engine: OcrEngine, webhook_client: httpx.AsyncClient | None = None
) -> JobService:
    return JobService(
        engine=engine,
        reader=PyMuPdfRenderer(dpi=settings.pdf_dpi, max_pages=settings.job_max_pdf_pages),
        store=FileJobStore(settings.job_dir),
        max_pages=settings.job_max_pdf_pages,
        max_running_jobs=settings.job_max_running,
        page_concurrency=settings.job_page_concurrency,
        max_attempts=settings.job_page_attempts,
        retry_backoff_seconds=settings.job_retry_backoff_seconds,
        first_pass_max_tokens=settings.job_first_pass_max_tokens,
        retry_max_tokens=settings.max_tokens,
        retry_temperature_step=settings.job_retry_temperature_step,
        escalation_dpi=settings.job_escalation_dpi,
        max_fallback_ratio=settings.job_max_fallback_ratio,
        notifier=WebhookNotifier(webhook_client, settings.webhook_secret)
        if webhook_client and settings.webhooks_enabled
        else None,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with (
            httpx.AsyncClient(
                base_url=settings.vllm_base_url,
                headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
                timeout=httpx.Timeout(settings.request_timeout_seconds, connect=10),
                limits=httpx.Limits(max_connections=settings.max_engine_connections),
            ) as client,
            httpx.AsyncClient(timeout=10, follow_redirects=False) as webhook_client,
            httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as azure_client,
        ):
            engine = build_engine(settings, client)
            app.state.azure_ocr = build_azure_ocr(settings, azure_client)
            app.state.ocr_service = build_ocr_service(settings, engine)
            app.state.job_service = build_job_service(settings, engine, webhook_client)
            await app.state.job_service.start()
            try:
                yield
            finally:
                await app.state.job_service.stop()

    app = FastAPI(
        title="Unlimited-OCR API",
        version="1.0.0",
        description="Document parsing for images and PDFs, backed by baidu/Unlimited-OCR on vLLM.",
        lifespan=lifespan,
    )
    app.include_router(router)
    register_error_handlers(app)
    return app


logging.basicConfig(level=logging.INFO)
app = create_app()
