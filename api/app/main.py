import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.infrastructure.image_loader import load_image
from app.infrastructure.pdf_renderer import PyMuPdfRenderer
from app.infrastructure.vllm_engine import GenerationConfig, VllmOcrEngine
from app.services.ocr_service import OcrService


def build_ocr_service(settings: Settings, client: httpx.AsyncClient) -> OcrService:
    """Composition root: wires concrete adapters into the service."""
    engine = VllmOcrEngine(
        client,
        GenerationConfig(
            model_name=settings.model_name,
            max_tokens=settings.max_tokens,
            ngram_size=settings.ngram_size,
            single_image_window=settings.single_image_window,
            multi_image_window=settings.multi_image_window,
        ),
    )
    return OcrService(
        engine=engine,
        pdf_renderer=PyMuPdfRenderer(dpi=settings.pdf_dpi, max_pages=settings.max_pdf_pages),
        image_loader=load_image,
        page_concurrency=settings.page_concurrency,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            base_url=settings.vllm_base_url,
            headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
            timeout=httpx.Timeout(settings.request_timeout_seconds, connect=10),
            limits=httpx.Limits(max_connections=settings.page_concurrency * 2),
        ) as client:
            app.state.ocr_service = build_ocr_service(settings, client)
            yield

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
