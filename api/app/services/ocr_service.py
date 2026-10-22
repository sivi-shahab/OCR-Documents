import asyncio
from collections.abc import Callable

from app.domain.models import OcrResult, PageImage, PdfStrategy
from app.domain.parser import parse_blocks, to_markdown
from app.domain.ports import DocumentRenderer, OcrEngine

ImageLoader = Callable[[bytes], PageImage]


class OcrService:
    """Use cases for recognising images and PDF documents."""

    def __init__(
        self,
        engine: OcrEngine,
        pdf_renderer: DocumentRenderer,
        image_loader: ImageLoader,
        page_concurrency: int,
    ) -> None:
        self._engine = engine
        self._pdf_renderer = pdf_renderer
        self._image_loader = image_loader
        self._page_limiter = asyncio.Semaphore(page_concurrency)

    async def recognize_image(self, content: bytes) -> OcrResult:
        image = self._image_loader(content)
        return await self._recognize([image], page=1)

    async def recognize_pdf(self, content: bytes, strategy: PdfStrategy) -> list[OcrResult]:
        pages = await asyncio.to_thread(self._pdf_renderer.render, content)
        if strategy is PdfStrategy.WHOLE_DOCUMENT:
            return [await self._recognize(pages, page=None)]
        return list(
            await asyncio.gather(
                *(self._recognize([image], page=number) for number, image in enumerate(pages, start=1))
            )
        )

    async def is_engine_healthy(self) -> bool:
        return await self._engine.is_healthy()

    async def _recognize(self, images: list[PageImage], page: int | None) -> OcrResult:
        async with self._page_limiter:
            raw = await self._engine.recognize(images)
        blocks = parse_blocks(raw)
        return OcrResult(page=page, markdown=to_markdown(blocks), blocks=blocks, raw=raw)
