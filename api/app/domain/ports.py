from collections.abc import Sequence
from typing import Protocol

from app.domain.models import PageImage


class OcrEngine(Protocol):
    """Anything that can turn one or more page images into raw model output."""

    async def recognize(self, images: Sequence[PageImage]) -> str: ...

    async def is_healthy(self) -> bool: ...


class DocumentRenderer(Protocol):
    """Converts a multi-page document into page images."""

    def render(self, content: bytes) -> list[PageImage]: ...
