import io
from collections.abc import Callable, Sequence

import pymupdf
from PIL import Image

from app.domain.models import EngineOutput, PageImage

RAW_PAGE = (
    "<|det|>title [117, 74, 581, 101]<|/det|>Laporan Keuangan\n"
    "<|det|>text [117, 130, 592, 148]<|/det|>Baris pertama\n"
    "baris kedua"
)


# Decides each response from the call number (1-based); may raise to simulate engine errors.
Script = Callable[[int], EngineOutput]


class FakeEngine:
    def __init__(self, output: str = RAW_PAGE, healthy: bool = True, script: Script | None = None) -> None:
        self.output = output
        self.healthy = healthy
        self.script = script
        self.calls: list[int] = []  # number of images per call
        self.params: list[tuple[int | None, float]] = []  # (max_tokens, temperature) per call
        self.image_sizes: list[int] = []  # bytes of the first image per call

    async def recognize(
        self, images: Sequence[PageImage], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> EngineOutput:
        self.calls.append(len(images))
        self.params.append((max_tokens, temperature))
        self.image_sizes.append(len(images[0].content))
        if self.script:
            return self.script(len(self.calls))
        return EngineOutput(text=self.output)

    async def is_healthy(self) -> bool:
        return self.healthy


def make_png(fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format=fmt)
    return buffer.getvalue()


def make_pdf(pages: int, text_layer: bool = True, text: str | None = None) -> bytes:
    """Each page's text layer is `text` when given, else "Page N"."""
    document = pymupdf.open()
    for number in range(pages):
        page = document.new_page()
        if text_layer and text:
            page.insert_textbox(pymupdf.Rect(72, 72, 540, 770), text, fontsize=8)
        elif text_layer:
            page.insert_text((72, 72), f"Page {number + 1}")
        else:
            page.draw_rect(pymupdf.Rect(72, 72, 144, 144), fill=(0, 0, 0))
    return document.tobytes()
