import io
from collections.abc import Sequence

import pymupdf
from PIL import Image

from app.domain.models import PageImage

RAW_PAGE = (
    "<|det|>title [117, 74, 581, 101]<|/det|>Laporan Keuangan\n"
    "<|det|>text [117, 130, 592, 148]<|/det|>Baris pertama\n"
    "baris kedua"
)


class FakeEngine:
    def __init__(self, output: str = RAW_PAGE, healthy: bool = True) -> None:
        self.output = output
        self.healthy = healthy
        self.calls: list[int] = []  # number of images per call

    async def recognize(self, images: Sequence[PageImage]) -> str:
        self.calls.append(len(images))
        return self.output

    async def is_healthy(self) -> bool:
        return self.healthy


def make_png(fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format=fmt)
    return buffer.getvalue()


def make_pdf(pages: int) -> bytes:
    document = pymupdf.open()
    for number in range(pages):
        document.new_page().insert_text((72, 72), f"Page {number + 1}")
    return document.tobytes()
