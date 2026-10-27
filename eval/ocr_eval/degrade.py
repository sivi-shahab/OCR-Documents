"""Synthetic scans: render pages and degrade them to image-only PDF pages."""
import io
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

DPI = 200
ROTATION_DEG = 1.0
BLUR_RADIUS = 0.6
NOISE_SIGMA = 6.0
JPEG_QUALITY = 60


def _degrade(image: Image.Image, rng: np.random.Generator) -> bytes:
    gray = image.convert("L")
    angle = float(rng.uniform(-ROTATION_DEG, ROTATION_DEG))
    gray = gray.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    gray = gray.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    noisy = np.asarray(gray, dtype=np.float64) + rng.normal(0, NOISE_SIGMA, (gray.height, gray.width))
    out = io.BytesIO()
    Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "L").save(out, "JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def degrade_pdf(src: Path, dst: Path, pages: list[int], seed: int = 20261027) -> dict[int, int]:
    """Write the chosen 1-based pages of src as image-only scans; return {dst page: src page}."""
    mapping: dict[int, int] = {}
    with pymupdf.open(src) as doc, pymupdf.open() as out:
        for n, src_page in enumerate(pages, start=1):
            page = doc[src_page - 1]
            pix = page.get_pixmap(dpi=DPI)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            jpeg = _degrade(image, np.random.default_rng(seed + src_page))
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=jpeg)
            mapping[n] = src_page
        out.save(dst)
    return mapping
