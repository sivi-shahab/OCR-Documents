import pymupdf
import pytest


def _write_text_pages(doc: pymupdf.Document, pages: int = 2) -> None:
    for n in range(pages):
        page = doc.new_page(width=595, height=842)
        for i in range(25):
            page.insert_text((50, 60 + i * 28), f"Row {n}-{i} total assets and liabilities 1.234.567.{i:03d}", fontsize=11)


@pytest.fixture
def digital_pdf(tmp_path):
    path = tmp_path / "digital.pdf"
    doc = pymupdf.open()
    _write_text_pages(doc)
    doc.save(path)
    return path


def _render_pages(src, scale: float = 2.0) -> list[bytes]:
    with pymupdf.open(src) as doc:
        return [p.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes("png") for p in doc]


@pytest.fixture
def scan_pdf(tmp_path, digital_pdf):
    path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    for png in _render_pages(digital_pdf):
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=png)
    doc.save(path)
    return path


@pytest.fixture
def hidden_text_pdf(tmp_path, digital_pdf):
    path = tmp_path / "hidden.pdf"
    doc = pymupdf.open()
    for png in _render_pages(digital_pdf):
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=png)
        for i in range(10):
            page.insert_text((50, 60 + i * 28), f"Hidden recognised line number {i} with enough characters", fontsize=11, render_mode=3)
    doc.save(path)
    return path
