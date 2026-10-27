import pymupdf

from ocr_eval.classify import PageKind, classify_pdf
from ocr_eval.degrade import degrade_pdf


def _image_bytes(path):
    with pymupdf.open(path) as doc:
        return [doc.extract_image(p.get_images()[0][0])["image"] for p in doc]


def test_degrade_is_image_only_and_deterministic(digital_pdf, tmp_path):
    a, b, c = tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"
    mapping = degrade_pdf(digital_pdf, a, [1, 2])
    degrade_pdf(digital_pdf, b, [1, 2])
    degrade_pdf(digital_pdf, c, [1, 2], seed=1)
    assert mapping == {1: 1, 2: 2}
    assert set(classify_pdf(a).values()) == {PageKind.SCAN}
    assert _image_bytes(a) == _image_bytes(b)
    assert _image_bytes(a) != _image_bytes(c)
