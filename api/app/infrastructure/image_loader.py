import io

from PIL import Image, UnidentifiedImageError

from app.core.errors import InvalidDocumentError
from app.domain.models import PageImage

# Formats the model server accepts as-is; anything else is re-encoded to PNG.
_PASSTHROUGH_MIME_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def load_image(content: bytes) -> PageImage:
    """Validates an uploaded image by decoding it, converting exotic formats to PNG."""
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            mime_type = _PASSTHROUGH_MIME_TYPES.get(image.format or "")
            if mime_type:
                return PageImage(content=content, mime_type=mime_type)
            return PageImage(content=_encode_png(image), mime_type="image/png")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidDocumentError("File is not a readable image") from exc


def _encode_png(image: Image.Image) -> bytes:
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
