from pydantic import BaseModel, Field

from app.domain.models import Block, OcrResult


class BlockResponse(BaseModel):
    category: str = Field(examples=["title"])
    bbox: list[int] | None = Field(
        description="[x1, y1, x2, y2] normalised to a 0-999 grid of the page", examples=[[117, 74, 581, 101]]
    )
    text: str

    @classmethod
    def from_domain(cls, block: Block) -> "BlockResponse":
        return cls(category=block.category, bbox=list(block.bbox) if block.bbox else None, text=block.text)


class PageResponse(BaseModel):
    page: int | None = Field(
        description="1-based page number; null when the result covers the whole document"
    )
    markdown: str
    blocks: list[BlockResponse]
    raw: str | None = Field(
        default=None, description="Unprocessed model output, included when include_raw=true"
    )

    @classmethod
    def from_domain(cls, result: OcrResult, include_raw: bool) -> "PageResponse":
        return cls(
            page=result.page,
            markdown=result.markdown,
            blocks=[BlockResponse.from_domain(block) for block in result.blocks],
            raw=result.raw if include_raw else None,
        )


class OcrResponse(BaseModel):
    filename: str | None
    page_count: int
    markdown: str = Field(description="Markdown of all pages joined together")
    pages: list[PageResponse]

    @classmethod
    def from_domain(cls, filename: str | None, results: list[OcrResult], include_raw: bool) -> "OcrResponse":
        return cls(
            filename=filename,
            page_count=len(results),
            markdown="\n\n".join(result.markdown for result in results if result.markdown),
            pages=[PageResponse.from_domain(result, include_raw) for result in results],
        )


class HealthResponse(BaseModel):
    status: str
    engine: str


class ErrorResponse(BaseModel):
    detail: str
