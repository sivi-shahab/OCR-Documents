from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables prefixed with ``OCR_``."""

    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=".env", extra="ignore")

    # Upstream vLLM server
    vllm_base_url: str = "http://localhost:8100/v1"
    vllm_api_key: str = "EMPTY"
    model_name: str = "Unlimited-OCR"
    request_timeout_seconds: float = Field(default=1200, gt=0)

    # Generation parameters recommended by the Unlimited-OCR recipe
    max_tokens: int = Field(default=8192, gt=0)
    ngram_size: int = Field(default=35, ge=0)
    single_image_window: int = Field(default=128, ge=0)
    multi_image_window: int = Field(default=1024, ge=0)

    # Input limits
    max_upload_mb: int = Field(default=50, gt=0)
    max_pdf_pages: int = Field(default=200, gt=0)
    pdf_dpi: int = Field(default=200, ge=72, le=600)
    page_concurrency: int = Field(default=8, gt=0)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
