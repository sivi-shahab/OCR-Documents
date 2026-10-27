from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Repository root (api/app/core/config.py -> parents[3]); the shared .env lives there.
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables prefixed with ``OCR_``."""

    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=REPO_ROOT / ".env", extra="ignore")

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

    # Azure AI Document Intelligence (/v1/ocr/azure); the endpoint is off until both are set.
    azure_di_endpoint: str = ""  # e.g. https://<resource>.cognitiveservices.azure.com
    azure_di_api_key: str = ""
    azure_di_model_id: str = "prebuilt-layout"
    azure_di_api_version: str = "2024-11-30"
    azure_di_timeout_seconds: float = Field(default=600, gt=0)

    # Mistral OCR (/v1/ocr/mistral); the endpoint is off until the API key is set.
    mistral_api_key: str = ""
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_ocr_model: str = "mistral-ocr-latest"
    mistral_timeout_seconds: float = Field(default=600, gt=0)

    # Input limits
    max_upload_mb: int = Field(default=50, gt=0)
    max_pdf_pages: int = Field(default=200, gt=0)
    pdf_dpi: int = Field(default=200, ge=72, le=600)
    page_concurrency: int = Field(default=8, gt=0)

    # Asynchronous jobs (/v1/ocr/jobs) for large PDFs
    job_dir: Path = Path("data/jobs")
    job_max_upload_mb: int = Field(default=500, gt=0)
    job_max_pdf_pages: int = Field(default=2000, gt=0)
    job_max_running: int = Field(default=2, gt=0)  # jobs processed at the same time
    job_page_concurrency: int = Field(default=32, gt=0)  # in-flight page requests per running job
    job_page_attempts: int = Field(default=3, gt=0)
    job_retry_backoff_seconds: float = Field(default=2, ge=0)
    # First attempt uses a tighter budget so generation loops fail fast; truncated pages are
    # retried with max_tokens, adding temperature when the output looks like a loop.
    job_first_pass_max_tokens: int = Field(default=6144, gt=0)
    job_retry_temperature_step: float = Field(default=0.2, ge=0, le=1)
    # Pages still truncated, or empty although the PDF has text, get one more try rendered at
    # this resolution; 0 disables.
    job_escalation_dpi: int = Field(default=300, ge=0, le=600)
    # A completed job whose fallback + failed pages exceed this share of its pages gets
    # needs_review=true. 1/250, as in olmOCR.
    job_max_fallback_ratio: float = Field(default=0.004, ge=0, le=1)

    # Job webhooks are off unless both are set: callback URLs must point at an allowed host.
    webhook_secret: str = ""
    webhook_allowed_hosts: str = ""  # comma-separated host names, e.g. "erp.internal,10.0.0.5"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def job_max_upload_bytes(self) -> int:
        return self.job_max_upload_mb * 1024 * 1024

    @property
    def webhook_hosts(self) -> frozenset[str]:
        return frozenset(
            host.strip().lower() for host in self.webhook_allowed_hosts.split(",") if host.strip()
        )

    @property
    def webhooks_enabled(self) -> bool:
        return bool(self.webhook_secret and self.webhook_hosts)

    @property
    def azure_enabled(self) -> bool:
        return bool(self.azure_di_endpoint and self.azure_di_api_key)

    @property
    def mistral_enabled(self) -> bool:
        return bool(self.mistral_api_key)

    @property
    def max_engine_connections(self) -> int:
        return self.page_concurrency + self.job_max_running * self.job_page_concurrency


@lru_cache
def get_settings() -> Settings:
    return Settings()
