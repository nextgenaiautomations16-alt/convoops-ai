"""Environment-based application settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings are read from environment variables, then from a local `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ConvoOps AI"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: str = "INFO"

    # SQLite for local development. For PostgreSQL use e.g.
    # postgresql+psycopg://user:password@localhost:5432/convoops
    database_url: str = "sqlite:///./convoops.db"
    db_echo: bool = False

    api_v1_prefix: str = "/api/v1"
    default_page_size: int = Field(default=20, ge=1)
    max_page_size: int = Field(default=100, ge=1)

    # Message processing (Phase 2, Step 5)
    max_message_length: int = Field(default=1000, ge=1, le=5000)
    llm_provider: str = "mock"  # "mock" (offline, default) or "ollama"; "groq" is not implemented yet
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    llm_timeout_seconds: float = Field(default=10.0, gt=0)
    history_window: int = Field(default=20, ge=1, le=100)  # most recent messages sent to the provider
    # How long a duplicate request waits for the original request's reply before answering 409.
    idempotency_wait_seconds: float = Field(default=3.0, ge=0)
    # An inbound message with no reply after this long is treated as abandoned and resumed on retry.
    processing_stale_seconds: float = Field(default=60.0, ge=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
