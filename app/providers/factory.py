"""Chooses the LLM provider from configuration (LLM_PROVIDER). Credentials, when real
providers need them, must come from environment variables, never from source code.

Providers are cheap to build and hold no state, so they are created on every call. There is
deliberately no cache here: a cached provider would keep using stale settings."""

from app.core.config import Settings, get_settings
from app.providers.base import LLMProvider
from app.providers.mock import MockLLMProvider
from app.providers.ollama import OllamaProvider


class UnsupportedProviderError(ValueError):
    """LLM_PROVIDER names a provider that does not exist, is not implemented, or is misconfigured."""


def _create(name: str, settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    if name == "mock":
        return MockLLMProvider()
    if name == "ollama":
        try:
            return OllamaProvider(settings.ollama_base_url, settings.ollama_model)
        except ValueError as exc:
            raise UnsupportedProviderError(f"Invalid Ollama configuration: {exc}") from exc
    if name == "groq":
        raise UnsupportedProviderError("LLM provider 'groq' is planned but not implemented yet.")
    raise UnsupportedProviderError(f"Unknown LLM provider '{name}'.")


def get_llm_provider() -> LLMProvider:
    """FastAPI dependency. Tests replace it with `app.dependency_overrides`."""
    settings = get_settings()
    return _create(settings.llm_provider.strip().lower(), settings)
