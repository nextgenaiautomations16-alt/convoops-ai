"""Provider-independent LLM interface.

The message-processing service only knows about `LLMProvider`, `LLMRequest`,
`LLMResponse` and the error classes below. To add a real provider (Ollama, Groq)
write a class with `name`, `model` and `generate()`, honour `request.timeout_seconds`,
raise the errors below, and register it in `app/providers/factory.py`. The service
does not change.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


class LLMError(Exception):
    """Base class for provider failures. The service turns any of them into a safe fallback."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within `timeout_seconds`."""


class LLMConnectionError(LLMError):
    """The provider could not be reached (network down, service not running)."""


class LLMProviderError(LLMError):
    """The provider answered with an error or unusable output."""


@dataclass(frozen=True)
class ChatMessage:
    """One turn of conversation history. `customer` = inbound, `assistant` = our replies."""

    role: Literal["customer", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    """Everything a provider needs. History is oldest-first and ends with the newest customer message."""

    system_prompt: str
    messages: Sequence[ChatMessage]
    intent: str  # value from the deterministic classifier, e.g. "pricing"
    intent_confidence: float
    business_name: str
    timeout_seconds: float
    medical_topic: bool = False  # set by a deterministic safety check, not by the provider


@dataclass(frozen=True)
class LLMResponse:
    """A provider's answer plus metadata that is stored with the outbound message."""

    content: str
    provider: str
    model: str
    latency_ms: float | None = None


class LLMProvider(Protocol):
    """Anything with these members can generate replies."""

    name: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return a reply, or raise an `LLMError` subclass."""
        ...
