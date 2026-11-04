"""Ollama provider: sends the conversation to a local Ollama server (POST /api/chat).

Failure mapping (the message-processing service turns every LLMError into a safe fallback):
    timeout                      -> LLMTimeoutError
    cannot connect / network     -> LLMConnectionError
    non-200 HTTP status          -> LLMProviderError
    invalid JSON / wrong shape   -> LLMProviderError
    missing, empty, truncated or placeholder content -> LLMProviderError

Privacy: nothing here logs message text, replies or response bodies, and exception
messages contain only status codes and fixed phrases.
"""

import logging
import re
import time

import httpx

from app.providers.base import (
    LLMConnectionError,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)
from app.providers.safety import SAFETY_MODEL, SAFETY_PROVIDER, medical_safe_reply

logger = logging.getLogger(__name__)

_ROLE_MAP = {"customer": "user", "assistant": "assistant"}

# Low temperature for steadier, more predictable replies; num_predict caps the length (SMS-sized answers).
GENERATION_OPTIONS = {"temperature": 0.2, "num_predict": 256}

# Template leftovers such as "[Customer Name]" or "{{name}}" must never reach a customer.
_PLACEHOLDER = re.compile(r"\[[A-Za-z][^\]\n]{0,40}\]|\{\{[^}\n]*\}\}")


class OllamaProvider:
    """Provider for a local Ollama server. Safe to share between requests (no state)."""

    name = "ollama"

    def __init__(
        self, base_url: str, model: str, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        base = base_url.strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must start with http:// or https://")
        if not model.strip():
            raise ValueError("OLLAMA_MODEL must not be empty")
        self.base_url = base
        self.model = model.strip()
        self._transport = transport  # tests inject httpx.MockTransport; production uses the default

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return a reply from the model, or raise an LLMError subclass."""
        if request.medical_topic:
            # Safety rule: medical questions never reach the model, so no HTTP request is made.
            logger.info("ollama_bypassed reason=medical_topic")
            return LLMResponse(
                content=medical_safe_reply(request.business_name),
                provider=SAFETY_PROVIDER,
                model=SAFETY_MODEL,
                latency_ms=0.0,
            )

        payload = self._build_payload(request)
        started = time.perf_counter()
        try:
            # trust_env=False: a system HTTP proxy must not intercept calls to a local server.
            with httpx.Client(
                transport=self._transport,
                timeout=httpx.Timeout(request.timeout_seconds),
                trust_env=False,
            ) as client:
                http_response = client.post(self.endpoint, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Ollama request timed out") from exc
        except httpx.TransportError as exc:  # connection refused, DNS failure, dropped connection, ...
            raise LLMConnectionError("Could not reach the Ollama server") from exc
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise LLMProviderError("Ollama request failed") from exc

        if http_response.status_code != 200:
            raise LLMProviderError(f"Ollama returned HTTP {http_response.status_code}")

        content = self._extract_content(http_response)
        latency_ms = (time.perf_counter() - started) * 1000
        logger.debug("ollama_reply model=%s history_messages=%d latency_ms=%.0f", self.model, len(request.messages), latency_ms)
        return LLMResponse(content=content, provider=self.name, model=self.model, latency_ms=latency_ms)

    # ---- helpers ---------------------------------------------------------------------------
    def _build_payload(self, request: LLMRequest) -> dict[str, object]:
        """system prompt -> `system`, customer -> `user`, assistant -> `assistant` (oldest first)."""
        messages: list[dict[str, str]] = [{"role": "system", "content": request.system_prompt}]
        for message in request.messages:
            role = _ROLE_MAP.get(message.role)
            if role is None:
                raise LLMProviderError("Unsupported message role in conversation history")
            messages.append({"role": role, "content": message.content})
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": dict(GENERATION_OPTIONS),
        }

    @staticmethod
    def _extract_content(http_response: httpx.Response) -> str:
        try:
            data = http_response.json()
        except ValueError as exc:  # invalid JSON or invalid UTF-8
            raise LLMProviderError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise LLMProviderError("Ollama returned an unexpected response structure")
        if data.get("error"):
            raise LLMProviderError("Ollama reported an error")
        if data.get("done") is False:
            raise LLMProviderError("Ollama returned an incomplete response")
        if data.get("done_reason") == "length":
            raise LLMProviderError("Ollama reply was cut off before it finished")

        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMProviderError("Ollama response has no message")
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMProviderError("Ollama response has no text content")
        content = content.strip()
        if not content:
            raise LLMProviderError("Ollama returned an empty reply")
        if _PLACEHOLDER.search(content):
            raise LLMProviderError("Ollama reply contains an unfilled placeholder")
        return content
