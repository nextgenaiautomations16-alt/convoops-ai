"""Shared helpers and provider test doubles for the Step 5 tests."""

import threading
from typing import Any

from fastapi.testclient import TestClient
from httpx import Response

from app.providers.base import LLMRequest, LLMResponse
from app.providers.factory import get_llm_provider

CLIENTS = "/api/v1/clients"
BOOKING = "Hello, I would like to book an appointment."


def make_client(
    api: TestClient,
    slug: str = "brightpath-telehealth",
    status: str = "active",
    name: str | None = None,
) -> dict[str, Any]:
    response = api.post(
        CLIENTS,
        json={
            "business_name": name or f"BrightPath Telehealth {slug}",
            "slug": slug,
            "industry": "healthcare",
            "timezone": "America/New_York",
            "business_description": "Synthetic telehealth demo clinic. No real patient data.",
            "configuration_status": status,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_conversation(api: TestClient, client_id: str, customer_id: str = "cust_1001") -> dict[str, Any]:
    response = api.post(f"{CLIENTS}/{client_id}/conversations", json={"customer_id": customer_id})
    assert response.status_code == 201, response.text
    return response.json()


def messages_url(client_id: str, conversation_id: str) -> str:
    return f"{CLIENTS}/{client_id}/conversations/{conversation_id}/messages"


def post_message(
    api: TestClient,
    client_id: str,
    conversation_id: str,
    content: str = BOOKING,
    key: str | None = None,
) -> Response:
    body: dict[str, Any] = {"content": content}
    if key is not None:
        body["idempotency_key"] = key
    return api.post(messages_url(client_id, conversation_id), json=body)


def use_provider(api: TestClient, provider: Any) -> None:
    """Replace the LLM provider used by the app under test."""
    api.app.dependency_overrides[get_llm_provider] = lambda: provider  # type: ignore[attr-defined]


class RecordingProvider:
    """Thread-safe provider that records every request it receives."""

    name = "recording"
    model = "recording-v1"

    def __init__(self, reply: str = "Recorded synthetic reply.") -> None:
        self.reply = reply
        self.requests: list[LLMRequest] = []
        self._lock = threading.Lock()

    def generate(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            self.requests.append(request)
        return LLMResponse(content=self.reply, provider=self.name, model=self.model)


class RaisingProvider:
    """Provider that always raises the given exception."""

    name = "flaky"
    model = "flaky-v1"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        raise self.exc


class StaticProvider:
    """Provider that returns whatever object it was given (for invalid-output tests)."""

    name = "static"
    model = "static-v1"

    def __init__(self, result: Any) -> None:
        self.result = result

    def generate(self, request: LLMRequest) -> Any:
        return self.result
