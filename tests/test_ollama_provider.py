"""Ollama provider tests. All HTTP is mocked with httpx.MockTransport.

No running Ollama server and no internet access are needed. An autouse fixture makes any
attempt to use a REAL httpx transport fail loudly, so a test can never silently hit the network.
"""

import json
import logging
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.models import Client
from app.providers import factory
from app.providers.base import (
    ChatMessage,
    LLMConnectionError,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)
from app.providers.factory import UnsupportedProviderError, get_llm_provider
from app.providers.mock import MockLLMProvider
from app.providers.ollama import GENERATION_OPTIONS, OllamaProvider
from app.providers.safety import SAFETY_MODEL, SAFETY_PROVIDER
from app.services.message_processing_service import build_system_prompt
from tests.helpers import make_client, make_conversation, post_message, use_provider

BASE = "http://ollama.test:11434"
SYSTEM = "SYSTEM PROMPT FOR THE TEST"


@pytest.fixture(autouse=True)
def forbid_real_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(self: Any, request: Any) -> Any:
        raise AssertionError("a real HTTP request was attempted during a test")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)


# ---- helpers --------------------------------------------------------------------------
def ok_body(content: Any = "Thanks for reaching out. Clinic staff will follow up.", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "qwen2.5:7b",
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
    }
    body.update(extra)
    return body


class Calls:
    """Records the requests an OllamaProvider sent."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    @property
    def count(self) -> int:
        return len(self.requests)

    def json(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def provider_for(
    handler: Callable[[httpx.Request], httpx.Response], calls: Calls | None = None, model: str = "qwen2.5:7b",
    base_url: str = BASE,
) -> OllamaProvider:  # fmt: skip
    def recording(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.requests.append(request)
        return handler(request)

    return OllamaProvider(base_url, model, transport=httpx.MockTransport(recording))


def responding(body: Any = None, status: int = 200, content: bytes | None = None) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if content is not None:
            return httpx.Response(status, content=content)
        return httpx.Response(status, json=ok_body() if body is None else body)

    return handler


def raising(exc: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def make_request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "system_prompt": SYSTEM,
        "messages": (ChatMessage(role="customer", content="Hello, I would like to book an appointment."),),
        "intent": "appointment_booking",
        "intent_confidence": 0.9,
        "business_name": "BrightPath Telehealth",
        "timeout_seconds": 7.5,
        "medical_topic": False,
    }
    values.update(overrides)
    return LLMRequest(**values)


# ============================ successful request =====================================
def test_successful_response_is_returned_as_an_llm_response() -> None:
    provider = provider_for(responding(ok_body("  Thanks! Clinic staff will follow up.  \n")))
    response = provider.generate(make_request())
    assert isinstance(response, LLMResponse)
    assert response.content == "Thanks! Clinic staff will follow up."  # trimmed
    assert (response.provider, response.model) == ("ollama", "qwen2.5:7b")
    assert response.latency_ms is not None and response.latency_ms >= 0


def test_request_goes_to_api_chat_with_the_expected_json_body() -> None:
    calls = Calls()
    provider_for(responding(), calls).generate(make_request())
    assert calls.count == 1
    sent = calls.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "http://ollama.test:11434/api/chat"
    assert sent.headers["content-type"] == "application/json"
    assert "authorization" not in sent.headers  # no secrets are sent
    body = calls.json()
    assert set(body) == {"model", "messages", "stream", "options"}
    assert body["model"] == "qwen2.5:7b"
    assert body["stream"] is False
    assert body["options"] == GENERATION_OPTIONS == {"temperature": 0.2, "num_predict": 256}


@pytest.mark.parametrize("base", ["http://localhost:11434", "http://localhost:11434/", "  http://localhost:11434//  "])
def test_base_url_is_normalised(base: str) -> None:
    calls = Calls()
    provider_for(responding(), calls, base_url=base).generate(make_request())
    assert str(calls.requests[0].url) == "http://localhost:11434/api/chat"


@pytest.mark.parametrize("model", ["qwen2.5:7b", "llama3.2:3b", "gemma3:latest", "tinyllama:latest"])
def test_the_configured_model_is_sent_and_recorded(model: str) -> None:
    calls = Calls()
    provider = provider_for(responding(ok_body(model=model)), calls, model=model)
    response = provider.generate(make_request())
    assert calls.json()["model"] == model
    assert (provider.model, response.model, provider.name) == (model, model, "ollama")


def test_the_timeout_from_the_request_is_applied() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.extensions["timeout"])
        return httpx.Response(200, json=ok_body())

    provider_for(handler).generate(make_request(timeout_seconds=7.5))
    assert seen == {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}


# ============================ messages ================================================
def test_system_prompt_is_the_first_message_and_roles_are_mapped() -> None:
    calls = Calls()
    history = (
        ChatMessage(role="customer", content="first customer message"),
        ChatMessage(role="assistant", content="our first reply"),
        ChatMessage(role="customer", content="second customer message"),
    )
    provider_for(responding(), calls).generate(make_request(messages=history))
    assert calls.json()["messages"] == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "first customer message"},
        {"role": "assistant", "content": "our first reply"},
        {"role": "user", "content": "second customer message"},
    ]  # order preserved, roles mapped: customer->user, assistant->assistant


def test_unknown_history_role_is_rejected_without_calling_ollama() -> None:
    calls = Calls()
    bad = (ChatMessage(role="robot", content="x"),)  # type: ignore[arg-type]
    with pytest.raises(LLMProviderError):
        provider_for(responding(), calls).generate(make_request(messages=bad))
    assert calls.count == 0


def test_the_real_system_prompt_is_sent_unchanged() -> None:
    client = Client(
        business_name="BrightPath Telehealth", slug="b", industry="healthcare", timezone="UTC",
        business_description="Synthetic demo clinic.",
    )  # fmt: skip
    prompt = build_system_prompt(client)
    calls = Calls()
    provider_for(responding(), calls).generate(make_request(system_prompt=prompt))
    assert calls.json()["messages"][0] == {"role": "system", "content": prompt}


# ============================ system prompt rules ======================================
def test_system_prompt_keeps_the_original_rules_and_adds_the_new_ones() -> None:
    client = Client(
        business_name="BrightPath Telehealth", slug="b", industry="healthcare", timezone="UTC",
        business_description="Synthetic demo clinic.",
    )  # fmt: skip
    prompt = build_system_prompt(client)
    for original in (
        "Reply briefly, in plain text suitable for SMS",
        "Never give medical advice, diagnoses or treatment suggestions",
        "local emergency number",
        "Never invent prices, appointment availability, policies or patient records",
        "staff member has been notified",
        "Use only the information in this conversation",
    ):
        assert original in prompt
    for added in (
        "ready to send to the customer",
        "[Customer Name]",
        "no markdown",
        "Never invent business policies",
        "clinic staff will follow up",
        "unless the conversation shows it",
    ):
        assert added in prompt
    assert "BrightPath Telehealth" in prompt and "Synthetic demo clinic." in prompt


# ============================ failures ================================================
@pytest.mark.parametrize(
    "exc",
    [httpx.ConnectTimeout("t"), httpx.ReadTimeout("t"), httpx.WriteTimeout("t"), httpx.PoolTimeout("t")],
)
def test_timeouts_map_to_llm_timeout_error(exc: Exception) -> None:
    with pytest.raises(LLMTimeoutError):
        provider_for(raising(exc)).generate(make_request())


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadError("reset"),
        httpx.WriteError("broken pipe"),
        httpx.RemoteProtocolError("server disconnected"),
        httpx.UnsupportedProtocol("bad scheme"),
    ],
)
def test_connection_failures_map_to_llm_connection_error(exc: Exception) -> None:
    with pytest.raises(LLMConnectionError):
        provider_for(raising(exc)).generate(make_request())


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 502, 503])
def test_http_errors_map_to_llm_provider_error_without_leaking_the_body(status: int) -> None:
    body = {"error": "secret-detail: model 'qwen2.5:7b' not found"}
    with pytest.raises(LLMProviderError) as info:
        provider_for(responding(body, status=status)).generate(make_request())
    assert str(info.value) == f"Ollama returned HTTP {status}"
    assert "secret-detail" not in str(info.value)


@pytest.mark.parametrize("content", [b"not json at all", b"", b"{broken", b"\xff\xfe\x00 not utf8"])
def test_invalid_json_maps_to_llm_provider_error(content: bytes) -> None:
    with pytest.raises(LLMProviderError, match="invalid JSON"):
        provider_for(responding(content=content)).generate(make_request())


@pytest.mark.parametrize("body", [[], ["a"], "text", None, 5, True])
def test_json_that_is_not_an_object_is_rejected(body: Any) -> None:
    with pytest.raises(LLMProviderError, match="unexpected response structure"):
        provider_for(responding(content=json.dumps(body).encode())).generate(make_request())


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"done": True},
        {"message": None},
        {"message": "just text"},
        {"message": ["x"]},
        {"message": {}},
        {"message": {"role": "assistant"}},
        {"message": {"role": "assistant", "content": None}},
        {"message": {"role": "assistant", "content": 123}},
        {"message": {"role": "assistant", "content": ["a"]}},
        {"response": "wrong field (that is /api/generate)"},
    ],
)
def test_missing_or_malformed_content_maps_to_llm_provider_error(body: dict[str, Any]) -> None:
    with pytest.raises(LLMProviderError):
        provider_for(responding(body)).generate(make_request())


@pytest.mark.parametrize("content", ["", "   ", "\n\t \r\n"])
def test_empty_replies_are_never_returned(content: str) -> None:
    with pytest.raises(LLMProviderError, match="empty reply"):
        provider_for(responding(ok_body(content))).generate(make_request())


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"error": "model failed to load"}, "reported an error"),
        ({"done": False}, "incomplete"),
        ({"done_reason": "length"}, "cut off"),
    ],
)
def test_error_incomplete_and_truncated_responses_are_rejected(extra: dict[str, Any], message: str) -> None:
    with pytest.raises(LLMProviderError, match=message):
        provider_for(responding(ok_body(**extra))).generate(make_request())


@pytest.mark.parametrize(
    "content",
    [
        "Hi [Customer Name], thanks for contacting us.",
        "Call us at [Clinic Phone Number].",
        "Hello {{name}}, we will follow up.",
        "Your appointment is on [date].",
    ],
)
def test_replies_with_unfilled_placeholders_are_rejected(content: str) -> None:
    with pytest.raises(LLMProviderError, match="placeholder"):
        provider_for(responding(ok_body(content))).generate(make_request())


@pytest.mark.parametrize("content", ["Staff will follow up (1).", "Options: 1) call us 2) wait.", "Call us {anytime}."])
def test_ordinary_punctuation_is_not_mistaken_for_a_placeholder(content: str) -> None:
    assert provider_for(responding(ok_body(content))).generate(make_request()).content == content


def test_invalid_configuration_is_rejected_at_construction() -> None:
    for bad_url in ("localhost:11434", "ftp://x", ""):
        with pytest.raises(ValueError):
            OllamaProvider(bad_url, "qwen2.5:7b")
    with pytest.raises(ValueError):
        OllamaProvider(BASE, "   ")


# ============================ medical safety ===========================================
def test_medical_request_never_reaches_ollama() -> None:
    calls = Calls()
    provider = provider_for(responding(), calls)
    response = provider.generate(make_request(medical_topic=True, intent="general_question"))

    assert calls.count == 0  # NO HTTP request was made
    assert (response.provider, response.model) == (SAFETY_PROVIDER, SAFETY_MODEL) == ("safety_guard", "deterministic-medical-v1")
    text = response.content.lower()
    assert "not able to give medical advice" in text
    assert "qualified clinician" in text and "brightpath telehealth" in text
    assert "emergency number" in text  # emergencies are directed to local emergency services
    for forbidden in ("you should take", "i recommend", "your diagnosis", "dosage", "your records"):
        assert forbidden not in text


def test_medical_request_is_answered_even_when_ollama_is_down() -> None:
    calls = Calls()
    provider = provider_for(raising(httpx.ConnectError("refused")), calls)
    response = provider.generate(make_request(medical_topic=True))  # must not raise
    assert response.content and calls.count == 0


def test_non_medical_requests_still_use_ollama() -> None:
    calls = Calls()
    provider_for(responding(), calls).generate(make_request(medical_topic=False))
    assert calls.count == 1


# ============================ factory and configuration ==================================
def test_mock_remains_the_default_provider() -> None:
    assert Settings().llm_provider == "mock"
    assert isinstance(get_llm_provider(), MockLLMProvider)


def test_default_ollama_settings() -> None:
    settings = Settings()
    assert (settings.ollama_base_url, settings.ollama_model) == ("http://localhost:11434", "qwen2.5:7b")


def test_factory_builds_an_ollama_provider_from_settings() -> None:
    provider = factory._create("ollama", Settings(llm_provider="ollama", ollama_base_url="http://h:1/", ollama_model="llama3.2:3b"))
    assert isinstance(provider, OllamaProvider)
    assert (provider.base_url, provider.model, provider.endpoint) == ("http://h:1", "llama3.2:3b", "http://h:1/api/chat")


def test_factory_follows_the_environment_and_never_serves_stale_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def current() -> Any:
        get_settings.cache_clear()
        return get_llm_provider()

    try:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
        first = current()
        assert isinstance(first, OllamaProvider) and first.model == "qwen2.5:7b"

        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")  # same provider name, new model
        second = current()
        assert isinstance(second, OllamaProvider) and second.model == "llama3.2:3b"

        monkeypatch.setenv("LLM_PROVIDER", "MOCK")  # switching back
        assert isinstance(current(), MockLLMProvider)
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_factory_wraps_bad_ollama_configuration_in_unsupported_provider_error() -> None:
    with pytest.raises(UnsupportedProviderError, match="Invalid Ollama configuration"):
        factory._create("ollama", Settings(ollama_base_url="not-a-url"))


def test_groq_is_still_not_implemented() -> None:
    with pytest.raises(UnsupportedProviderError, match="not implemented"):
        factory._create("groq")


def test_both_providers_satisfy_the_provider_protocol() -> None:
    for provider in (MockLLMProvider(), OllamaProvider(BASE, "qwen2.5:7b")):
        assert isinstance(provider.name, str) and isinstance(provider.model, str)
        assert callable(provider.generate)


def test_mock_and_ollama_are_interchangeable_for_the_same_request() -> None:
    request = make_request()
    for provider in (MockLLMProvider(), provider_for(responding())):
        response = provider.generate(request)
        assert isinstance(response, LLMResponse) and response.content.strip()


# ============================ through the real message pipeline ==========================
@pytest.fixture()
def ctx(app_client: TestClient) -> SimpleNamespace:
    client = make_client(app_client)
    conversation = make_conversation(app_client, client["id"])
    return SimpleNamespace(api=app_client, cid=client["id"], vid=conversation["id"])


def test_pipeline_uses_ollama_and_records_provider_and_model(ctx: SimpleNamespace) -> None:
    calls = Calls()
    use_provider(ctx.api, provider_for(responding(ok_body("Happy to help. Clinic staff will confirm a time.")), calls))
    body = post_message(ctx.api, ctx.cid, ctx.vid, "Hello, I would like to book an appointment.").json()

    assert body["reply_type"] == "generated" and body["processing_status"] == "processed"
    assert (body["provider"], body["model"]) == ("ollama", "qwen2.5:7b")
    assert body["outbound_message"]["content"] == "Happy to help. Clinic staff will confirm a time."
    sent = calls.json()["messages"]
    assert sent[0]["role"] == "system" and "BrightPath Telehealth" in sent[0]["content"]
    assert sent[-1] == {"role": "user", "content": "Hello, I would like to book an appointment."}


def test_pipeline_sends_conversation_history_to_ollama(ctx: SimpleNamespace) -> None:
    calls = Calls()
    use_provider(ctx.api, provider_for(responding(ok_body("Staff will follow up.")), calls))
    post_message(ctx.api, ctx.cid, ctx.vid, "first question about booking")
    post_message(ctx.api, ctx.cid, ctx.vid, "second question about pricing")
    roles = [(m["role"], m["content"]) for m in calls.json()["messages"][1:]]
    assert roles == [
        ("user", "first question about booking"),
        ("assistant", "Staff will follow up."),
        ("user", "second question about pricing"),
    ]


@pytest.mark.parametrize(
    "handler, category",
    [
        (raising(httpx.ReadTimeout("t")), "provider_timeout"),
        (raising(httpx.ConnectError("refused")), "provider_connection"),
        (responding({"error": "boom"}, status=500), "provider_error"),
        (responding(content=b"not json"), "provider_error"),
        (responding({"message": {"role": "assistant"}}), "provider_error"),
        (responding(ok_body("   ")), "provider_error"),
        (responding(ok_body("Hi [Customer Name]!")), "provider_error"),
    ],
)
def test_pipeline_falls_back_safely_when_ollama_fails(
    ctx: SimpleNamespace, handler: Callable[[httpx.Request], httpx.Response], category: str
) -> None:
    use_provider(ctx.api, provider_for(handler))
    response = post_message(ctx.api, ctx.cid, ctx.vid, "I would like to book an appointment", key="k-ollama")
    assert response.status_code == 201
    body = response.json()
    assert (body["reply_type"], body["processing_status"], body["failure_category"]) == ("fallback", "failed", category)
    assert (body["provider"], body["model"]) == ("fallback", "deterministic-fallback-v1")
    assert body["inbound_message"]["content"] == "I would like to book an appointment"  # never lost
    assert body["inbound_message"]["provider"] == "ollama"  # which provider was attempted
    assert body["conversation_status"] == "open"
    assert "[" not in body["outbound_message"]["content"] and "boom" not in response.text


def test_pipeline_medical_message_makes_no_ollama_request(ctx: SimpleNamespace) -> None:
    calls = Calls()
    use_provider(ctx.api, provider_for(responding(), calls))
    body = post_message(ctx.api, ctx.cid, ctx.vid, "I have chest pain, what medication should I take?").json()

    assert calls.count == 0  # proves no Ollama HTTP request was made for the medical message
    assert (body["provider"], body["model"], body["reply_type"]) == ("safety_guard", "deterministic-medical-v1", "generated")
    reply = body["outbound_message"]["content"].lower()
    assert "not able to give medical advice" in reply and "emergency number" in reply

    post_message(ctx.api, ctx.cid, ctx.vid, "How much does a visit cost?")  # a normal message afterwards
    assert calls.count == 1


def test_pipeline_idempotent_replay_does_not_call_ollama_again(ctx: SimpleNamespace) -> None:
    calls = Calls()
    use_provider(ctx.api, provider_for(responding(), calls))
    first = post_message(ctx.api, ctx.cid, ctx.vid, "book an appointment", key="same")
    second = post_message(ctx.api, ctx.cid, ctx.vid, "book an appointment", key="same")
    assert (first.status_code, second.status_code) == (201, 200) and calls.count == 1


def test_ollama_traffic_content_is_never_logged(ctx: SimpleNamespace, caplog: pytest.LogCaptureFixture) -> None:
    use_provider(ctx.api, provider_for(responding(ok_body("ZZ-OLLAMA-REPLY-TEXT"))))
    with caplog.at_level(logging.DEBUG):
        post_message(ctx.api, ctx.cid, ctx.vid, "ZZ-CUSTOMER-TEXT please book an appointment")
    use_provider(ctx.api, provider_for(responding({"error": "ZZ-ERROR-BODY"}, status=500)))
    with caplog.at_level(logging.DEBUG):
        post_message(ctx.api, ctx.cid, ctx.vid, "ZZ-CUSTOMER-TEXT-2 please book an appointment")
    for secret in ("ZZ-OLLAMA-REPLY-TEXT", "ZZ-CUSTOMER-TEXT", "ZZ-ERROR-BODY"):
        assert secret not in caplog.text
    assert "ollama_reply" in caplog.text  # useful metadata is still logged
