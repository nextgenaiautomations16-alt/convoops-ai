"""Provider interface, mock provider, provider factory and status-transition rules."""

import pytest

from app.core.config import Settings
from app.models.conversation import ConversationStatus as S
from app.providers import factory
from app.providers.base import ChatMessage, LLMRequest, LLMResponse
from app.providers.factory import UnsupportedProviderError, get_llm_provider
from app.providers.mock import MockLLMProvider
from app.services.conversation_status import (
    ALLOWED_TRANSITIONS,
    InvalidStatusTransitionError,
    ensure_transition_allowed,
)
from app.services.intent_classifier import Intent


def make_request(
    intent: Intent = Intent.UNKNOWN,
    medical: bool = False,
    business: str = "BrightPath Telehealth",
) -> LLMRequest:
    return LLMRequest(
        system_prompt="system",
        messages=(ChatMessage(role="customer", content="hello"),),
        intent=intent.value,
        intent_confidence=0.9,
        business_name=business,
        timeout_seconds=5.0,
        medical_topic=medical,
    )


# ---- mock provider -----------------------------------------------------------------


@pytest.mark.parametrize("intent", list(Intent))
def test_mock_replies_for_every_intent(intent: Intent) -> None:
    response = MockLLMProvider().generate(make_request(intent))
    assert isinstance(response, LLMResponse)
    assert response.content.strip()
    assert "BrightPath Telehealth" in response.content


def test_mock_is_deterministic() -> None:
    provider = MockLLMProvider()
    request = make_request(Intent.PRICING)
    assert provider.generate(request) == provider.generate(request)


def test_mock_reports_provider_and_model_metadata() -> None:
    response = MockLLMProvider().generate(make_request(Intent.PRICING))
    assert (response.provider, response.model) == ("mock", "mock-deterministic-v1")
    assert (MockLLMProvider.name, MockLLMProvider.model) == (
        "mock",
        "mock-deterministic-v1",
    )


@pytest.mark.parametrize("intent", list(Intent))
def test_mock_never_invents_prices_or_availability(intent: Intent) -> None:
    content = MockLLMProvider().generate(make_request(intent)).content
    assert "$" not in content and not any(ch.isdigit() for ch in content)


def test_mock_pricing_reply_defers_to_staff() -> None:
    content = MockLLMProvider().generate(make_request(Intent.PRICING)).content.lower()
    assert "don't have verified pricing" in content and "staff" in content


def test_mock_booking_reply_does_not_claim_a_booking() -> None:
    content = MockLLMProvider().generate(
        make_request(Intent.APPOINTMENT_BOOKING)
    ).content.lower()

    assert "confirm" in content
    assert "can't see or confirm real appointment times" in content
    assert "you are booked" not in content
    assert "booked for" not in content


def test_mock_cancellation_and_reschedule_do_not_claim_completion() -> None:
    cancel = MockLLMProvider().generate(
        make_request(Intent.CANCELLATION)
    ).content.lower()

    resched = MockLLMProvider().generate(
        make_request(Intent.RESCHEDULE)
    ).content.lower()

    assert "can't cancel appointments myself" in cancel
    assert "can't change appointments myself" in resched


def test_mock_handoff_does_not_claim_a_person_was_notified() -> None:
    content = MockLLMProvider().generate(
        make_request(Intent.HUMAN_HANDOFF)
    ).content.lower()

    assert "does not notify anyone in real time" in content
    assert "has been notified" not in content
    assert "we have notified" not in content


def test_mock_medical_topic_gets_a_safe_reply_regardless_of_intent() -> None:
    for intent in (
        Intent.GENERAL_QUESTION,
        Intent.APPOINTMENT_BOOKING,
        Intent.UNKNOWN,
    ):
        response = MockLLMProvider().generate(
            make_request(intent, medical=True)
        )

        # Medical replies must be attributed to the dedicated safety provider.
        assert (response.provider, response.model) == (
            "safety_guard",
            "deterministic-medical-v1",
        )

        content = response.content.lower()

        assert "not able to give medical advice" in content
        assert "qualified clinician" in content
        assert "emergency number" in content

        for forbidden in (
            "you should take",
            "you have ",
            "diagnosis is",
            "i recommend taking",
        ):
            assert forbidden not in content


def test_mock_business_name_with_braces_cannot_break_the_template() -> None:
    content = MockLLMProvider().generate(
        make_request(Intent.PRICING, business="Odd {name} Clinic")
    ).content

    assert "Odd {name} Clinic" in content


def test_mock_works_without_any_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GROQ_API_KEY",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    assert MockLLMProvider().generate(make_request(Intent.PRICING)).content


# ---- factory ---------------------------------------------------------------------------


def test_factory_defaults_to_the_mock_provider() -> None:
    assert Settings().llm_provider == "mock"
    assert isinstance(get_llm_provider(), MockLLMProvider)


def test_factory_respects_llm_provider_setting_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import get_settings

    monkeypatch.setenv("LLM_PROVIDER", " MOCK ")
    get_settings.cache_clear()

    try:
        assert isinstance(get_llm_provider(), MockLLMProvider)
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


@pytest.mark.parametrize("name", ["groq", "does-not-exist"])
def test_factory_rejects_unimplemented_or_unknown_providers(name: str) -> None:
    with pytest.raises(UnsupportedProviderError):
        factory._create(name)


# ---- status transition rules --------------------------------------------------------------


@pytest.mark.parametrize(
    "current, target",
    [
        (S.OPEN, S.AWAITING_RESPONSE),
        (S.OPEN, S.RESOLVED),
        (S.OPEN, S.CLOSED),
        (S.AWAITING_RESPONSE, S.OPEN),
        (S.AWAITING_RESPONSE, S.RESOLVED),
        (S.AWAITING_RESPONSE, S.CLOSED),
        (S.RESOLVED, S.OPEN),
        (S.RESOLVED, S.CLOSED),
        (S.OPEN, S.OPEN),
        (S.CLOSED, S.CLOSED),
    ],
)
def test_allowed_transitions(current: S, target: S) -> None:
    ensure_transition_allowed(current, target)


@pytest.mark.parametrize(
    "current, target",
    [
        (S.RESOLVED, S.AWAITING_RESPONSE),
        (S.CLOSED, S.OPEN),
        (S.CLOSED, S.RESOLVED),
        (S.CLOSED, S.AWAITING_RESPONSE),
    ],
)
def test_invalid_transitions_are_rejected(current: S, target: S) -> None:
    with pytest.raises(InvalidStatusTransitionError) as info:
        ensure_transition_allowed(current, target)

    assert info.value.code == "invalid_status_transition"
    assert info.value.status_code == 409


def test_closed_is_final_and_every_status_is_covered() -> None:
    assert ALLOWED_TRANSITIONS[S.CLOSED] == frozenset()
    assert set(ALLOWED_TRANSITIONS) == set(S)