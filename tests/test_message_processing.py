"""Step 5 API tests: POST /clients/{id}/conversations/{id}/messages.

Each test gets a fresh in-memory database; convoops.db is never touched. No test
needs internet access or an API key.
"""

import logging
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Conversation, ConversationStatus, Message
from app.providers.base import (
    LLMConnectionError,
    LLMProviderError,
    LLMResponse,
    LLMTimeoutError,
)
from app.providers.factory import UnsupportedProviderError, get_llm_provider
from tests.helpers import (
    BOOKING,
    CLIENTS,
    RaisingProvider,
    RecordingProvider,
    StaticProvider,
    make_client,
    make_conversation,
    messages_url,
    post_message,
    use_provider,
)


@pytest.fixture()
def ctx(app_client: TestClient) -> SimpleNamespace:
    client = make_client(app_client)
    conversation = make_conversation(app_client, client["id"])
    return SimpleNamespace(api=app_client, cid=client["id"], vid=conversation["id"])


def stored(db: Session, conversation_id: str) -> list[Message]:
    db.expire_all()
    return list(
        db.scalars(
            select(Message).where(Message.conversation_id == uuid.UUID(conversation_id)).order_by(Message.sequence)
        )
    )


def conversation_row(db: Session, conversation_id: str) -> Conversation:
    db.expire_all()
    row = db.get(Conversation, uuid.UUID(conversation_id))
    assert row is not None
    return row


# ============================ successful processing ===============================
def test_successful_processing_returns_a_structured_response(ctx: SimpleNamespace) -> None:
    response = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="customer-message-1001")
    assert response.status_code == 201
    body = response.json()
    assert body["conversation_id"] == ctx.vid
    assert body["conversation_status"] == "awaiting_response"
    assert body["idempotent_replay"] is False
    assert body["processing_status"] == "processed"
    assert body["reply_type"] == "generated"
    assert body["failure_category"] is None
    assert body["intent"] == {"intent": "appointment_booking", "confidence": 0.9, "method": "deterministic_rules_v1"}
    assert (body["provider"], body["model"]) == ("mock", "mock-deterministic-v1")
    assert body["inbound_message"]["content"] == BOOKING
    assert "BrightPath Telehealth" in body["outbound_message"]["content"]
    assert set(body) == {
        "conversation_id", "conversation_status", "idempotent_replay", "processing_status", "reply_type",
        "failure_category", "intent", "provider", "model", "inbound_message", "outbound_message",
    }  # fmt: skip


def test_inbound_and_outbound_messages_are_persisted(ctx: SimpleNamespace, db_session: Session) -> None:
    body = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="k-1").json()
    inbound, outbound = stored(db_session, ctx.vid)

    assert str(inbound.id) == body["inbound_message"]["id"] and str(outbound.id) == body["outbound_message"]["id"]
    assert (inbound.sender_type.value, inbound.direction.value) == ("customer", "inbound")
    assert inbound.content == BOOKING and inbound.idempotency_key == "k-1"
    assert inbound.processing_status.value == "processed"
    assert (inbound.intent, inbound.intent_confidence, inbound.intent_method) == (
        "appointment_booking", 0.9, "deterministic_rules_v1",
    )  # fmt: skip

    assert (outbound.sender_type.value, outbound.direction.value) == ("ai", "outbound")
    assert outbound.in_reply_to_id == inbound.id
    assert (outbound.provider, outbound.model) == ("mock", "mock-deterministic-v1")
    assert outbound.processing_status.value == "generated" and outbound.failure_category is None
    assert outbound.idempotency_key is None and outbound.intent is None


def test_sequence_numbers_increase_across_messages(ctx: SimpleNamespace, db_session: Session) -> None:
    for text in ("book an appointment", "how much does it cost", "what are your hours?"):
        assert post_message(ctx.api, ctx.cid, ctx.vid, text).status_code == 201
    rows = stored(db_session, ctx.vid)
    assert [m.sequence for m in rows] == [1, 2, 3, 4, 5, 6]
    assert [m.direction.value for m in rows] == ["inbound", "outbound"] * 3
    assert conversation_row(db_session, ctx.vid).last_message_sequence == 6


def test_history_endpoint_shows_processed_messages_in_order(ctx: SimpleNamespace) -> None:
    post_message(ctx.api, ctx.cid, ctx.vid, "book an appointment")
    post_message(ctx.api, ctx.cid, ctx.vid, "how much does it cost")
    history = ctx.api.get(messages_url(ctx.cid, ctx.vid)).json()
    assert history["total"] == 4
    assert [m["sequence"] for m in history["items"]] == [1, 2, 3, 4]
    assert [m["direction"] for m in history["items"]] == ["inbound", "outbound", "inbound", "outbound"]
    assert history["items"][1]["in_reply_to_id"] == history["items"][0]["id"]


def test_conversation_timestamps_and_status_are_updated(ctx: SimpleNamespace, db_session: Session) -> None:
    before = conversation_row(db_session, ctx.vid)
    assert before.last_message_at is None and before.status is ConversationStatus.OPEN
    created, updated_before = before.created_at, before.updated_at

    post_message(ctx.api, ctx.cid, ctx.vid, BOOKING)
    after = conversation_row(db_session, ctx.vid)
    outbound = stored(db_session, ctx.vid)[-1]
    assert after.last_message_at == outbound.created_at  # stamped when the reply was stored
    assert after.last_message_at >= created
    assert after.updated_at > updated_before
    assert after.status is ConversationStatus.AWAITING_RESPONSE


def test_resolved_conversation_is_reopened_by_a_customer_message(ctx: SimpleNamespace, db_session: Session) -> None:
    post_message(ctx.api, ctx.cid, ctx.vid, BOOKING)
    row = conversation_row(db_session, ctx.vid)
    row.status = ConversationStatus.RESOLVED
    db_session.commit()

    body = post_message(ctx.api, ctx.cid, ctx.vid, "one more question: how much is a visit?").json()
    assert body["conversation_status"] == "awaiting_response"  # resolved -> open -> awaiting_response
    assert conversation_row(db_session, ctx.vid).status is ConversationStatus.AWAITING_RESPONSE


def test_second_customer_message_while_awaiting_response_is_processed(ctx: SimpleNamespace) -> None:
    assert post_message(ctx.api, ctx.cid, ctx.vid, BOOKING).json()["conversation_status"] == "awaiting_response"
    second = post_message(ctx.api, ctx.cid, ctx.vid, "actually, how much does it cost?")
    assert second.status_code == 201
    assert second.json()["conversation_status"] == "awaiting_response"
    assert second.json()["intent"]["intent"] == "pricing"


# ============================ intents and demo replies ==============================
@pytest.mark.parametrize(
    "text, intent, reply_fragment",
    [
        ("I would like to book an appointment", "appointment_booking", "appointment request"),
        ("How much does a visit cost?", "pricing", "don't have verified pricing"),
        ("Can I reschedule to Friday?", "reschedule", "can't change appointments myself"),
        ("I need to cancel my appointment", "cancellation", "can't cancel appointments myself"),
        ("What are your opening hours?", "general_question", "don't have verified clinic details"),
        ("I want to talk to a real person", "human_handoff", "does not notify anyone in real time"),
        ("asdf qwerty", "unknown", "not sure I understood"),
    ],
)
def test_each_intent_gets_a_realistic_demo_reply(
    ctx: SimpleNamespace, text: str, intent: str, reply_fragment: str
) -> None:
    body = post_message(ctx.api, ctx.cid, ctx.vid, text).json()
    assert body["intent"]["intent"] == intent
    assert reply_fragment in body["outbound_message"]["content"]
    assert body["reply_type"] == "generated"


def test_medical_question_gets_a_safe_reply_without_advice(ctx: SimpleNamespace) -> None:
    body = post_message(ctx.api, ctx.cid, ctx.vid, "I have chest pain, what medication should I take?").json()
    reply = body["outbound_message"]["content"].lower()
    assert "not able to give medical advice" in reply and "qualified clinician" in reply
    assert "emergency number" in reply
    assert "you should take" not in reply and "diagnos" not in reply.replace("diagnoses", "")


def test_human_handoff_keeps_the_conversation_open_for_staff(ctx: SimpleNamespace) -> None:
    body = post_message(ctx.api, ctx.cid, ctx.vid, "I want to talk to a real person").json()
    assert body["conversation_status"] == "open"
    assert "has been notified" not in body["outbound_message"]["content"].lower()


# ============================ provider interaction =================================
def test_conversation_history_is_passed_to_the_provider(ctx: SimpleNamespace) -> None:
    provider = RecordingProvider(reply="Synthetic reply.")
    use_provider(ctx.api, provider)
    post_message(ctx.api, ctx.cid, ctx.vid, "first message about booking")
    post_message(ctx.api, ctx.cid, ctx.vid, "second message about pricing")

    assert len(provider.requests) == 2
    assert [(m.role, m.content) for m in provider.requests[0].messages] == [("customer", "first message about booking")]
    assert [(m.role, m.content) for m in provider.requests[1].messages] == [
        ("customer", "first message about booking"),
        ("assistant", "Synthetic reply."),
        ("customer", "second message about pricing"),
    ]  # oldest first, ending with the newest customer message


def test_provider_receives_intent_business_context_and_safety_rules(ctx: SimpleNamespace) -> None:
    provider = RecordingProvider()
    use_provider(ctx.api, provider)
    post_message(ctx.api, ctx.cid, ctx.vid, "How much does a visit cost?")
    request = provider.requests[0]
    assert (request.intent, request.intent_confidence) == ("pricing", 0.9)
    assert request.business_name.startswith("BrightPath Telehealth")
    assert request.timeout_seconds == 10.0 and request.medical_topic is False
    assert "BrightPath Telehealth" in request.system_prompt
    for rule in ("Never give medical advice", "Never invent prices", "staff member has been notified"):
        assert rule in request.system_prompt


def test_medical_flag_is_passed_to_the_provider(ctx: SimpleNamespace) -> None:
    provider = RecordingProvider()
    use_provider(ctx.api, provider)
    post_message(ctx.api, ctx.cid, ctx.vid, "I have a fever and a rash")
    assert provider.requests[0].medical_topic is True


def test_custom_provider_metadata_is_recorded_on_the_reply(ctx: SimpleNamespace, db_session: Session) -> None:
    use_provider(ctx.api, RecordingProvider(reply="Hi there."))
    body = post_message(ctx.api, ctx.cid, ctx.vid).json()
    assert (body["provider"], body["model"]) == ("recording", "recording-v1")
    assert (stored(db_session, ctx.vid)[1].provider, stored(db_session, ctx.vid)[1].model) == ("recording", "recording-v1")


# ============================ idempotency =====================================
def test_same_key_and_same_content_returns_the_original_result(ctx: SimpleNamespace, db_session: Session) -> None:
    provider = RecordingProvider()
    use_provider(ctx.api, provider)
    first = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="retry-1")
    second = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="retry-1")

    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["idempotent_replay"] is False and second.json()["idempotent_replay"] is True
    assert second.json()["inbound_message"] == first.json()["inbound_message"]
    assert second.json()["outbound_message"] == first.json()["outbound_message"]
    assert second.json()["intent"] == first.json()["intent"]
    assert len(provider.requests) == 1  # no second AI call
    assert len(stored(db_session, ctx.vid)) == 2  # no duplicate inbound or outbound message


def test_replay_ignores_surrounding_whitespace_in_content(ctx: SimpleNamespace) -> None:
    first = post_message(ctx.api, ctx.cid, ctx.vid, "book an appointment", key="retry-ws")
    second = post_message(ctx.api, ctx.cid, ctx.vid, "  book an appointment \n", key="retry-ws")
    assert second.status_code == 200 and second.json()["idempotent_replay"] is True
    assert second.json()["inbound_message"]["id"] == first.json()["inbound_message"]["id"]


def test_same_key_with_different_content_is_a_409_conflict(ctx: SimpleNamespace, db_session: Session) -> None:
    provider = RecordingProvider()
    use_provider(ctx.api, provider)
    post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="retry-2")
    response = post_message(ctx.api, ctx.cid, ctx.vid, "A completely different message", key="retry-2")

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "idempotency_key_conflict",
            "message": "This idempotency key was already used with different message content.",
        }
    }
    assert len(stored(db_session, ctx.vid)) == 2 and len(provider.requests) == 1
    assert stored(db_session, ctx.vid)[0].content == BOOKING  # the original is untouched


def test_the_same_key_may_be_used_in_a_different_conversation(ctx: SimpleNamespace, db_session: Session) -> None:
    other = make_conversation(ctx.api, ctx.cid, "cust_2002")
    assert post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="shared-key").status_code == 201
    second = post_message(ctx.api, ctx.cid, other["id"], "How much is a visit?", key="shared-key")
    assert second.status_code == 201 and second.json()["idempotent_replay"] is False
    assert len(stored(db_session, other["id"])) == 2


def test_requests_without_a_key_are_never_deduplicated(ctx: SimpleNamespace, db_session: Session) -> None:
    assert post_message(ctx.api, ctx.cid, ctx.vid, BOOKING).status_code == 201
    assert post_message(ctx.api, ctx.cid, ctx.vid, BOOKING).status_code == 201
    assert len(stored(db_session, ctx.vid)) == 4


def test_replay_still_returns_the_stored_result_after_the_conversation_is_closed(
    ctx: SimpleNamespace, db_session: Session
) -> None:
    first = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="retry-3")
    conversation_row(db_session, ctx.vid).status = ConversationStatus.CLOSED
    db_session.commit()

    replay = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="retry-3")
    assert replay.status_code == 200 and replay.json()["inbound_message"]["id"] == first.json()["inbound_message"]["id"]
    assert post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="brand-new").status_code == 409  # new work is refused


def test_a_fallback_result_is_also_replayed_not_regenerated(ctx: SimpleNamespace, db_session: Session) -> None:
    provider = RaisingProvider(LLMTimeoutError("timed out"))
    use_provider(ctx.api, provider)
    first = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="retry-4")
    second = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="retry-4")
    assert first.json()["reply_type"] == second.json()["reply_type"] == "fallback"
    assert second.json()["idempotent_replay"] is True and provider.calls == 1
    assert len(stored(db_session, ctx.vid)) == 2


# ============================ provider failures ======================================
@pytest.mark.parametrize(
    "exception, category",
    [
        (LLMTimeoutError("secret-token-123 timed out"), "provider_timeout"),
        (LLMConnectionError("secret-token-123 refused"), "provider_connection"),
        (LLMProviderError("secret-token-123 bad gateway"), "provider_error"),
        (RuntimeError("secret-token-123 unexpected bug"), "provider_error"),
        (KeyError("secret-token-123"), "provider_error"),
    ],
)
def test_provider_failure_produces_a_stored_safe_fallback(
    ctx: SimpleNamespace, db_session: Session, exception: Exception, category: str
) -> None:
    use_provider(ctx.api, RaisingProvider(exception))
    response = post_message(ctx.api, ctx.cid, ctx.vid, "I would like to book an appointment")

    assert response.status_code == 201  # a valid API response, not an error
    body = response.json()
    assert (body["processing_status"], body["reply_type"], body["failure_category"]) == ("failed", "fallback", category)
    assert (body["provider"], body["model"]) == ("fallback", "deterministic-fallback-v1")
    assert body["conversation_status"] == "open"  # staff should follow up
    assert "saved" in body["outbound_message"]["content"] and "follow up" in body["outbound_message"]["content"]
    assert "secret-token-123" not in response.text  # internals never reach the client

    inbound, outbound = stored(db_session, ctx.vid)  # the inbound message was not lost
    assert inbound.content == "I would like to book an appointment"
    assert inbound.processing_status.value == "failed" and inbound.failure_category.value == category  # type: ignore[union-attr]
    assert (inbound.provider, inbound.model) == ("flaky", "flaky-v1")  # which provider was attempted
    assert inbound.intent == "appointment_booking"  # classification still recorded
    assert (outbound.sender_type.value, outbound.processing_status.value) == ("system", "fallback")
    assert outbound.failure_category.value == category  # type: ignore[union-attr]
    assert outbound.in_reply_to_id == inbound.id


@pytest.mark.parametrize(
    "result",
    [
        None,
        "just a string",
        object(),
        LLMResponse(content="", provider="static", model="static-v1"),
        LLMResponse(content="   \n\t", provider="static", model="static-v1"),
        LLMResponse(content="x" * 1601, provider="static", model="static-v1"),
        LLMResponse(content="Valid text", provider="", model="static-v1"),
        LLMResponse(content="Valid text", provider="static", model=""),
        LLMResponse(content="Valid text", provider="p" * 65, model="static-v1"),
        LLMResponse(content="Valid text", provider="static", model="m" * 129),
        LLMResponse(content=None, provider="static", model="static-v1"),  # type: ignore[arg-type]
        LLMResponse(content=12345, provider="static", model="static-v1"),  # type: ignore[arg-type]
    ],
)
def test_invalid_provider_output_falls_back_safely(
    ctx: SimpleNamespace, db_session: Session, result: Any
) -> None:
    use_provider(ctx.api, StaticProvider(result))
    response = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING)
    assert response.status_code == 201
    body = response.json()
    assert (body["reply_type"], body["failure_category"]) == ("fallback", "provider_error")
    assert len(stored(db_session, ctx.vid)) == 2


def test_valid_provider_output_is_trimmed_and_accepted(ctx: SimpleNamespace) -> None:
    use_provider(ctx.api, StaticProvider(LLMResponse(content="  Hello!  \n", provider="static", model="static-v1")))
    body = post_message(ctx.api, ctx.cid, ctx.vid, BOOKING).json()
    assert body["reply_type"] == "generated" and body["outbound_message"]["content"] == "Hello!"


def test_provider_failure_details_are_not_logged_but_the_event_is(
    ctx: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    use_provider(ctx.api, RaisingProvider(RuntimeError("secret-token-123 leaked body: ZZ-CUSTOMER-TEXT")))
    with caplog.at_level(logging.DEBUG):
        post_message(ctx.api, ctx.cid, ctx.vid, "ZZ-CUSTOMER-TEXT book an appointment")
    assert "provider_unexpected_error" in caplog.text and "RuntimeError" in caplog.text
    assert "secret-token-123" not in caplog.text and "ZZ-CUSTOMER-TEXT" not in caplog.text


# ============================ validation and tenant rules =================================
def test_unknown_client_returns_404(ctx: SimpleNamespace) -> None:
    provider = RecordingProvider()
    use_provider(ctx.api, provider)
    response = post_message(ctx.api, str(uuid.uuid4()), ctx.vid)
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"
    assert provider.requests == []


def test_unknown_conversation_returns_404(ctx: SimpleNamespace) -> None:
    provider = RecordingProvider()
    use_provider(ctx.api, provider)
    response = post_message(ctx.api, ctx.cid, str(uuid.uuid4()))
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"
    assert provider.requests == []


def test_cross_tenant_access_returns_404_and_stores_nothing(ctx: SimpleNamespace, db_session: Session) -> None:
    other_client = make_client(ctx.api, "other-clinic")
    provider = RecordingProvider()
    use_provider(ctx.api, provider)

    cross = post_message(ctx.api, other_client["id"], ctx.vid, BOOKING, key="x-tenant")
    missing = post_message(ctx.api, other_client["id"], str(uuid.uuid4()), BOOKING, key="x-tenant")
    assert cross.status_code == missing.status_code == 404
    assert cross.json()["error"]["message"] == f"Conversation {ctx.vid} was not found."
    assert provider.requests == [] and stored(db_session, ctx.vid) == []
    # the real owner is unaffected
    assert post_message(ctx.api, ctx.cid, ctx.vid, BOOKING, key="x-tenant").status_code == 201


def test_a_key_used_by_another_tenant_never_leaks_across_clients(ctx: SimpleNamespace) -> None:
    other_client = make_client(ctx.api, "other-clinic")
    other_conv = make_conversation(ctx.api, other_client["id"])
    post_message(ctx.api, ctx.cid, ctx.vid, "tenant one secret booking", key="same-key")
    response = post_message(ctx.api, other_client["id"], other_conv["id"], "book a visit", key="same-key")
    assert response.status_code == 201 and response.json()["idempotent_replay"] is False
    assert "tenant one" not in response.text


@pytest.mark.parametrize("bad", ["not-a-uuid", "123", "00000000-0000-0000-0000"])
def test_invalid_ids_return_422(ctx: SimpleNamespace, bad: str) -> None:
    assert post_message(ctx.api, bad, ctx.vid).status_code == 422
    assert post_message(ctx.api, ctx.cid, bad).status_code == 422


def test_inactive_client_is_rejected_and_nothing_is_stored(ctx: SimpleNamespace, db_session: Session) -> None:
    ctx.api.patch(f"{CLIENTS}/{ctx.cid}", json={"configuration_status": "inactive"})
    provider = RecordingProvider()
    use_provider(ctx.api, provider)
    response = post_message(ctx.api, ctx.cid, ctx.vid)
    assert response.status_code == 409 and response.json()["error"]["code"] == "client_not_active"
    assert provider.requests == [] and stored(db_session, ctx.vid) == []


def test_closed_conversation_is_rejected_and_nothing_is_stored(ctx: SimpleNamespace, db_session: Session) -> None:
    conversation_row(db_session, ctx.vid).status = ConversationStatus.CLOSED
    db_session.commit()
    provider = RecordingProvider()
    use_provider(ctx.api, provider)

    response = post_message(ctx.api, ctx.cid, ctx.vid)
    assert response.status_code == 409 and response.json()["error"]["code"] == "conversation_closed"
    assert provider.requests == [] and stored(db_session, ctx.vid) == []
    assert conversation_row(db_session, ctx.vid).status is ConversationStatus.CLOSED  # never reopened


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"content": ""},
        {"content": "   "},
        {"content": "\t\n \r"},
        {"content": "x" * 1001},
        {"content": "bad\u0000char"},
        {"content": 123},
        {"content": None},
        {"content": ["list"]},
        {"content": "hi", "unexpected": 1},
        {"content": "hi", "sender_type": "ai"},
        {"idempotency_key": "k-only"},
        {"content": "hi", "idempotency_key": ""},
        {"content": "hi", "idempotency_key": "has space"},
        {"content": "hi", "idempotency_key": "x" * 129},
        {"content": "hi", "idempotency_key": 123},
        {"content": "hi", "idempotency_key": "-leading-hyphen"},
    ],
)
def test_invalid_payloads_are_rejected_before_anything_is_stored(
    ctx: SimpleNamespace, db_session: Session, body: dict[str, Any]
) -> None:
    provider = RecordingProvider()
    use_provider(ctx.api, provider)
    response = ctx.api.post(messages_url(ctx.cid, ctx.vid), json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error" and error["details"]
    assert provider.requests == [] and stored(db_session, ctx.vid) == []


def test_non_json_body_is_rejected(ctx: SimpleNamespace) -> None:
    response = ctx.api.post(
        messages_url(ctx.cid, ctx.vid), content="not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"


def test_message_length_boundaries(ctx: SimpleNamespace) -> None:
    assert post_message(ctx.api, ctx.cid, ctx.vid, "a" * 1000).status_code == 201  # exactly the limit
    assert post_message(ctx.api, ctx.cid, ctx.vid, "a" * 1001).status_code == 422
    assert post_message(ctx.api, ctx.cid, ctx.vid, "b", key="k" * 128).status_code == 201


def test_content_is_trimmed_before_it_is_stored(ctx: SimpleNamespace, db_session: Session) -> None:
    post_message(ctx.api, ctx.cid, ctx.vid, "   book an appointment \n")
    assert stored(db_session, ctx.vid)[0].content == "book an appointment"


def test_misconfigured_provider_returns_a_generic_500_and_stores_nothing(
    ctx: SimpleNamespace, db_session: Session
) -> None:
    def broken() -> None:
        raise UnsupportedProviderError("secret-config-detail: provider 'groq' not implemented")

    ctx.api.app.dependency_overrides[get_llm_provider] = broken  # type: ignore[attr-defined]
    lenient = TestClient(ctx.api.app, raise_server_exceptions=False)
    response = post_message(lenient, ctx.cid, ctx.vid)
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "An internal error occurred."}}
    assert "secret-config-detail" not in response.text
    assert stored(db_session, ctx.vid) == []


# ============================ logging ================================================
def test_message_bodies_and_keys_are_never_logged(ctx: SimpleNamespace, caplog: pytest.LogCaptureFixture) -> None:
    use_provider(ctx.api, RecordingProvider(reply="ZZ-PRIVATE-REPLY-TEXT"))
    with caplog.at_level(logging.DEBUG):
        response = post_message(
            ctx.api, ctx.cid, ctx.vid, "ZZ-PRIVATE-INBOUND-TEXT please book an appointment", key="ZZ-KEY-VALUE"
        )
    assert "ZZ-PRIVATE-REPLY-TEXT" in response.text  # the API returns it...
    for secret in ("ZZ-PRIVATE-INBOUND-TEXT", "ZZ-PRIVATE-REPLY-TEXT", "ZZ-KEY-VALUE"):
        assert secret not in caplog.text  # ...but the logs never do
    assert "message_processed" in caplog.text and "inbound_chars=" in caplog.text  # useful metadata only


# ============================ regression: existing API ===================================
def test_existing_endpoints_still_work_alongside_the_new_one(ctx: SimpleNamespace) -> None:
    paths = ctx.api.get("/openapi.json").json()["paths"]
    assert set(paths[CLIENTS]) == {"get", "post"}
    assert set(paths[f"{CLIENTS}/{{client_id}}"]) == {"get", "patch"}
    assert set(paths[f"{CLIENTS}/{{client_id}}/conversations"]) == {"get", "post"}
    assert set(paths[f"{CLIENTS}/{{client_id}}/conversations/{{conversation_id}}"]) == {"get"}
    assert set(paths[f"{CLIENTS}/{{client_id}}/conversations/{{conversation_id}}/messages"]) == {"get", "post"}
    assert ctx.api.get("/health").json()["status"] == "ok"
    assert ctx.api.get(f"{CLIENTS}/{ctx.cid}").status_code == 200
    assert ctx.api.get(f"{CLIENTS}/{ctx.cid}/conversations/{ctx.vid}").status_code == 200
    assert ctx.api.get(f"{CLIENTS}/{ctx.cid}/conversations").json()["total"] == 1


def test_openapi_documents_the_message_endpoint(ctx: SimpleNamespace) -> None:
    operation = ctx.api.get("/openapi.json").json()["paths"][
        f"{CLIENTS}/{{client_id}}/conversations/{{conversation_id}}/messages"
    ]["post"]
    assert set(operation["responses"]) >= {"200", "201", "404", "409", "422"}
    assert "idempotency" in operation["description"].lower() and operation["summary"]
    schema = ctx.api.get("/openapi.json").json()["components"]["schemas"]["MessageCreate"]
    assert schema["examples"][0]["idempotency_key"] == "customer-message-1001"
