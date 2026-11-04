"""Step 5 reliability tests: concurrency, idempotency races, rollback and crash recovery.

Service-level tests call MessageProcessingService directly so a test can break one
commit at a time. Concurrency tests use a real SQLite FILE (separate connections per
thread), because an in-memory test database shares one connection and cannot show races.
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.database import Base, build_engine, get_db
from app.main import create_app
from app.models import (
    Client,
    ConfigurationStatus,
    Conversation,
    ConversationStatus,
    FailureCategory,
    Message,
    MessageProcessingStatus,
)
from app.providers.base import LLMProvider, LLMRequest, LLMResponse, LLMTimeoutError
from app.providers.factory import get_llm_provider
from app.services.conversation_service import ClientNotActiveError
from app.services.intent_classifier import classify_intent
from app.services.message_processing_service import (
    MessageProcessingService,
    ProcessingInProgressError,
    _Outcome,
)
from tests.helpers import BOOKING, RaisingProvider, RecordingProvider

QUICK = {"idempotency_wait_seconds": 0.2}


# ---- helpers ------------------------------------------------------------------------
def seed(db: Session) -> tuple[uuid.UUID, uuid.UUID]:
    client = Client(
        business_name="BrightPath Telehealth",
        slug="brightpath",
        industry="healthcare",
        timezone="America/New_York",
        business_description="Synthetic demo clinic.",
        configuration_status=ConfigurationStatus.ACTIVE,
    )
    db.add(client)
    db.commit()
    conversation = Conversation(client_id=client.id, customer_id="cust_1001")
    db.add(conversation)
    db.commit()
    return client.id, conversation.id


_OPEN_SESSIONS: list[Session] = []


@pytest.fixture(autouse=True)
def _close_sessions() -> Any:
    yield
    while _OPEN_SESSIONS:
        session = _OPEN_SESSIONS.pop()
        try:
            session.close()
        except SQLAlchemyError:
            pass  # pytest may already have disposed the in-memory database; nothing left to clean up


def open_session(engine: Engine) -> Session:
    """A session configured exactly like the app's (autoflush off, no expire on commit)."""
    session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()
    _OPEN_SESSIONS.append(session)
    return session


def service(db: Session, provider: LLMProvider, **overrides: Any) -> MessageProcessingService:
    return MessageProcessingService(db, provider, settings=Settings(**overrides))


def fail_commits(db: Session, fail_on: set[int]) -> dict[str, int]:
    """Make the Nth commit on this session raise a database error (1-based)."""
    real_commit = db.commit
    state = {"calls": 0}

    def flaky_commit() -> None:
        state["calls"] += 1
        if state["calls"] in fail_on:
            raise OperationalError("COMMIT", {}, Exception("secret internal database detail"))
        real_commit()

    db.commit = flaky_commit  # type: ignore[method-assign]
    return state


def fresh(engine: Engine, conversation_id: uuid.UUID) -> tuple[list[Message], Conversation]:
    """Read committed state through a brand-new session."""
    with Session(engine) as check:
        rows = list(check.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.sequence)))
        conversation = check.get(Conversation, conversation_id)
        assert conversation is not None
        for row in rows:
            check.expunge(row)
        check.expunge(conversation)
        return rows, conversation


@pytest.fixture()
def file_engine(tmp_path: Path) -> Any:
    engine = build_engine(f"sqlite:///{tmp_path / 'concurrency.db'}")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


class SlowProvider(RecordingProvider):
    """Takes a while to answer, so duplicate requests really overlap with the first one."""

    def __init__(self, delay: float = 0.4) -> None:
        super().__init__(reply="Slow synthetic reply.")
        self.delay = delay

    def generate(self, request: LLMRequest) -> LLMResponse:
        time.sleep(self.delay)
        return super().generate(request)


# ============================ history window ===========================================
def test_only_the_most_recent_messages_are_sent_to_the_provider(engine: Engine, db_session: Session) -> None:
    client_id, conversation_id = seed(db_session)
    provider = RecordingProvider(reply="R")
    svc = service(db_session, provider, history_window=3)
    for i in range(1, 5):
        svc.process_inbound(client_id, conversation_id, f"message number {i} about booking")

    last = provider.requests[-1].messages
    assert len(last) == 3  # window of 3: reply 3, then... newest customer message last
    assert [m.role for m in last] == ["customer", "assistant", "customer"]
    assert last[-1].content == "message number 4 about booking"


# ============================ concurrency (real SQLite file) ===============================
def run_threads(engine: Engine, provider: LLMProvider, client_id: uuid.UUID, conversation_id: uuid.UUID,
                keys: list[str | None], content: str | list[str] = BOOKING) -> list[dict[str, Any]]:  # fmt: skip
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    barrier = threading.Barrier(len(keys))
    contents = content if isinstance(content, list) else [content] * len(keys)

    def worker(index: int) -> dict[str, Any]:
        with factory() as db:
            svc = service(db, provider, idempotency_wait_seconds=15)
            barrier.wait()  # release all threads at the same moment
            result = svc.process_inbound(client_id, conversation_id, contents[index], keys[index])
            return {
                "inbound": result.inbound.id,
                "outbound": result.outbound.id,
                "replay": result.idempotent_replay,
                "reply_type": result.outbound.processing_status,
            }

    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        return list(pool.map(worker, range(len(keys))))


@pytest.mark.parametrize("provider_factory", [RecordingProvider, SlowProvider], ids=["fast-provider", "slow-provider"])
def test_concurrent_duplicate_requests_create_one_inbound_and_one_reply(
    file_engine: Engine, provider_factory: Any
) -> None:
    with Session(file_engine) as setup:
        client_id, conversation_id = seed(setup)
    provider = provider_factory()

    results = run_threads(file_engine, provider, client_id, conversation_id, ["same-key"] * 8)

    assert len({r["inbound"] for r in results}) == 1  # every caller got the SAME inbound message
    assert len({r["outbound"] for r in results}) == 1  # ...and the SAME reply
    assert sorted(r["replay"] for r in results) == [False] + [True] * 7  # exactly one request did the work
    assert len(provider.requests) == 1  # the AI was called once

    rows, conversation = fresh(file_engine, conversation_id)
    assert [m.sequence for m in rows] == [1, 2]  # no duplicates, no gaps
    assert [m.direction.value for m in rows] == ["inbound", "outbound"]
    assert rows[1].in_reply_to_id == rows[0].id
    assert conversation.last_message_sequence == 2  # every losing transaction rolled its increment back
    assert conversation.status is ConversationStatus.AWAITING_RESPONSE


def test_concurrent_requests_with_different_keys_all_succeed_with_unique_sequences(file_engine: Engine) -> None:
    with Session(file_engine) as setup:
        client_id, conversation_id = seed(setup)
    provider = RecordingProvider()

    results = run_threads(file_engine, provider, client_id, conversation_id, [f"key-{i}" for i in range(6)],
                          [f"message {i} about booking" for i in range(6)])  # fmt: skip

    assert all(not r["replay"] for r in results) and len({r["inbound"] for r in results}) == 6
    assert len(provider.requests) == 6
    rows, conversation = fresh(file_engine, conversation_id)
    assert sorted(m.sequence for m in rows) == list(range(1, 13))  # unique, contiguous
    inbound_ids = {m.id for m in rows if m.direction.value == "inbound"}
    assert {m.in_reply_to_id for m in rows if m.direction.value == "outbound"} == inbound_ids  # one reply each
    assert conversation.last_message_sequence == 12


def test_concurrent_duplicates_through_the_http_api(file_engine: Engine) -> None:
    with Session(file_engine) as setup:
        client_id, conversation_id = seed(setup)
    factory = sessionmaker(bind=file_engine, autoflush=False, expire_on_commit=False)
    provider = SlowProvider(delay=0.3)

    def override_db() -> Any:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_llm_provider] = lambda: provider
    url = f"/api/v1/clients/{client_id}/conversations/{conversation_id}/messages"
    barrier = threading.Barrier(6)

    def call(_: int) -> tuple[int, dict[str, Any]]:
        barrier.wait()
        response = TestClient(app).post(url, json={"content": BOOKING, "idempotency_key": "http-same-key"})
        return response.status_code, response.json()

    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(call, range(6)))

    assert sorted(status for status, _ in outcomes) == [200] * 5 + [201]  # one created, five replays
    assert len({body["inbound_message"]["id"] for _, body in outcomes}) == 1
    assert len({body["outbound_message"]["id"] for _, body in outcomes}) == 1
    assert len(provider.requests) == 1
    rows, _ = fresh(file_engine, conversation_id)
    assert len(rows) == 2


# ============================ transactions and rollback ====================================
def test_failure_storing_the_inbound_message_stores_nothing_and_leaves_sequences_clean(
    engine: Engine, db_session: Session
) -> None:
    client_id, conversation_id = seed(db_session)
    provider = RecordingProvider()
    fail_commits(db_session, {1})

    with pytest.raises(OperationalError):
        service(db_session, provider).process_inbound(client_id, conversation_id, BOOKING, "k-1")

    rows, conversation = fresh(engine, conversation_id)
    assert rows == [] and provider.requests == []  # nothing stored, the provider was never called
    assert conversation.last_message_sequence == 0 and conversation.last_message_at is None
    assert conversation.status is ConversationStatus.OPEN

    result = service(open_session(engine), provider).process_inbound(client_id, conversation_id, BOOKING, "k-1")
    assert (result.inbound.sequence, result.outbound.sequence) == (1, 2)  # the retry starts cleanly at 1


def test_failure_storing_the_reply_is_retried_as_a_stored_fallback(engine: Engine, db_session: Session) -> None:
    client_id, conversation_id = seed(db_session)
    provider = RecordingProvider()
    state = fail_commits(db_session, {2})  # commit 1 = inbound, commit 2 = reply (fails), commit 3 = fallback

    result = service(db_session, provider).process_inbound(client_id, conversation_id, BOOKING, "k-2")

    assert state["calls"] == 3 and len(provider.requests) == 1
    rows, conversation = fresh(engine, conversation_id)
    assert [m.sequence for m in rows] == [1, 2]  # the failed attempt left no partial rows or gaps
    inbound, outbound = rows
    assert inbound.content == BOOKING  # the inbound message was never lost
    assert outbound.processing_status is MessageProcessingStatus.FALLBACK
    assert outbound.failure_category is FailureCategory.INTERNAL_ERROR
    assert inbound.processing_status is MessageProcessingStatus.FAILED
    assert inbound.intent == "appointment_booking"
    assert (inbound.provider, inbound.model) == ("recording", "recording-v1")
    assert conversation.status is ConversationStatus.OPEN
    assert result.outbound.processing_status is MessageProcessingStatus.FALLBACK


def test_when_every_attempt_to_store_the_reply_fails_the_inbound_message_survives(
    engine: Engine, db_session: Session
) -> None:
    client_id, conversation_id = seed(db_session)
    fail_commits(db_session, {2, 3})

    with pytest.raises(SQLAlchemyError):
        service(db_session, RecordingProvider()).process_inbound(client_id, conversation_id, BOOKING, "k-3")

    rows, conversation = fresh(engine, conversation_id)
    assert len(rows) == 1  # only the inbound message
    inbound = rows[0]
    assert inbound.content == BOOKING and inbound.processing_status is MessageProcessingStatus.RECEIVED
    assert inbound.intent is None and inbound.failure_category is None  # the failed transaction left no partial update
    assert conversation.last_message_sequence == 1  # the reply's reserved sequence was rolled back


def test_a_mid_transaction_error_rolls_back_the_reserved_sequence(engine: Engine, db_session: Session) -> None:
    """The error happens AFTER the sequence number was reserved; the rollback must undo it."""
    client_id, conversation_id = seed(db_session)
    svc = service(db_session, RecordingProvider())
    original = MessageProcessingService._move_status
    calls = {"n": 0}

    def explode_once(self: MessageProcessingService, *args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OperationalError("UPDATE", {}, Exception("secret"))
        original(self, *args, **kwargs)

    MessageProcessingService._move_status = explode_once  # type: ignore[method-assign]
    try:
        svc.process_inbound(client_id, conversation_id, BOOKING)
    finally:
        MessageProcessingService._move_status = original  # type: ignore[method-assign]

    rows, _ = fresh(engine, conversation_id)
    assert [m.sequence for m in rows] == [1, 2]  # contiguous: the aborted attempt's number 2 was reused, not skipped


def test_database_failure_over_http_returns_a_generic_500_and_keeps_the_inbound_message(
    app_client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.helpers import make_client, make_conversation, messages_url, post_message

    client = make_client(app_client)
    conversation = make_conversation(app_client, client["id"])

    def boom(self: MessageProcessingService, *args: Any, **kwargs: Any) -> None:
        raise OperationalError("INSERT", {}, Exception("secret internal database detail"))

    monkeypatch.setattr(MessageProcessingService, "_persist_outcome", boom)
    response = post_message(app_client, client["id"], conversation["id"], BOOKING, key="http-fail")
    monkeypatch.undo()

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "An internal error occurred."}}
    assert "secret" not in response.text
    history = app_client.get(messages_url(client["id"], conversation["id"])).json()
    assert history["total"] == 1 and history["items"][0]["content"] == BOOKING  # not lost
    assert history["items"][0]["processing_status"] == "received"


# ============================ interrupted requests and recovery ================================
def stuck_after_inbound(engine: Engine, db_session: Session) -> tuple[uuid.UUID, uuid.UUID]:
    """Leave an inbound message with no reply (as if the process died after tx 1)."""
    client_id, conversation_id = seed(db_session)
    fail_commits(db_session, {2, 3})
    with pytest.raises(SQLAlchemyError):
        service(db_session, RecordingProvider()).process_inbound(client_id, conversation_id, BOOKING, "stuck-key")
    return client_id, conversation_id


def test_retry_of_a_request_that_is_still_in_flight_gets_a_retryable_409(engine: Engine, db_session: Session) -> None:
    client_id, conversation_id = stuck_after_inbound(engine, db_session)
    provider = RecordingProvider()

    with pytest.raises(ProcessingInProgressError) as info:
        service(open_session(engine), provider, processing_stale_seconds=60, **QUICK).process_inbound(
            client_id, conversation_id, BOOKING, "stuck-key"
        )
    assert info.value.code == "processing_in_progress" and info.value.status_code == 409
    assert provider.requests == []  # a young request is not taken over
    assert len(fresh(engine, conversation_id)[0]) == 1


def test_an_abandoned_request_is_resumed_on_retry_without_a_duplicate(engine: Engine, db_session: Session) -> None:
    client_id, conversation_id = stuck_after_inbound(engine, db_session)
    provider = RecordingProvider()

    result = service(open_session(engine), provider, processing_stale_seconds=0, **QUICK).process_inbound(
        client_id, conversation_id, BOOKING, "stuck-key"
    )

    assert result.idempotent_replay is True  # no new inbound message was created
    assert len(provider.requests) == 1
    rows, conversation = fresh(engine, conversation_id)
    assert [m.sequence for m in rows] == [1, 2] and rows[1].in_reply_to_id == rows[0].id
    assert rows[0].processing_status is MessageProcessingStatus.PROCESSED and rows[0].intent == "appointment_booking"
    assert conversation.status is ConversationStatus.AWAITING_RESPONSE

    again = service(open_session(engine), provider, processing_stale_seconds=0, **QUICK).process_inbound(
        client_id, conversation_id, BOOKING, "stuck-key"
    )
    assert again.outbound.id == result.outbound.id and len(provider.requests) == 1  # now a plain replay
    assert len(fresh(engine, conversation_id)[0]) == 2


def test_a_second_completer_can_never_add_a_second_reply(engine: Engine, db_session: Session) -> None:
    """If another request answers while we were waiting on the provider, our reply is discarded."""
    client_id, conversation_id = stuck_after_inbound(engine, db_session)
    rows, _ = fresh(engine, conversation_id)
    inbound_id = rows[0].id

    winner = service(open_session(engine), RecordingProvider(reply="winner"), processing_stale_seconds=0, **QUICK)
    winner.process_inbound(client_id, conversation_id, BOOKING, "stuck-key")

    # A slow request that already has its own provider answer now tries to store it.
    late_db = open_session(engine)
    if True:
        late = service(late_db, RecordingProvider(reply="late"))
        conversation = late_db.get(Conversation, conversation_id)
        inbound = late_db.get(Message, inbound_id)
        assert conversation is not None and inbound is not None
        outcome = _Outcome("late reply", "late", "late-v1", MessageProcessingStatus.GENERATED)
        result = late._persist_outcome(conversation, inbound, classify_intent(BOOKING), outcome, replay=True)
        assert result.outbound.content != "late reply"  # the winner's reply is returned instead

    rows, conversation_after = fresh(engine, conversation_id)
    assert [m.direction.value for m in rows] == ["inbound", "outbound"]  # still exactly one reply
    assert rows[1].provider == "recording" and conversation_after.last_message_sequence == 2


def test_replays_do_not_require_an_active_client(engine: Engine, db_session: Session) -> None:
    client_id, conversation_id = seed(db_session)
    provider = RecordingProvider()
    first = service(db_session, provider).process_inbound(client_id, conversation_id, BOOKING, "k-9")
    client = db_session.get(Client, client_id)
    assert client is not None
    client.configuration_status = ConfigurationStatus.INACTIVE
    db_session.commit()

    replay = service(db_session, provider).process_inbound(client_id, conversation_id, BOOKING, "k-9")
    assert replay.idempotent_replay and replay.outbound.id == first.outbound.id
    with pytest.raises(ClientNotActiveError):
        service(db_session, provider).process_inbound(client_id, conversation_id, BOOKING, "k-10")


def test_provider_timeout_does_not_hold_a_database_transaction_open(file_engine: Engine) -> None:
    """While the provider is 'thinking', other writers must not be blocked by this request."""
    with Session(file_engine) as setup:
        client_id, conversation_id = seed(setup)
    entered, release = threading.Event(), threading.Event()

    class BlockingProvider(RecordingProvider):
        def generate(self, request: LLMRequest) -> LLMResponse:
            entered.set()
            release.wait(timeout=10)
            raise LLMTimeoutError("simulated timeout")

    factory = sessionmaker(bind=file_engine, autoflush=False, expire_on_commit=False)

    def slow_request() -> None:
        with factory() as db:
            service(db, BlockingProvider()).process_inbound(client_id, conversation_id, BOOKING, "slow")

    thread = threading.Thread(target=slow_request)
    thread.start()
    assert entered.wait(timeout=10)
    started = time.monotonic()
    with factory() as other:  # a write from a different connection while the provider call is pending
        row = other.get(Conversation, conversation_id)
        assert row is not None
        row.customer_id = "cust_changed"
        other.commit()
    assert time.monotonic() - started < 2  # was not blocked by the in-flight request
    release.set()
    thread.join(timeout=15)

    rows, _ = fresh(file_engine, conversation_id)
    assert [m.processing_status for m in rows] == [MessageProcessingStatus.FAILED, MessageProcessingStatus.FALLBACK]


def test_provider_that_fails_is_called_exactly_once_per_new_message(engine: Engine, db_session: Session) -> None:
    client_id, conversation_id = seed(db_session)
    provider = RaisingProvider(LLMTimeoutError("timeout"))
    service(db_session, provider).process_inbound(client_id, conversation_id, BOOKING, "k-11")
    service(db_session, provider).process_inbound(client_id, conversation_id, BOOKING, "k-11")  # replay
    assert provider.calls == 1
