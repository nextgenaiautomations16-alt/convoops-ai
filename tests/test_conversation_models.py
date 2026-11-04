"""Step 3 tests: Conversation and Message database models.

These tests exercise the database layer only (no API, services or LLM).
Constraint tests use raw SQL on purpose: the ORM would reject some bad values
in Python first, and we want proof that the DATABASE refuses them too.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from app.db.database import Base, build_engine
from app.models import (
    Channel,
    Client,
    Conversation,
    ConversationStatus,
    FailureCategory,
    Message,
    MessageDirection,
    MessageProcessingStatus,
    SenderType,
    allocate_message_sequence,
)

TS = "2026-01-01 00:00:00.000000"


# ---- helpers -----------------------------------------------------------------
def make_client(db: Session, slug: str = "demo-clinic") -> Client:
    client = Client(
        business_name=f"Demo Clinic {slug}",
        slug=slug,
        industry="healthcare",
        timezone="America/New_York",
        business_description="Synthetic demo clinic. No real patient data.",
    )
    db.add(client)
    db.commit()
    return client


def make_conversation(db: Session, client: Client, customer_id: str = "cust_1001") -> Conversation:
    conversation = Conversation(client_id=client.id, customer_id=customer_id)
    db.add(conversation)
    db.commit()
    return conversation


def make_message(db: Session, conversation: Conversation, sequence: int, **overrides: Any) -> Message:
    values: dict[str, Any] = {
        "conversation_id": conversation.id,
        "sequence": sequence,
        "sender_type": SenderType.CUSTOMER,
        "direction": MessageDirection.INBOUND,
        "content": "Hello, this is synthetic test text.",
    }
    values.update(overrides)
    message = Message(**values)
    db.add(message)
    db.commit()
    return message


def raw_insert_conversation(db: Session, client_id: uuid.UUID, **overrides: Any) -> None:
    values: dict[str, Any] = {
        "id": uuid.uuid4().hex,  # SQLite stores UUIDs as 32-char hex
        "client_id": client_id.hex,
        "customer_id": "cust_raw",
        "channel": "sms",
        "status": "open",
        "last_message_sequence": 0,
        "created_at": TS,
        "updated_at": TS,
    }
    values.update(overrides)
    db.execute(
        text(
            "INSERT INTO conversations (id, client_id, customer_id, channel, status, "
            "last_message_sequence, created_at, updated_at) VALUES "
            "(:id, :client_id, :customer_id, :channel, :status, :last_message_sequence, "
            ":created_at, :updated_at)"
        ),
        values,
    )


def raw_insert_message(db: Session, conversation_id: uuid.UUID, **overrides: Any) -> None:
    values: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "conversation_id": conversation_id.hex,
        "sequence": 1,
        "sender_type": "customer",
        "direction": "inbound",
        "content": "raw synthetic text",
        "intent": None,
        "intent_confidence": None,
        "processing_status": "received",
        "failure_category": None,
        "idempotency_key": None,
        "in_reply_to_id": None,
        "created_at": TS,
    }
    values.update(overrides)
    db.execute(
        text(
            "INSERT INTO messages (id, conversation_id, sequence, sender_type, direction, "
            "content, intent, intent_confidence, processing_status, failure_category, "
            "idempotency_key, in_reply_to_id, created_at) "
            "VALUES (:id, :conversation_id, :sequence, :sender_type, :direction, :content, "
            ":intent, :intent_confidence, :processing_status, :failure_category, "
            ":idempotency_key, :in_reply_to_id, :created_at)"
        ),
        values,
    )


# ---- 1 & 2. tables exist -------------------------------------------------------
def test_conversations_table_is_created(engine: Engine) -> None:
    inspector = inspect(engine)
    assert "conversations" in inspector.get_table_names()
    columns = {c["name"] for c in inspector.get_columns("conversations")}
    assert columns == {
        "id", "client_id", "customer_id", "channel", "status",
        "last_message_sequence", "created_at", "updated_at", "last_message_at",
    }  # fmt: skip
    index_names = {i["name"] for i in inspector.get_indexes("conversations")}
    assert {"ix_conversations_client_created", "ix_conversations_client_customer"} <= index_names


def test_messages_table_is_created(engine: Engine) -> None:
    inspector = inspect(engine)
    assert "messages" in inspector.get_table_names()
    columns = {c["name"] for c in inspector.get_columns("messages")}
    assert columns == {
        "id", "conversation_id", "sequence", "sender_type", "direction", "content",
        "intent", "intent_confidence", "intent_method", "provider", "model",
        "processing_status", "failure_category", "idempotency_key", "in_reply_to_id", "created_at",
    }  # fmt: skip
    index_names = {i["name"] for i in inspector.get_indexes("messages")}
    assert {"ix_messages_conversation_id", "ix_messages_in_reply_to_id"} <= index_names
    unique_names = {u["name"] for u in inspector.get_unique_constraints("messages")}
    assert {
        "uq_messages_conversation_sequence",
        "uq_messages_conversation_idempotency_key",
    } <= unique_names


def test_phase1_clients_table_is_unchanged(engine: Engine) -> None:
    columns = {c["name"] for c in inspect(engine).get_columns("clients")}
    assert columns == {
        "id", "business_name", "slug", "industry", "timezone", "business_description",
        "contact_email", "configuration_status", "created_at", "updated_at",
    }  # fmt: skip


def test_sqlite_enforces_foreign_keys(db_session: Session) -> None:
    assert db_session.execute(text("PRAGMA foreign_keys")).scalar_one() == 1


# ---- defaults ------------------------------------------------------------------
def test_conversation_and_message_defaults(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    assert isinstance(conversation.id, uuid.UUID)
    assert conversation.channel is Channel.SMS
    assert conversation.status is ConversationStatus.OPEN
    assert conversation.last_message_sequence == 0
    assert conversation.last_message_at is None
    assert conversation.created_at.tzinfo is not None

    message = make_message(db_session, conversation, 1)
    assert message.processing_status is MessageProcessingStatus.RECEIVED
    assert message.intent is None and message.provider is None and message.failure_category is None


def test_raw_insert_helpers_produce_valid_rows(db_session: Session) -> None:
    """Guards the constraint tests below: the baseline raw rows must be valid."""
    client = make_client(db_session)
    conversation_id = uuid.uuid4()
    raw_insert_conversation(db_session, client.id, id=conversation_id.hex)
    raw_insert_message(db_session, conversation_id)
    db_session.commit()
    loaded = db_session.get(Conversation, conversation_id)
    assert loaded is not None and len(loaded.messages) == 1


# ---- 3 & 4. foreign keys -------------------------------------------------------
def test_conversation_requires_existing_client(db_session: Session) -> None:
    with pytest.raises(IntegrityError):
        raw_insert_conversation(db_session, uuid.uuid4())
    db_session.rollback()
    with pytest.raises(IntegrityError):
        db_session.add(Conversation(client_id=uuid.uuid4(), customer_id="cust_x"))
        db_session.commit()


def test_message_requires_existing_conversation(db_session: Session) -> None:
    with pytest.raises(IntegrityError):
        raw_insert_message(db_session, uuid.uuid4())
    db_session.rollback()
    with pytest.raises(IntegrityError):
        db_session.add(
            Message(
                conversation_id=uuid.uuid4(),
                sequence=1,
                sender_type=SenderType.CUSTOMER,
                direction=MessageDirection.INBOUND,
                content="orphan",
            )
        )
        db_session.commit()


def test_client_with_conversations_cannot_be_deleted(db_session: Session) -> None:
    client = make_client(db_session)
    make_conversation(db_session, client)
    with pytest.raises(IntegrityError):
        db_session.execute(text("DELETE FROM clients WHERE id = :id"), {"id": client.id.hex})


def test_conversation_with_messages_cannot_be_deleted(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    make_message(db_session, conversation, 1)
    with pytest.raises(IntegrityError):
        db_session.execute(text("DELETE FROM conversations WHERE id = :id"), {"id": conversation.id.hex})


# ---- 5. unique sequence --------------------------------------------------------
def test_duplicate_sequence_in_same_conversation_is_rejected(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    make_message(db_session, conversation, 1)
    with pytest.raises(IntegrityError):
        make_message(db_session, conversation, 1)


def test_same_sequence_is_allowed_in_different_conversations(db_session: Session) -> None:
    client = make_client(db_session)
    first = make_conversation(db_session, client, "cust_a")
    second = make_conversation(db_session, client, "cust_b")
    make_message(db_session, first, 1)
    make_message(db_session, second, 1)  # must not raise


def test_sequence_must_be_positive(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    with pytest.raises(IntegrityError):
        raw_insert_message(db_session, conversation.id, sequence=0)


# ---- 6-9. enum-like CHECK constraints (database level) ---------------------------
@pytest.mark.parametrize(
    "column, bad_value",
    [("channel", "fax"), ("status", "pending")],
)
def test_conversation_check_constraints_reject_bad_values(
    db_session: Session, column: str, bad_value: str
) -> None:
    client = make_client(db_session)
    with pytest.raises(IntegrityError):
        raw_insert_conversation(db_session, client.id, **{column: bad_value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"sender_type": "robot", "direction": "outbound"},
        {"sender_type": "customer", "direction": "sideways"},
        {"processing_status": "maybe"},
        {"processing_status": "failed", "failure_category": "cosmic_rays"},
    ],
)
def test_message_check_constraints_reject_bad_values(
    db_session: Session, overrides: dict[str, Any]
) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    with pytest.raises(IntegrityError):
        raw_insert_message(db_session, conversation.id, **overrides)


# ...and the ORM refuses them earlier, in Python.
@pytest.mark.parametrize(
    "factory",
    [
        lambda cid: Conversation(client_id=cid, customer_id="c", channel="fax"),
        lambda cid: Conversation(client_id=cid, customer_id="c", status="pending"),
    ],
)
def test_orm_rejects_invalid_conversation_enum_values(db_session: Session, factory: Any) -> None:
    client = make_client(db_session)
    with pytest.raises((LookupError, StatementError)):
        db_session.add(factory(client.id))
        db_session.commit()


def test_orm_rejects_invalid_message_enum_values(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    for field, bad in [("sender_type", "robot"), ("direction", "sideways")]:
        db_session.rollback()
        with pytest.raises((LookupError, StatementError)):
            make_message(db_session, conversation, 1, **{field: bad})


def test_sender_type_must_match_direction(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    with pytest.raises(IntegrityError):
        raw_insert_message(db_session, conversation.id, sender_type="customer", direction="outbound")
    db_session.rollback()
    with pytest.raises(IntegrityError):
        raw_insert_message(db_session, conversation.id, sender_type="ai", direction="inbound")


# ---- 10. blank content ---------------------------------------------------------
# Known limit: SQL trim() removes only SPACES (SQLite and PostgreSQL alike), so the
# database CHECK blocks "" and "   " but not tab/newline-only text. The Step 5 request
# schema must strip all whitespace before storing. Phase 1's Client CHECKs share this limit.
@pytest.mark.parametrize("content", ["", " ", "     "])
def test_blank_message_content_is_rejected(db_session: Session, content: str) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    with pytest.raises(IntegrityError):
        raw_insert_message(db_session, conversation.id, content=content)


def test_blank_customer_id_is_rejected(db_session: Session) -> None:
    client = make_client(db_session)
    with pytest.raises(IntegrityError):
        raw_insert_conversation(db_session, client.id, customer_id="   ")


# ---- 11. intent confidence range -----------------------------------------------
@pytest.mark.parametrize("confidence", [-0.01, 1.01, 5.0])
def test_out_of_range_intent_confidence_is_rejected(db_session: Session, confidence: float) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    with pytest.raises(IntegrityError):
        raw_insert_message(db_session, conversation.id, intent="unknown", intent_confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0, None])
def test_valid_intent_confidence_is_accepted(db_session: Session, confidence: float | None) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    message = make_message(
        db_session, conversation, 1,
        intent="pricing_question", intent_confidence=confidence, intent_method="rule_based_v1",
    )  # fmt: skip
    assert message.intent_confidence == confidence


# ---- failure tracking ----------------------------------------------------------
def test_provider_failure_can_be_recorded_on_a_fallback_reply(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    reply = make_message(
        db_session, conversation, 1,
        sender_type=SenderType.SYSTEM, direction=MessageDirection.OUTBOUND,
        content="Sorry, a team member will follow up.",
        provider="fallback", processing_status=MessageProcessingStatus.FALLBACK,
        failure_category=FailureCategory.PROVIDER_TIMEOUT,
    )  # fmt: skip
    assert reply.failure_category is FailureCategory.PROVIDER_TIMEOUT


def test_failure_category_requires_failed_or_fallback_status(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    with pytest.raises(IntegrityError):
        raw_insert_message(
            db_session, conversation.id, processing_status="processed", failure_category="provider_error"
        )


# ---- 12. relationships ---------------------------------------------------------
def test_relationships_between_client_conversation_and_message(db_session: Session) -> None:
    client = make_client(db_session)
    conversation = make_conversation(db_session, client)
    make_message(db_session, conversation, 2, content="second")  # inserted out of order on purpose
    make_message(db_session, conversation, 1, content="first")
    db_session.expire_all()

    assert [c.id for c in client.conversations] == [conversation.id]
    assert conversation.client is client
    assert [m.content for m in conversation.messages] == ["first", "second"]  # ordered by sequence
    assert conversation.messages[0].conversation is conversation


def test_conversations_of_different_clients_stay_separate(db_session: Session) -> None:
    client_a = make_client(db_session, "clinic-a")
    client_b = make_client(db_session, "clinic-b")
    conv_a = make_conversation(db_session, client_a, "cust_1")
    conv_b = make_conversation(db_session, client_b, "cust_1")  # same customer id, other tenant
    db_session.expire_all()
    assert [c.id for c in client_a.conversations] == [conv_a.id]
    assert [c.id for c in client_b.conversations] == [conv_b.id]


# ---- sequence allocation -------------------------------------------------------
def test_allocate_message_sequence_counts_up_from_one(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    assert [allocate_message_sequence(db_session, conversation.id) for _ in range(3)] == [1, 2, 3]
    db_session.commit()
    db_session.refresh(conversation)
    assert conversation.last_message_sequence == 3


def test_allocate_message_sequence_is_per_conversation(db_session: Session) -> None:
    client = make_client(db_session)
    first = make_conversation(db_session, client, "cust_a")
    second = make_conversation(db_session, client, "cust_b")
    assert allocate_message_sequence(db_session, first.id) == 1
    assert allocate_message_sequence(db_session, first.id) == 2
    assert allocate_message_sequence(db_session, second.id) == 1


def test_allocate_message_sequence_unknown_conversation(db_session: Session) -> None:
    with pytest.raises(LookupError):
        allocate_message_sequence(db_session, uuid.uuid4())


def test_rolled_back_allocation_leaves_no_gap(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    assert allocate_message_sequence(db_session, conversation.id) == 1
    db_session.rollback()  # e.g. the message insert failed
    assert allocate_message_sequence(db_session, conversation.id) == 1


def test_concurrent_allocation_never_collides(tmp_path: Any) -> None:
    """8 threads x 5 messages against a real SQLite *file* (separate connections)."""
    engine = build_engine(f"sqlite:///{tmp_path / 'concurrency.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as setup:
        conversation = make_conversation(setup, make_client(setup))
        conversation_id = conversation.id

    threads, per_thread = 8, 5

    def worker(_: int) -> list[int]:
        sequences: list[int] = []
        for _ in range(per_thread):
            with Session(engine) as db:  # own session + own connection = own transaction
                sequence = allocate_message_sequence(db, conversation_id)
                db.add(
                    Message(
                        conversation_id=conversation_id,
                        sequence=sequence,
                        sender_type=SenderType.CUSTOMER,
                        direction=MessageDirection.INBOUND,
                        content="synthetic concurrent message",
                    )
                )
                db.commit()
            sequences.append(sequence)
        return sequences

    with ThreadPoolExecutor(max_workers=threads) as pool:
        handed_out = [seq for batch in pool.map(worker, range(threads)) for seq in batch]

    total = threads * per_thread
    assert sorted(handed_out) == list(range(1, total + 1))  # no duplicates, no gaps
    with Session(engine) as check:
        stored = check.execute(text("SELECT sequence FROM messages ORDER BY sequence")).scalars().all()
        assert stored == list(range(1, total + 1))
        assert check.get(Conversation, conversation_id).last_message_sequence == total  # type: ignore[union-attr]
    engine.dispose()


# ---- idempotency support (schema level; replay logic itself arrives in Step 4) ----------
def test_duplicate_idempotency_key_in_same_conversation_is_rejected(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    make_message(db_session, conversation, 1, idempotency_key="retry-abc")
    with pytest.raises(IntegrityError):
        make_message(db_session, conversation, 2, idempotency_key="retry-abc")


def test_same_idempotency_key_is_allowed_in_different_conversations(db_session: Session) -> None:
    client = make_client(db_session)
    first = make_conversation(db_session, client, "cust_a")
    second = make_conversation(db_session, client, "cust_b")
    make_message(db_session, first, 1, idempotency_key="retry-abc")
    make_message(db_session, second, 1, idempotency_key="retry-abc")  # must not raise


def test_messages_without_idempotency_key_never_conflict(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    for sequence in (1, 2, 3):
        make_message(db_session, conversation, sequence)  # all NULL keys: must not raise
    db_session.refresh(conversation)
    assert [m.idempotency_key for m in conversation.messages] == [None, None, None]


def test_idempotency_key_can_be_looked_up_by_conversation_and_key(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    stored = make_message(db_session, conversation, 1, idempotency_key="retry-abc", content="original text")
    found = db_session.execute(
        select(Message).where(
            Message.conversation_id == conversation.id, Message.idempotency_key == "retry-abc"
        )
    ).scalar_one()
    assert found.id == stored.id and found.content == "original text"


@pytest.mark.parametrize("key", ["", " ", "     "])
def test_blank_idempotency_key_is_rejected(db_session: Session, key: str) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    with pytest.raises(IntegrityError):
        raw_insert_message(db_session, conversation.id, idempotency_key=key)


def test_idempotency_key_is_only_allowed_on_inbound_messages(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    with pytest.raises(IntegrityError):
        raw_insert_message(
            db_session, conversation.id, sender_type="ai", direction="outbound", idempotency_key="k"
        )


def test_idempotency_key_longer_than_128_chars_is_a_schema_limit() -> None:
    assert Message.__table__.c.idempotency_key.type.length == 128


# ---- in_reply_to_id ----------------------------------------------------------------
def test_reply_links_to_the_inbound_message_it_answers(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    inbound = make_message(db_session, conversation, 1, idempotency_key="k1")
    reply = make_message(
        db_session, conversation, 2,
        sender_type=SenderType.AI, direction=MessageDirection.OUTBOUND,
        content="synthetic reply", in_reply_to_id=inbound.id,
        processing_status=MessageProcessingStatus.GENERATED, provider="mock", model="mock-v1",
    )  # fmt: skip
    db_session.expire_all()
    assert reply.in_reply_to.id == inbound.id
    assert [r.id for r in inbound.replies] == [reply.id]
    assert inbound.in_reply_to is None


def test_reply_link_must_point_to_an_existing_message(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    with pytest.raises(IntegrityError):
        raw_insert_message(
            db_session, conversation.id, sender_type="ai", direction="outbound",
            in_reply_to_id=uuid.uuid4().hex,
        )  # fmt: skip


def test_reply_link_is_only_allowed_on_outbound_messages(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    inbound = make_message(db_session, conversation, 1)
    with pytest.raises(IntegrityError):
        raw_insert_message(db_session, conversation.id, sequence=2, in_reply_to_id=inbound.id.hex)


def test_replied_to_message_cannot_be_deleted(db_session: Session) -> None:
    conversation = make_conversation(db_session, make_client(db_session))
    inbound = make_message(db_session, conversation, 1)
    make_message(
        db_session, conversation, 2, sender_type=SenderType.AI, direction=MessageDirection.OUTBOUND,
        in_reply_to_id=inbound.id,
    )  # fmt: skip
    with pytest.raises(IntegrityError):
        db_session.execute(text("DELETE FROM messages WHERE id = :id"), {"id": inbound.id.hex})


# ---- concurrent retries with the same key (file-based SQLite, real separate connections) --
def _file_engine_with_conversation(tmp_path: Any) -> tuple[Engine, uuid.UUID]:
    engine = build_engine(f"sqlite:///{tmp_path / 'idempotency.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as setup:
        conversation = make_conversation(setup, make_client(setup))
        return engine, conversation.id


def _try_store_inbound(engine: Engine, conversation_id: uuid.UUID, key: str) -> str:
    """Mimics what the Step 4 service will do: reserve a sequence and insert in ONE transaction."""
    with Session(engine) as db:
        try:
            sequence = allocate_message_sequence(db, conversation_id)
            db.add(
                Message(
                    conversation_id=conversation_id,
                    sequence=sequence,
                    sender_type=SenderType.CUSTOMER,
                    direction=MessageDirection.INBOUND,
                    content="synthetic retried message",
                    idempotency_key=key,
                )
            )
            db.commit()
            return "stored"
        except IntegrityError:
            db.rollback()  # also rolls back the sequence increment
            return "duplicate"


def test_concurrent_requests_with_same_key_store_exactly_one_message(tmp_path: Any) -> None:
    engine, conversation_id = _file_engine_with_conversation(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: _try_store_inbound(engine, conversation_id, "same-key"), range(8)))

    assert sorted(outcomes) == ["duplicate"] * 7 + ["stored"]
    with Session(engine) as check:
        rows = check.execute(
            text("SELECT sequence FROM messages WHERE idempotency_key = 'same-key'")
        ).scalars().all()
        assert rows == [1]
        # The 7 losing transactions rolled back their increments: no sequence gaps.
        assert check.get(Conversation, conversation_id).last_message_sequence == 1  # type: ignore[union-attr]
    engine.dispose()


def test_concurrent_requests_with_different_keys_all_succeed(tmp_path: Any) -> None:
    engine, conversation_id = _file_engine_with_conversation(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(
            pool.map(lambda n: _try_store_inbound(engine, conversation_id, f"key-{n}"), range(8))
        )

    assert outcomes == ["stored"] * 8
    with Session(engine) as check:
        sequences = check.execute(text("SELECT sequence FROM messages ORDER BY sequence")).scalars().all()
        assert sequences == list(range(1, 9))
    engine.dispose()
