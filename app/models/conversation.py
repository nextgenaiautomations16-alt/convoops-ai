"""Conversation and Message ORM models (Phase 2).

Ownership chain:  Client 1 --- * Conversation 1 --- * Message

Every conversation belongs to exactly one client and every message to exactly
one conversation, enforced by foreign keys (see `configure_sqlite` in
app/db/database.py - SQLite needs a PRAGMA to enforce them).

Safe retries (database support only; the logic is implemented in the Step 4 service):
  * `uq_messages_conversation_idempotency_key` lets the database admit exactly one
    inbound message per (conversation, key), even when identical requests race.
  * The service looks the key up first. Same key + same content -> return the original
    result (the inbound message plus the reply found through `in_reply_to_id`).
    Same key + different content -> conflict error.
  * A request that loses a race hits the unique constraint, rolls back (which also rolls
    back its sequence increment) and then re-reads the winner's row as a normal replay.
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    update,
)
from sqlalchemy.orm import Mapped, Session, backref, mapped_column, relationship

from app.db.database import Base, UTCDateTime
from app.models.client import Client


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _text_enum(enum_cls: type[enum.Enum], name: str, length: int = 32) -> Enum:
    """Enum stored as plain text + a named CHECK constraint (portable, same as Client)."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        length=length,
        create_constraint=True,
        values_callable=lambda cls: [member.value for member in cls],
    )


class Channel(str, enum.Enum):
    SMS = "sms"


class ConversationStatus(str, enum.Enum):
    OPEN = "open"
    AWAITING_RESPONSE = "awaiting_response"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SenderType(str, enum.Enum):
    CUSTOMER = "customer"
    AI = "ai"
    SYSTEM = "system"  # platform-written messages, e.g. a safe fallback or handoff notice


class MessageDirection(str, enum.Enum):
    INBOUND = "inbound"  # customer -> platform
    OUTBOUND = "outbound"  # platform -> customer


class MessageProcessingStatus(str, enum.Enum):
    RECEIVED = "received"  # inbound message stored, not processed yet
    PROCESSED = "processed"  # inbound message fully processed
    FAILED = "failed"  # inbound message processing failed
    GENERATED = "generated"  # outbound reply produced by a provider
    FALLBACK = "fallback"  # outbound reply is a deterministic safe fallback


class FailureCategory(str, enum.Enum):
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_CONNECTION = "provider_connection"
    PROVIDER_ERROR = "provider_error"
    INTERNAL_ERROR = "internal_error"


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint("length(trim(customer_id)) > 0", name="ck_conversations_customer_id_not_blank"),
        CheckConstraint("last_message_sequence >= 0", name="ck_conversations_last_sequence_non_negative"),
        # "list a client's conversations, newest first"
        Index("ix_conversations_client_created", "client_id", "created_at"),
        # "find a customer's conversations within one client"
        Index("ix_conversations_client_customer", "client_id", "customer_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    client_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("clients.id", ondelete="RESTRICT"), nullable=False
    )
    customer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    channel: Mapped[Channel] = mapped_column(
        _text_enum(Channel, "conversation_channel"),
        nullable=False,
        default=Channel.SMS,
        server_default=Channel.SMS.value,
    )
    status: Mapped[ConversationStatus] = mapped_column(
        _text_enum(ConversationStatus, "conversation_status"),
        nullable=False,
        default=ConversationStatus.OPEN,
        server_default=ConversationStatus.OPEN.value,
    )
    # Per-conversation message counter used to hand out message sequence numbers.
    # Only change it through `allocate_message_sequence` (atomic UPDATE ... RETURNING).
    last_message_sequence: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )
    last_message_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    # `backref` adds `Client.conversations` without editing the Phase 1 Client model.
    client: Mapped[Client] = relationship(
        backref=backref("conversations", order_by="Conversation.created_at")
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", order_by="Message.sequence"
    )

    def __repr__(self) -> str:
        return f"Conversation(id={self.id}, client_id={self.client_id}, status={self.status.value})"


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "sequence", name="uq_messages_conversation_sequence"),
        # Idempotency: one key per conversation. NULL keys never conflict (NULLs are
        # distinct in SQLite and PostgreSQL), so messages without a key are unaffected.
        UniqueConstraint(
            "conversation_id", "idempotency_key", name="uq_messages_conversation_idempotency_key"
        ),
        CheckConstraint(
            "idempotency_key IS NULL OR length(trim(idempotency_key)) > 0",
            name="ck_messages_idempotency_key_not_blank",
        ),
        # Only customer (inbound) requests are retried, so only they carry a key.
        CheckConstraint(
            "idempotency_key IS NULL OR direction = 'inbound'",
            name="ck_messages_idempotency_key_inbound_only",
        ),
        # A reply link points from an outbound message back to the inbound one it answers.
        CheckConstraint(
            "in_reply_to_id IS NULL OR direction = 'outbound'",
            name="ck_messages_reply_link_outbound_only",
        ),
        CheckConstraint("sequence >= 1", name="ck_messages_sequence_positive"),
        CheckConstraint("length(trim(content)) > 0", name="ck_messages_content_not_blank"),
        CheckConstraint(
            "intent_confidence IS NULL OR (intent_confidence >= 0 AND intent_confidence <= 1)",
            name="ck_messages_intent_confidence_range",
        ),
        # Customers send inbound messages; the AI and the platform send outbound ones.
        CheckConstraint(
            "(sender_type = 'customer' AND direction = 'inbound') "
            "OR (sender_type IN ('ai', 'system') AND direction = 'outbound')",
            name="ck_messages_sender_matches_direction",
        ),
        # A failure category only makes sense on a failed or fallback row.
        CheckConstraint(
            "failure_category IS NULL OR processing_status IN ('failed', 'fallback')",
            name="ck_messages_failure_category_requires_failure",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("conversations.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    sender_type: Mapped[SenderType] = mapped_column(
        _text_enum(SenderType, "message_sender_type"), nullable=False
    )
    direction: Mapped[MessageDirection] = mapped_column(
        _text_enum(MessageDirection, "message_direction"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    intent_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    intent_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    processing_status: Mapped[MessageProcessingStatus] = mapped_column(
        _text_enum(MessageProcessingStatus, "message_processing_status"),
        nullable=False,
        default=MessageProcessingStatus.RECEIVED,
        server_default=MessageProcessingStatus.RECEIVED.value,
    )
    failure_category: Mapped[FailureCategory | None] = mapped_column(
        _text_enum(FailureCategory, "message_failure_category"), nullable=True
    )
    # Client-supplied retry token for inbound messages (see module notes below).
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Links an outbound reply to the inbound message it answers. Lets a retried
    # request find its original reply even when other messages interleave.
    in_reply_to_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("messages.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    in_reply_to: Mapped["Message | None"] = relationship(
        back_populates="replies", remote_side="Message.id", foreign_keys="Message.in_reply_to_id"
    )
    replies: Mapped[list["Message"]] = relationship(
        back_populates="in_reply_to", foreign_keys="Message.in_reply_to_id"
    )

    def __repr__(self) -> str:
        return f"Message(id={self.id}, conversation_id={self.conversation_id}, sequence={self.sequence})"


def allocate_message_sequence(db: Session, conversation_id: uuid.UUID) -> int:
    """Atomically reserve the next message sequence number for a conversation.

    Runs `UPDATE conversations SET last_message_sequence = last_message_sequence + 1
    WHERE id = :id RETURNING last_message_sequence` as ONE statement.

    - The increment and the read happen together inside the database, so two
      requests can never be handed the same number.
    - The UPDATE takes the write lock. A concurrent request waits until the first
      transaction commits or rolls back, then gets the next number.
    - The caller must insert the Message with this number in the SAME transaction.
      If that transaction rolls back, the increment rolls back too (no gaps).
    - `uq_messages_conversation_sequence` is the final safety net.

    Raises LookupError if the conversation does not exist. Callers must already have
    verified tenant ownership (conversation_id AND client_id) before calling this.
    """
    stmt = (
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(last_message_sequence=Conversation.last_message_sequence + 1)
        .returning(Conversation.last_message_sequence)
        .execution_options(synchronize_session="fetch")
    )
    sequence = db.execute(stmt).scalar_one_or_none()
    if sequence is None:
        raise LookupError(f"Conversation {conversation_id} does not exist.")
    return sequence
