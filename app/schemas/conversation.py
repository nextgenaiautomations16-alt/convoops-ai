"""Pydantic request/response schemas for conversations and their messages."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.conversation import (
    Channel,
    ConversationStatus,
    FailureCategory,
    MessageDirection,
    MessageProcessingStatus,
    SenderType,
)

# Opaque identifier chosen by the client, e.g. "cust_1001". Deliberately narrow:
# no spaces, '@' or '+', so emails and phone numbers are not accepted by accident.
CUSTOMER_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"


class ConversationCreate(BaseModel):
    """Body for starting a conversation. New conversations always start as `open`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    customer_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=CUSTOMER_ID_PATTERN,
        description="Opaque customer identifier, e.g. 'cust_1001'. Letters, digits and _ . : - only.",
    )
    channel: Channel = Field(default=Channel.SMS, description="Communication channel. Only 'sms' for now.")


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    client_id: uuid.UUID
    customer_id: str
    channel: Channel
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None


class ConversationList(BaseModel):
    items: list[ConversationRead]
    total: int
    limit: int
    offset: int


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    sequence: int
    sender_type: SenderType
    direction: MessageDirection
    content: str
    intent: str | None
    intent_confidence: float | None
    intent_method: str | None
    provider: str | None
    model: str | None
    processing_status: MessageProcessingStatus
    failure_category: FailureCategory | None
    in_reply_to_id: uuid.UUID | None
    idempotency_key: str | None
    created_at: datetime


class MessageList(BaseModel):
    items: list[MessageRead]
    total: int
    limit: int
    offset: int
