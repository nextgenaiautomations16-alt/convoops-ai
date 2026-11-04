"""Request/response schemas for submitting an inbound customer message."""

import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings
from app.models.conversation import ConversationStatus, FailureCategory, MessageProcessingStatus
from app.schemas.conversation import MessageRead

# Same shape rule as customer IDs: opaque, no spaces.
IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"


class MessageCreate(BaseModel):
    """An inbound customer message. Leading/trailing whitespace is removed, so
    whitespace-only content (including tabs and newlines) is rejected as empty."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "content": "Hello, I would like to book an appointment.",
                    "idempotency_key": "customer-message-1001",
                }
            ]
        },
    )

    content: str = Field(
        min_length=1,
        max_length=get_settings().max_message_length,
        description="The customer's message (synthetic data only in this demo).",
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=IDEMPOTENCY_KEY_PATTERN,
        description=(
            "Optional retry token, unique per conversation. Re-sending the same key with the same "
            "content returns the original result; the same key with different content returns 409."
        ),
    )

    @field_validator("content")
    @classmethod
    def _no_nul_characters(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("must not contain NUL characters")
        return value


class IntentInfo(BaseModel):
    intent: str = Field(description="One of the supported intents, e.g. 'pricing'.")
    confidence: float = Field(description="Heuristic score from 0 to 1. NOT a calibrated probability.")
    method: str = Field(description="How the intent was determined, e.g. 'deterministic_rules_v1'.")


class MessageProcessingResponse(BaseModel):
    conversation_id: uuid.UUID
    conversation_status: ConversationStatus
    idempotent_replay: bool = Field(
        description="True when this request repeated an earlier one and no new inbound message was created."
    )
    processing_status: MessageProcessingStatus = Field(
        description="Status of the inbound message: 'processed', or 'failed' when a fallback reply was used."
    )
    reply_type: MessageProcessingStatus = Field(
        description="'generated' for a provider reply, 'fallback' for the safe deterministic reply."
    )
    failure_category: FailureCategory | None = Field(
        description="Why the fallback was used, if it was; otherwise null."
    )
    intent: IntentInfo
    provider: str | None = Field(description="Provider that produced the reply ('fallback' for the safe reply).")
    model: str | None
    inbound_message: MessageRead
    outbound_message: MessageRead
