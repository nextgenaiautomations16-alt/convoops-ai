"""Synchronous inbound-message processing pipeline.

    validate tenant -> (idempotent replay?) -> store inbound [tx 1] -> load history
    -> classify intent -> call provider -> store reply + update conversation [tx 2]

TRANSACTIONS
  tx 1 stores the inbound message and commits BEFORE the provider is called, so the
  customer's message is never lost. No database transaction is held open during the
  provider call. tx 2 stores the reply, the intent fields and the conversation update
  together: all of it or none of it.

IDEMPOTENCY (see also the notes in app/models/conversation.py)
  * The database constraint uq_messages_conversation_idempotency_key admits exactly one
    inbound message per (conversation, key). A request that loses a race hits the
    constraint, rolls back, and is handled as a replay of the winner. A pre-check exists
    only to avoid needless work; correctness never depends on it.
  * Only the request that created the inbound message normally writes the reply. As a
    second line of defence, tx 2 reserves a sequence number first (which takes the write
    lock on the conversation) and then checks that no reply to this inbound message exists
    yet, so even a resumed request cannot create a second reply.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import ConflictError
from app.models.client import Client, ConfigurationStatus
from app.models.conversation import (
    Conversation,
    ConversationStatus,
    FailureCategory,
    Message,
    MessageDirection,
    MessageProcessingStatus,
    SenderType,
    allocate_message_sequence,
)
from app.providers.base import (
    ChatMessage,
    LLMConnectionError,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)
from app.services.conversation_service import ClientNotActiveError, ConversationService
from app.services.conversation_status import InvalidStatusTransitionError, apply_status
from app.services.intent_classifier import (
    CLASSIFICATION_METHOD,
    Intent,
    IntentResult,
    classify_intent,
    detect_medical_topic,
)

logger = logging.getLogger(__name__)

FALLBACK_PROVIDER = "fallback"
FALLBACK_MODEL = "deterministic-fallback-v1"
MAX_REPLY_LENGTH = 1600
_POLL_INTERVAL_SECONDS = 0.05

FALLBACK_REPLY = (
    "Thanks for your message to {business}. We couldn't prepare a reply just now, but your message has been "
    "saved so the clinic staff can follow up. If this is urgent, please contact the clinic directly."
)


class IdempotencyConflictError(ConflictError):
    """The idempotency key was already used with different message content."""

    code = "idempotency_key_conflict"


class ConversationClosedError(ConflictError):
    """Messages cannot be added to a closed conversation."""

    code = "conversation_closed"


class ProcessingInProgressError(ConflictError):
    """A request with this idempotency key is still being processed. Retrying later is safe."""

    code = "processing_in_progress"


@dataclass(frozen=True)
class ProcessingResult:
    conversation: Conversation
    inbound: Message
    outbound: Message
    intent: IntentResult
    idempotent_replay: bool


@dataclass(frozen=True)
class _Outcome:
    """What the provider step produced: a real reply or a safe fallback."""

    content: str
    provider: str
    model: str
    status: MessageProcessingStatus  # GENERATED or FALLBACK
    failure_category: FailureCategory | None = None
    attempted_provider: str | None = None  # provider that failed (fallback only)
    attempted_model: str | None = None


def build_system_prompt(client: Client) -> str:
    """Instructions sent to every provider. Real LLMs rely on these rules; the mock follows them by design."""
    return (
        f"You are the text-message assistant for {client.business_name} ({client.industry}). "
        f"About the business: {client.business_description}\n"
        "Rules:\n"
        "1. Reply briefly, in plain text suitable for SMS.\n"
        "2. Never give medical advice, diagnoses or treatment suggestions. Direct medical questions to "
        "qualified clinic staff, and tell people to call their local emergency number in an emergency.\n"
        "3. Never invent prices, appointment availability, policies or patient records.\n"
        "4. Never claim that a booking, change or cancellation is done, or that a staff member has been "
        "notified. Say that clinic staff will follow up.\n"
        "5. Use only the information in this conversation.\n"
        "6. Write one short, professional message that is ready to send to the customer: at most three short "
        "sentences, no greeting line such as a subject, and no sign-off block.\n"
        "7. Never use placeholders such as [Customer Name] or {{name}}. If you do not know a detail, leave it out.\n"
        "8. Output only the message text itself: no markdown, bullet points, headings, quotation marks around the "
        "message, or explanation of your reasoning.\n"
        "9. Never invent business policies, opening hours, addresses, phone numbers, prices or appointment times.\n"
        "10. Never state that an action has been completed, or that staff have been told, unless the conversation "
        "shows it. When information is unavailable, say that clinic staff will follow up."
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MessageProcessingService:
    """Processes one inbound customer message end to end."""

    def __init__(self, db: Session, provider: LLMProvider, settings: Settings | None = None) -> None:
        self.db = db
        self.provider = provider
        self.settings = settings or get_settings()
        self.conversations = ConversationService(db)

    # ---- public API ----------------------------------------------------------
    def process_inbound(
        self,
        client_id: uuid.UUID,
        conversation_id: uuid.UUID,
        content: str,
        idempotency_key: str | None = None,
    ) -> ProcessingResult:
        """Store the customer's message, generate a reply, and return both.

        Raises NotFoundError (unknown client, or conversation not owned by the client),
        ClientNotActiveError, ConversationClosedError, IdempotencyConflictError and
        ProcessingInProgressError. Provider failures never raise: they produce a fallback reply.
        """
        started = time.perf_counter()
        # Tenant gate: filters on client_id AND conversation_id. Raises 404 otherwise.
        conversation = self.conversations.get(client_id, conversation_id)
        client = self.db.get(Client, client_id)
        assert client is not None  # the line above proved the client exists

        if idempotency_key is not None:
            existing = self._find_by_key(conversation.id, idempotency_key)
            if existing is not None:
                return self._handle_duplicate(client, conversation, existing, content)

        self._ensure_accepting_messages(client, conversation)
        try:
            inbound = self._store_inbound(conversation, content, idempotency_key)
        except IntegrityError:
            # Lost a race on the idempotency constraint (the transaction is already rolled back).
            existing = self._find_by_key(conversation.id, idempotency_key) if idempotency_key else None
            if existing is None:
                raise
            return self._handle_duplicate(client, conversation, existing, content)

        result = self._complete_processing(client, conversation, inbound, replay=False)
        self._log_result(result, content_chars=len(content), started=started)
        return result

    # ---- validation ----------------------------------------------------------------
    def _ensure_accepting_messages(self, client: Client, conversation: Conversation) -> None:
        if client.configuration_status is not ConfigurationStatus.ACTIVE:
            raise ClientNotActiveError(
                f"Client {client.id} is '{client.configuration_status.value}'; "
                "messages can only be processed for active clients."
            )
        if conversation.status is ConversationStatus.CLOSED:
            raise ConversationClosedError(f"Conversation {conversation.id} is closed and cannot accept messages.")

    # ---- transaction 1: store the inbound message -------------------------------------
    def _store_inbound(self, conversation: Conversation, content: str, key: str | None) -> Message:
        now = _utcnow()
        try:
            # The sequence reservation and the insert share ONE transaction.
            sequence = allocate_message_sequence(self.db, conversation.id)
            inbound = Message(
                conversation_id=conversation.id,
                sequence=sequence,
                sender_type=SenderType.CUSTOMER,
                direction=MessageDirection.INBOUND,
                content=content,
                idempotency_key=key,
                processing_status=MessageProcessingStatus.RECEIVED,
                created_at=now,
            )
            self.db.add(inbound)
            apply_status(conversation, ConversationStatus.OPEN)  # a customer message reopens resolved/awaiting
            conversation.last_message_at = now
            self.db.commit()
        except SQLAlchemyError:
            self.db.rollback()  # also rolls back the sequence increment
            raise
        return inbound

    # ---- idempotent replays -----------------------------------------------------------------
    def _find_by_key(self, conversation_id: uuid.UUID, key: str) -> Message | None:
        return self.db.scalar(
            select(Message).where(Message.conversation_id == conversation_id, Message.idempotency_key == key)
        )

    def _find_reply(self, inbound_id: uuid.UUID) -> Message | None:
        return self.db.scalar(
            select(Message).where(Message.in_reply_to_id == inbound_id).order_by(Message.sequence).limit(1)
        )

    def _handle_duplicate(
        self, client: Client, conversation: Conversation, existing: Message, content: str
    ) -> ProcessingResult:
        if existing.content != content:
            raise IdempotencyConflictError("This idempotency key was already used with different message content.")
        reply = self._find_reply(existing.id) or self._wait_for_reply(existing.id)
        if reply is not None:
            return self._result(conversation, existing, reply, replay=True)
        if self._is_abandoned(existing):
            logger.warning("processing_resumed conversation_id=%s inbound_id=%s", conversation.id, existing.id)
            return self._complete_processing(client, conversation, existing, replay=True)
        raise ProcessingInProgressError(
            "A request with this idempotency key is still being processed. Retry shortly."
        )

    def _wait_for_reply(self, inbound_id: uuid.UUID) -> Message | None:
        """Briefly poll for the reply of a request that is still in flight (bounded wait)."""
        deadline = time.monotonic() + self.settings.idempotency_wait_seconds
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL_SECONDS)
            self.db.rollback()  # end the read transaction so the next read sees fresh data
            reply = self._find_reply(inbound_id)
            if reply is not None:
                return reply
        return None

    def _is_abandoned(self, inbound: Message) -> bool:
        age = (_utcnow() - inbound.created_at).total_seconds()
        return age >= self.settings.processing_stale_seconds

    # ---- provider step ----------------------------------------------------------------------
    def _complete_processing(
        self, client: Client, conversation: Conversation, inbound: Message, *, replay: bool
    ) -> ProcessingResult:
        history = self._load_history(conversation.id)
        intent = classify_intent(inbound.content)
        request = LLMRequest(
            system_prompt=build_system_prompt(client),
            messages=history,
            intent=intent.intent.value,
            intent_confidence=intent.confidence,
            business_name=client.business_name,
            timeout_seconds=self.settings.llm_timeout_seconds,
            medical_topic=detect_medical_topic(inbound.content),
        )
        outcome = self._generate(client, request)
        try:
            return self._persist_outcome(conversation, inbound, intent, outcome, replay=replay)
        except SQLAlchemyError as exc:
            # tx 2 failed and was rolled back. The inbound message is safe (tx 1). Try once more with
            # the safe fallback so the customer still gets a stored answer.
            logger.error(
                "outcome_persist_failed conversation_id=%s inbound_id=%s error_type=%s",
                conversation.id,
                inbound.id,
                type(exc).__name__,
            )
            fallback = self._fallback(client, FailureCategory.INTERNAL_ERROR, outcome.provider, outcome.model)
            return self._persist_outcome(conversation, inbound, intent, fallback, replay=replay)

    def _load_history(self, conversation_id: uuid.UUID) -> tuple[ChatMessage, ...]:
        rows = self.db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence.desc())
            .limit(self.settings.history_window)
        ).all()
        return tuple(
            ChatMessage(
                role="customer" if m.direction is MessageDirection.INBOUND else "assistant", content=m.content
            )
            for m in reversed(rows)
        )

    def _generate(self, client: Client, request: LLMRequest) -> _Outcome:
        """Call the provider. Any failure or unusable answer becomes a safe fallback (never raises)."""
        category: FailureCategory
        try:
            response = self.provider.generate(request)
        except LLMTimeoutError:
            category = FailureCategory.PROVIDER_TIMEOUT
        except LLMConnectionError:
            category = FailureCategory.PROVIDER_CONNECTION
        except LLMError:
            category = FailureCategory.PROVIDER_ERROR
        except Exception as exc:  # noqa: BLE001 - a provider bug must never break the customer flow
            logger.error("provider_unexpected_error provider=%s error_type=%s", self.provider.name, type(exc).__name__)
            category = FailureCategory.PROVIDER_ERROR
        else:
            problem = _validate_response(response)
            if problem is None:
                return _Outcome(
                    content=response.content.strip(),
                    provider=response.provider,
                    model=response.model,
                    status=MessageProcessingStatus.GENERATED,
                )
            logger.warning("provider_invalid_output provider=%s reason=%s", self.provider.name, problem)
            category = FailureCategory.PROVIDER_ERROR
        logger.warning(
            "provider_failed provider=%s model=%s category=%s", self.provider.name, self.provider.model, category.value
        )
        return self._fallback(client, category, self.provider.name, self.provider.model)

    @staticmethod
    def _fallback(client: Client, category: FailureCategory, provider: str, model: str) -> _Outcome:
        return _Outcome(
            content=FALLBACK_REPLY.replace("{business}", client.business_name),
            provider=FALLBACK_PROVIDER,
            model=FALLBACK_MODEL,
            status=MessageProcessingStatus.FALLBACK,
            failure_category=category,
            attempted_provider=provider[:64],
            attempted_model=model[:128],
        )

    # ---- transaction 2: store the reply and update the conversation -------------------------
    def _persist_outcome(
        self,
        conversation: Conversation,
        inbound: Message,
        intent: IntentResult,
        outcome: _Outcome,
        *,
        replay: bool,
    ) -> ProcessingResult:
        now = _utcnow()
        generated = outcome.status is MessageProcessingStatus.GENERATED
        try:
            # Reserving the sequence takes the write lock on the conversation. Everything after it
            # runs while no other writer can interleave.
            sequence = allocate_message_sequence(self.db, conversation.id)
            existing_reply = self._find_reply(inbound.id)
            if existing_reply is not None:  # another request already answered this message
                self.db.rollback()
                return self._result(conversation, inbound, existing_reply, replay=True)

            inbound.intent = intent.intent.value
            inbound.intent_confidence = intent.confidence
            inbound.intent_method = intent.method
            if generated:
                inbound.processing_status = MessageProcessingStatus.PROCESSED
            else:
                inbound.processing_status = MessageProcessingStatus.FAILED
                inbound.failure_category = outcome.failure_category
                inbound.provider = outcome.attempted_provider  # which provider was attempted
                inbound.model = outcome.attempted_model

            outbound = Message(
                conversation_id=conversation.id,
                sequence=sequence,
                sender_type=SenderType.AI if generated else SenderType.SYSTEM,
                direction=MessageDirection.OUTBOUND,
                content=outcome.content,
                provider=outcome.provider,
                model=outcome.model,
                processing_status=outcome.status,
                failure_category=outcome.failure_category,
                in_reply_to_id=inbound.id,
                created_at=now,
            )
            self.db.add(outbound)

            # After a normal reply we wait for the customer. A handoff request or a fallback reply
            # keeps the conversation open for staff.
            waiting_for_customer = generated and intent.intent is not Intent.HUMAN_HANDOFF
            self._move_status(
                conversation, ConversationStatus.AWAITING_RESPONSE if waiting_for_customer else ConversationStatus.OPEN
            )
            conversation.last_message_at = now
            self.db.commit()
        except SQLAlchemyError:
            self.db.rollback()
            raise
        return ProcessingResult(
            conversation=conversation, inbound=inbound, outbound=outbound, intent=intent, idempotent_replay=replay
        )

    def _move_status(self, conversation: Conversation, target: ConversationStatus) -> None:
        """Apply a status change only if the transition rules allow it (e.g. never reopen a closed conversation)."""
        try:
            apply_status(conversation, target)
        except InvalidStatusTransitionError:
            logger.warning(
                "status_change_skipped conversation_id=%s from=%s to=%s",
                conversation.id,
                conversation.status.value,
                target.value,
            )

    # ---- result building ----------------------------------------------------------------------
    @staticmethod
    def _result(conversation: Conversation, inbound: Message, outbound: Message, *, replay: bool) -> ProcessingResult:
        try:
            intent = Intent(inbound.intent) if inbound.intent else Intent.UNKNOWN
        except ValueError:
            intent = Intent.UNKNOWN
        stored = IntentResult(
            intent=intent,
            confidence=inbound.intent_confidence if inbound.intent_confidence is not None else 0.0,
            method=inbound.intent_method or CLASSIFICATION_METHOD,
        )
        return ProcessingResult(
            conversation=conversation, inbound=inbound, outbound=outbound, intent=stored, idempotent_replay=replay
        )

    def _log_result(self, result: ProcessingResult, *, content_chars: int, started: float) -> None:
        """Routine log line. Contains IDs, labels and sizes only, never message text."""
        logger.info(
            "message_processed client_id=%s conversation_id=%s inbound_id=%s outbound_id=%s intent=%s "
            "confidence=%.2f method=%s reply_type=%s provider=%s model=%s replay=%s inbound_chars=%d duration_ms=%.1f",
            result.conversation.client_id,
            result.conversation.id,
            result.inbound.id,
            result.outbound.id,
            result.intent.intent.value,
            result.intent.confidence,
            result.intent.method,
            result.outbound.processing_status.value,
            result.outbound.provider,
            result.outbound.model,
            result.idempotent_replay,
            content_chars,
            (time.perf_counter() - started) * 1000,
        )


def _validate_response(response: object) -> str | None:
    """Return a short reason code if the provider output is unusable, else None."""
    if not isinstance(response, LLMResponse):
        return "not_an_llm_response"
    if not isinstance(response.content, str) or not response.content.strip():
        return "empty_content"
    if len(response.content.strip()) > MAX_REPLY_LENGTH:
        return "content_too_long"
    if not isinstance(response.provider, str) or not response.provider.strip() or len(response.provider) > 64:
        return "bad_provider_name"
    if not isinstance(response.model, str) or not response.model.strip() or len(response.model) > 128:
        return "bad_model_name"
    return None
