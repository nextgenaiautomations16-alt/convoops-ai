"""Conversation business logic: tenant-safe reads and conversation creation.

Tenant isolation rule: a conversation is never fetched by its ID alone. Every
lookup goes through `_get_owned_conversation`, which filters on BOTH
`client_id` and `conversation_id`.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError, NotFoundError
from app.models.client import Client, ConfigurationStatus
from app.models.conversation import Conversation, ConversationStatus, Message
from app.schemas.conversation import ConversationCreate

logger = logging.getLogger(__name__)


class ClientNotActiveError(ConflictError):
    """Raised when a conversation is requested for a client that is not `active`."""

    code = "client_not_active"


class ConversationService:
    """Reads and creates conversations. Read methods never commit."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ---- reads -----------------------------------------------------------
    def get(self, client_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation:
        """Return one conversation owned by `client_id`, or raise NotFoundError."""
        return self._get_owned_conversation(client_id, conversation_id)

    def list_conversations(
        self,
        client_id: uuid.UUID,
        *,
        limit: int,
        offset: int,
        status: ConversationStatus | None = None,
        customer_id: str | None = None,
    ) -> tuple[list[Conversation], int]:
        """Return (page of conversations, total matching) for one client, newest first."""
        self._get_client(client_id)
        conditions = [Conversation.client_id == client_id]
        if status is not None:
            conditions.append(Conversation.status == status)
        if customer_id is not None:
            conditions.append(Conversation.customer_id == customer_id)

        total = self.db.scalar(select(func.count()).select_from(Conversation).where(*conditions)) or 0
        items = self.db.scalars(
            select(Conversation)
            .where(*conditions)
            .order_by(Conversation.created_at.desc(), Conversation.id)
            .limit(limit)
            .offset(offset)
        ).all()
        return list(items), total

    def list_messages(
        self, client_id: uuid.UUID, conversation_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Message], int]:
        """Return (page of messages, total) in chronological (sequence) order.

        Ownership is verified first; messages are never queried for a conversation
        that does not belong to `client_id`.
        """
        conversation = self._get_owned_conversation(client_id, conversation_id)
        total = (
            self.db.scalar(
                select(func.count()).select_from(Message).where(Message.conversation_id == conversation.id)
            )
            or 0
        )
        items = self.db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.sequence)
            .limit(limit)
            .offset(offset)
        ).all()
        return list(items), total

    # ---- writes ----------------------------------------------------------
    def create(self, client_id: uuid.UUID, data: ConversationCreate) -> Conversation:
        """Start a conversation for an existing, active client (single commit)."""
        client = self._get_client(client_id)
        if client.configuration_status is not ConfigurationStatus.ACTIVE:
            raise ClientNotActiveError(
                f"Client {client_id} is '{client.configuration_status.value}'; "
                "conversations can only be created for active clients."
            )
        now = datetime.now(timezone.utc)  # identical created/updated stamps on insert
        conversation = Conversation(
            client_id=client.id,
            customer_id=data.customer_id,
            channel=data.channel,
            created_at=now,
            updated_at=now,
        )
        self.db.add(conversation)
        self._commit()
        self.db.refresh(conversation)
        logger.info("conversation_created client_id=%s conversation_id=%s", client.id, conversation.id)
        return conversation

    # ---- helpers (none of these commit) ------------------------------------
    def _get_client(self, client_id: uuid.UUID) -> Client:
        client = self.db.get(Client, client_id)
        if client is None:
            raise NotFoundError(f"Client {client_id} was not found.")
        return client

    def _get_owned_conversation(self, client_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation:
        """The single tenant-isolation gate: filters on client_id AND conversation_id.

        A conversation that exists but belongs to another client produces exactly the
        same error as one that does not exist, so its existence is not revealed.
        """
        self._get_client(client_id)
        conversation = self.db.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.client_id == client_id
            )
        )
        if conversation is None:
            raise NotFoundError(f"Conversation {conversation_id} was not found.")
        return conversation

    def _commit(self) -> None:
        """Commit, rolling back on failure. Internal error details are only logged."""
        try:
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            logger.warning("conversation_integrity_error error_type=%s", type(exc.orig).__name__)
            raise ConflictError("The conversation could not be created.") from exc
        except SQLAlchemyError:
            self.db.rollback()
            raise
