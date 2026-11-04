"""HTTP layer for conversations: parsing, status codes and response shapes only."""

import uuid

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.models.conversation import ConversationStatus
from app.schemas.client import ErrorResponse
from app.schemas.conversation import (
    ConversationCreate,
    ConversationList,
    ConversationRead,
    MessageList,
    MessageRead,
)
from app.services.conversation_service import ConversationService

settings = get_settings()
router = APIRouter(prefix="/clients/{client_id}/conversations", tags=["conversations"])


def get_conversation_service(db: Session = Depends(get_db)) -> ConversationService:
    return ConversationService(db)


@router.post(
    "",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Start a conversation for a client",
    description="The client must exist and be `active`. New conversations start as `open`.",
    responses={
        404: {"model": ErrorResponse, "description": "Client not found"},
        409: {"model": ErrorResponse, "description": "Client is not active"},
        422: {"model": ErrorResponse},
    },
)
def create_conversation(
    client_id: uuid.UUID,
    payload: ConversationCreate,
    response: Response,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationRead:
    conversation = service.create(client_id, payload)
    response.headers["Location"] = (
        f"{settings.api_v1_prefix}/clients/{client_id}/conversations/{conversation.id}"
    )
    return ConversationRead.model_validate(conversation)


@router.get(
    "",
    response_model=ConversationList,
    summary="List a client's conversations (paginated, newest first)",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def list_conversations(
    client_id: uuid.UUID,
    limit: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size),
    offset: int = Query(0, ge=0),
    conversation_status: ConversationStatus | None = Query(None, alias="status"),
    customer_id: str | None = Query(None, min_length=1, max_length=128),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationList:
    items, total = service.list_conversations(
        client_id, limit=limit, offset=offset, status=conversation_status, customer_id=customer_id
    )
    return ConversationList(
        items=[ConversationRead.model_validate(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationRead,
    summary="Get one conversation",
    description="Returns 404 if the conversation does not exist OR belongs to another client.",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def get_conversation(
    client_id: uuid.UUID,
    conversation_id: uuid.UUID,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationRead:
    return ConversationRead.model_validate(service.get(client_id, conversation_id))


@router.get(
    "/{conversation_id}/messages",
    response_model=MessageList,
    summary="Get a conversation's messages (paginated, oldest first)",
    description="Messages are ordered by their per-conversation `sequence`.",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def list_messages(
    client_id: uuid.UUID,
    conversation_id: uuid.UUID,
    limit: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size),
    offset: int = Query(0, ge=0),
    service: ConversationService = Depends(get_conversation_service),
) -> MessageList:
    items, total = service.list_messages(client_id, conversation_id, limit=limit, offset=offset)
    return MessageList(
        items=[MessageRead.model_validate(m) for m in items],
        total=total,
        limit=limit,
        offset=offset,
    )
