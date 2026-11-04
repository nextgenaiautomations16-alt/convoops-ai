"""HTTP layer for submitting an inbound customer message."""

import uuid

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.providers.base import LLMProvider
from app.providers.factory import get_llm_provider
from app.schemas.client import ErrorResponse
from app.schemas.conversation import MessageRead
from app.schemas.message import IntentInfo, MessageCreate, MessageProcessingResponse
from app.services.message_processing_service import MessageProcessingService, ProcessingResult

router = APIRouter(
    prefix="/clients/{client_id}/conversations/{conversation_id}/messages", tags=["messages"]
)


def get_message_processing_service(
    db: Session = Depends(get_db), provider: LLMProvider = Depends(get_llm_provider)
) -> MessageProcessingService:
    return MessageProcessingService(db, provider)


def _to_response(result: ProcessingResult) -> MessageProcessingResponse:
    return MessageProcessingResponse(
        conversation_id=result.conversation.id,
        conversation_status=result.conversation.status,
        idempotent_replay=result.idempotent_replay,
        processing_status=result.inbound.processing_status,
        reply_type=result.outbound.processing_status,
        failure_category=result.outbound.failure_category,
        intent=IntentInfo(
            intent=result.intent.intent.value, confidence=result.intent.confidence, method=result.intent.method
        ),
        provider=result.outbound.provider,
        model=result.outbound.model,
        inbound_message=MessageRead.model_validate(result.inbound),
        outbound_message=MessageRead.model_validate(result.outbound),
    )


@router.post(
    "",
    response_model=MessageProcessingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a customer message and get the reply",
    description=(
        "Stores the inbound message, classifies its intent (deterministic rules, not an LLM), asks the "
        "configured provider for a reply and stores it. The inbound message is saved before the provider is "
        "called; if the provider fails, a safe fallback reply is stored and returned (`reply_type: fallback`).\n\n"
        "**Idempotency:** send `idempotency_key` to make retries safe. The same key with the same content "
        "returns the original result with HTTP 200 and `idempotent_replay: true`; the same key with different "
        "content returns 409."
    ),
    responses={
        200: {
            "model": MessageProcessingResponse,
            "description": "Idempotent replay: this request repeated an earlier one, nothing new was created.",
        },
        404: {"model": ErrorResponse, "description": "Unknown client, or conversation not found for this client"},
        409: {
            "model": ErrorResponse,
            "description": (
                "client_not_active, conversation_closed, idempotency_key_conflict "
                "or processing_in_progress (retry shortly)"
            ),
        },
        422: {"model": ErrorResponse, "description": "Invalid message or identifiers"},
    },
)
def submit_message(
    client_id: uuid.UUID,
    conversation_id: uuid.UUID,
    payload: MessageCreate,
    response: Response,
    service: MessageProcessingService = Depends(get_message_processing_service),
) -> MessageProcessingResponse:
    result = service.process_inbound(client_id, conversation_id, payload.content, payload.idempotency_key)
    if result.idempotent_replay:
        response.status_code = status.HTTP_200_OK
    return _to_response(result)
