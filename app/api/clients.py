"""HTTP layer for clients: parsing, status codes and response shapes only."""

import uuid

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.models.client import ConfigurationStatus
from app.schemas.client import ClientCreate, ClientList, ClientRead, ClientUpdate, ErrorResponse
from app.services.client_service import ClientService

settings = get_settings()
router = APIRouter(prefix="/clients", tags=["clients"])


def get_client_service(db: Session = Depends(get_db)) -> ClientService:
    return ClientService(db)


@router.post(
    "",
    response_model=ClientRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a client",
    responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def create_client(
    payload: ClientCreate,
    response: Response,
    service: ClientService = Depends(get_client_service),
) -> ClientRead:
    client = service.create(payload)
    response.headers["Location"] = f"{settings.api_v1_prefix}/clients/{client.id}"
    return ClientRead.model_validate(client)


@router.get("", response_model=ClientList, summary="List clients (paginated)")
def list_clients(
    limit: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size),
    offset: int = Query(0, ge=0),
    configuration_status: ConfigurationStatus | None = Query(None),
    service: ClientService = Depends(get_client_service),
) -> ClientList:
    items, total = service.list(limit=limit, offset=offset, status=configuration_status)
    return ClientList(
        items=[ClientRead.model_validate(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{client_id}",
    response_model=ClientRead,
    summary="Get a client",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def get_client(
    client_id: uuid.UUID, service: ClientService = Depends(get_client_service)
) -> ClientRead:
    return ClientRead.model_validate(service.get(client_id))


@router.patch(
    "/{client_id}",
    response_model=ClientRead,
    summary="Update a client's configuration or activate/deactivate it",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def update_client(
    client_id: uuid.UUID,
    payload: ClientUpdate,
    service: ClientService = Depends(get_client_service),
) -> ClientRead:
    return ClientRead.model_validate(service.update(client_id, payload))
