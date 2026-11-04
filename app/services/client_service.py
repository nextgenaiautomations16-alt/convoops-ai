"""Client business logic. Owns validation of uniqueness and transaction handling."""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError, NotFoundError
from app.models.client import Client, ConfigurationStatus
from app.schemas.client import ClientCreate, ClientUpdate

logger = logging.getLogger(__name__)


class ClientService:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ---- reads -----------------------------------------------------------
    def get(self, client_id: uuid.UUID) -> Client:
        client = self.db.get(Client, client_id)
        if client is None:
            raise NotFoundError(f"Client {client_id} was not found.")
        return client

    def list(
        self,
        *,
        limit: int,
        offset: int,
        status: ConfigurationStatus | None = None,
    ) -> tuple[list[Client], int]:
        query = select(Client)
        count_query = select(func.count()).select_from(Client)
        if status is not None:
            query = query.where(Client.configuration_status == status)
            count_query = count_query.where(Client.configuration_status == status)

        total = self.db.scalar(count_query) or 0
        items = self.db.scalars(
            query.order_by(Client.created_at.desc(), Client.id).limit(limit).offset(offset)
        ).all()
        return list(items), total

    # ---- writes ----------------------------------------------------------
    def create(self, data: ClientCreate) -> Client:
        self._ensure_unique(slug=data.slug, business_name=data.business_name)
        now = datetime.now(timezone.utc)  # identical created/updated stamps on insert
        client = Client(**data.model_dump(), created_at=now, updated_at=now)
        self.db.add(client)
        self._commit()
        self.db.refresh(client)
        logger.info("client_created client_id=%s slug=%s", client.id, client.slug)
        return client

    def update(self, client_id: uuid.UUID, data: ClientUpdate) -> Client:
        client = self.get(client_id)
        changes = data.model_dump(exclude_unset=True)

        new_name = changes.get("business_name")
        if new_name is not None:
            self._ensure_unique(business_name=new_name, exclude_id=client.id)

        for field, value in changes.items():
            setattr(client, field, value)
        self._commit()
        self.db.refresh(client)
        logger.info("client_updated client_id=%s fields=%s", client.id, sorted(changes))
        return client

    # ---- helpers ---------------------------------------------------------
    def _ensure_unique(
        self,
        *,
        slug: str | None = None,
        business_name: str | None = None,
        exclude_id: uuid.UUID | None = None,
    ) -> None:
        if slug is not None:
            query = select(Client.id).where(Client.slug == slug)
            if exclude_id is not None:
                query = query.where(Client.id != exclude_id)
            if self.db.scalar(query) is not None:
                raise ConflictError(f"A client with slug '{slug}' already exists.")
        if business_name is not None:
            query = select(Client.id).where(func.lower(Client.business_name) == business_name.lower())
            if exclude_id is not None:
                query = query.where(Client.id != exclude_id)
            if self.db.scalar(query) is not None:
                raise ConflictError(f"A client named '{business_name}' already exists.")

    def _commit(self) -> None:
        """Commit, rolling back on failure.

        The pre-checks above give friendly messages; the database constraints are
        the final guard against concurrent requests racing past them.
        """
        try:
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            logger.warning("client_integrity_conflict error_type=%s", type(exc.orig).__name__)
            raise ConflictError("A client with the same slug or business name already exists.") from exc
        except SQLAlchemyError:
            self.db.rollback()
            raise
