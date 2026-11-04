"""Client ORM model."""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, Enum, Index, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base, UTCDateTime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConfigurationStatus(str, enum.Enum):
    DRAFT = "draft"  # registered, not yet live
    ACTIVE = "active"  # configured and enabled
    INACTIVE = "inactive"  # deactivated


class Client(Base):
    __tablename__ = "clients"
    __table_args__ = (
        CheckConstraint("length(trim(business_name)) > 0", name="ck_clients_business_name_not_blank"),
        CheckConstraint("length(slug) >= 2", name="ck_clients_slug_min_length"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    business_name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    industry: Mapped[str] = mapped_column(String(100), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    business_description: Mapped[str] = mapped_column(Text, nullable=False)
    contact_email: Mapped[str | None] = mapped_column(String(254), nullable=True)
    configuration_status: Mapped[ConfigurationStatus] = mapped_column(
        Enum(
            ConfigurationStatus,
            name="configuration_status",
            native_enum=False,
            length=20,
            create_constraint=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=ConfigurationStatus.DRAFT,
        server_default=ConfigurationStatus.DRAFT.value,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"Client(id={self.id}, slug={self.slug!r}, status={self.configuration_status.value})"


# Case-insensitive uniqueness of business names (works on SQLite and PostgreSQL).
Index("uq_clients_business_name_lower", func.lower(Client.business_name), unique=True)
