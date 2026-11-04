"""Database engine, session management and declarative base.

Switching to PostgreSQL only requires changing DATABASE_URL (and installing a
driver such as `psycopg[binary]`); nothing here is SQLite-specific except the
`check_same_thread` connect argument, which is applied only for SQLite URLs.
"""

from collections.abc import Iterator
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy import DateTime, Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base shared by every model in the platform."""


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes on every backend (SQLite drops tzinfo)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetimes are not allowed; use timezone-aware UTC.")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def configure_sqlite(engine: Engine) -> None:
    """Make SQLite enforce foreign keys.

    SQLite ignores FOREIGN KEY constraints unless `PRAGMA foreign_keys=ON` is run
    on *every* connection. PostgreSQL always enforces them, so this keeps the two
    backends behaving the same. It changes connection behaviour only, never the
    database file.
    """

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def build_engine(database_url: str, echo: bool = False) -> Engine:
    kwargs: dict[str, Any] = {"echo": echo}
    is_sqlite = database_url.startswith("sqlite")
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
    engine = create_engine(database_url, **kwargs)
    if is_sqlite:
        configure_sqlite(engine)
    return engine


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return build_engine(settings.database_url, settings.db_echo)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db(engine: Engine | None = None) -> None:
    """Create tables that do not exist yet.

    Fine for Phase 1. Once the schema starts evolving (Phase 2+), replace this
    with Alembic migrations.
    """
    import app.models  # noqa: F401  (registers models on Base.metadata)

    Base.metadata.create_all(bind=engine or get_engine())
