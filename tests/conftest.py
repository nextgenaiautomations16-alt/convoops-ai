"""Shared fixtures for model-level tests.

`tests/test_clients.py` keeps its own `client` fixture untouched, so the
40 Phase 1 tests behave exactly as before.
"""

from collections.abc import Iterator

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings

# Tests must behave the same on every machine. They must never read the developer's `.env` file or
# shell variables: for example LLM_PROVIDER=ollama in `.env` would otherwise make tests call a real
# Ollama server. This runs BEFORE the application modules below are imported, because some of them
# read settings when they are imported.
Settings.model_config["env_file"] = None
for _name in Settings.model_fields:
    os.environ.pop(_name.upper(), None)
get_settings.cache_clear()

import app.models  # noqa: E402, F401  (registers every model on Base.metadata)
from app.db.database import Base, configure_sqlite, get_db  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _default_settings_for_every_test() -> Iterator[None]:
    """Start each test from default settings (the mock provider) and leave no cached settings behind."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def engine() -> Iterator[Engine]:
    """A brand-new in-memory SQLite database per test. Never touches convoops.db."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    configure_sqlite(test_engine)  # same foreign-key enforcement as the real engine
    Base.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture()
def db_session(engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with factory() as session:
        yield session


@pytest.fixture()
def app_client(engine: Engine) -> Iterator[TestClient]:
    """API test client wired to the SAME in-memory database as `db_session`.

    Lets a test call the API and also inspect or seed rows directly (for example
    messages, which have no creation endpoint yet).
    """
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db() -> Iterator[Session]:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    application = create_app()
    application.dependency_overrides[get_db] = override_get_db
    yield TestClient(application)  # no `with`: startup hook (real DB) is not run
