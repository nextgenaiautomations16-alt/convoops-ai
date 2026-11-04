"""API tests. Each test gets its own isolated in-memory SQLite database."""

import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.main import create_app
from app.services.client_service import ClientService

API = "/api/v1/clients"


@pytest.fixture()
def client() -> Iterator[TestClient]:
    import app.models  # noqa: F401  (register models)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one shared in-memory connection per test
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db() -> Iterator[Session]:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    application = create_app()
    application.dependency_overrides[get_db] = override_get_db
    yield TestClient(application)  # no `with`: the startup hook (real DB) is not run
    engine.dispose()


def make_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "business_name": "Sunrise Dental Studio",
        "slug": "sunrise-dental",
        "industry": "Dental",
        "timezone": "America/Chicago",
        "business_description": "Synthetic family dental practice used for testing.",
        "contact_email": "Front.Desk@example.com",
    }
    payload.update(overrides)
    return payload


def create(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post(API, json=make_payload(**overrides))
    assert response.status_code == 201, response.text
    return response.json()


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---- health ---------------------------------------------------------------
def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "up"


# ---- create ---------------------------------------------------------------
def test_create_client_success(client: TestClient) -> None:
    response = client.post(API, json=make_payload())
    assert response.status_code == 201
    body = response.json()
    uuid.UUID(body["id"])
    assert response.headers["Location"] == f"{API}/{body['id']}"
    assert body["business_name"] == "Sunrise Dental Studio"
    assert body["slug"] == "sunrise-dental"
    assert body["timezone"] == "America/Chicago"
    assert body["contact_email"] == "front.desk@example.com"  # normalised
    assert body["configuration_status"] == "draft"
    assert parse_ts(body["created_at"]).tzinfo is not None
    assert body["created_at"] == body["updated_at"]


def test_create_client_without_optional_email(client: TestClient) -> None:
    payload = make_payload()
    del payload["contact_email"]
    response = client.post(API, json=payload)
    assert response.status_code == 201
    assert response.json()["contact_email"] is None


def test_create_client_missing_required_fields(client: TestClient) -> None:
    response = client.post(API, json={"business_name": "Only A Name"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    missing = {d["field"] for d in error["details"]}
    assert {"body.slug", "body.industry", "body.timezone", "body.business_description"} <= missing


@pytest.mark.parametrize(
    "overrides",
    [
        {"slug": "Not A Slug"},
        {"slug": "-leading-hyphen"},
        {"slug": "double--hyphen"},
        {"timezone": "Mars/Olympus_Mons"},
        {"contact_email": "not-an-email"},
        {"business_name": "   "},
        {"configuration_status": "banana"},
        {"unexpected_field": "nope"},
    ],
)
def test_create_client_rejects_invalid_values(client: TestClient, overrides: dict[str, Any]) -> None:
    response = client.post(API, json=make_payload(**overrides))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_create_client_rejects_duplicate_slug(client: TestClient) -> None:
    create(client)
    response = client.post(API, json=make_payload(business_name="Another Business"))
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "conflict"
    assert "sunrise-dental" in error["message"]
    assert client.get(API).json()["total"] == 1


def test_create_client_rejects_duplicate_business_name_case_insensitive(client: TestClient) -> None:
    create(client)
    response = client.post(API, json=make_payload(slug="other-slug", business_name="SUNRISE dental studio"))
    assert response.status_code == 409


# ---- list -----------------------------------------------------------------
def test_list_clients(client: TestClient) -> None:
    create(client, slug="alpha-co", business_name="Alpha Co")
    create(client, slug="beta-co", business_name="Beta Co")
    response = client.get(API)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["limit"] == 20
    assert body["offset"] == 0
    assert [c["slug"] for c in body["items"]] == ["beta-co", "alpha-co"]  # newest first


def test_list_clients_empty(client: TestClient) -> None:
    body = client.get(API).json()
    assert body == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_list_clients_pagination(client: TestClient) -> None:
    for i in range(5):
        create(client, slug=f"client-{i}", business_name=f"Client {i}")
    first = client.get(API, params={"limit": 2, "offset": 0}).json()
    second = client.get(API, params={"limit": 2, "offset": 2}).json()
    last = client.get(API, params={"limit": 2, "offset": 4}).json()
    assert [first["total"], second["total"], last["total"]] == [5, 5, 5]
    slugs = [c["slug"] for page in (first, second, last) for c in page["items"]]
    assert slugs == ["client-4", "client-3", "client-2", "client-1", "client-0"]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "x"}])
def test_list_clients_rejects_bad_pagination(client: TestClient, params: dict[str, Any]) -> None:
    assert client.get(API, params=params).status_code == 422


def test_list_clients_filtered_by_status(client: TestClient) -> None:
    create(client, slug="a-co", business_name="A Co")
    active = create(client, slug="b-co", business_name="B Co")
    client.patch(f"{API}/{active['id']}", json={"configuration_status": "active"})
    body = client.get(API, params={"configuration_status": "active"}).json()
    assert body["total"] == 1
    assert body["items"][0]["slug"] == "b-co"


# ---- retrieve -------------------------------------------------------------
def test_get_client(client: TestClient) -> None:
    created = create(client)
    response = client.get(f"{API}/{created['id']}")
    assert response.status_code == 200
    assert response.json() == created


def test_get_nonexistent_client_returns_404(client: TestClient) -> None:
    response = client.get(f"{API}/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_get_client_with_invalid_id_returns_422(client: TestClient) -> None:
    response = client.get(f"{API}/not-a-uuid")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# ---- update ---------------------------------------------------------------
def test_update_client_configuration(client: TestClient) -> None:
    created = create(client)
    response = client.patch(
        f"{API}/{created['id']}",
        json={"industry": "Orthodontics", "timezone": "America/Denver", "business_name": "Sunrise Ortho"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["industry"] == "Orthodontics"
    assert body["timezone"] == "America/Denver"
    assert body["business_name"] == "Sunrise Ortho"
    assert body["slug"] == created["slug"]
    assert body["business_description"] == created["business_description"]  # untouched
    assert body["created_at"] == created["created_at"]
    assert parse_ts(body["updated_at"]) > parse_ts(created["updated_at"])
    assert client.get(f"{API}/{created['id']}").json() == body  # persisted


def test_update_can_clear_contact_email(client: TestClient) -> None:
    created = create(client)
    response = client.patch(f"{API}/{created['id']}", json={"contact_email": None})
    assert response.status_code == 200
    assert response.json()["contact_email"] is None


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"slug": "new-slug"},  # slug is immutable
        {"business_name": None},
        {"configuration_status": None},
        {"timezone": "Nowhere/Land"},
        {"configuration_status": "banana"},
    ],
)
def test_update_rejects_invalid_payloads(client: TestClient, body: dict[str, Any]) -> None:
    created = create(client)
    response = client.patch(f"{API}/{created['id']}", json=body)
    assert response.status_code == 422
    assert client.get(f"{API}/{created['id']}").json() == created  # unchanged


def test_update_nonexistent_client_returns_404(client: TestClient) -> None:
    response = client.patch(f"{API}/{uuid.uuid4()}", json={"industry": "Retail"})
    assert response.status_code == 404


def test_update_with_invalid_id_returns_422(client: TestClient) -> None:
    assert client.patch(f"{API}/12345", json={"industry": "Retail"}).status_code == 422


def test_update_business_name_conflict_returns_409(client: TestClient) -> None:
    create(client, slug="a-co", business_name="A Co")
    other = create(client, slug="b-co", business_name="B Co")
    response = client.patch(f"{API}/{other['id']}", json={"business_name": "a co"})
    assert response.status_code == 409
    assert client.get(f"{API}/{other['id']}").json()["business_name"] == "B Co"


def test_update_to_own_business_name_is_allowed(client: TestClient) -> None:
    created = create(client)
    response = client.patch(f"{API}/{created['id']}", json={"business_name": "sunrise dental studio"})
    assert response.status_code == 200


# ---- activation -----------------------------------------------------------
def test_client_activation_and_deactivation(client: TestClient) -> None:
    created = create(client)
    url = f"{API}/{created['id']}"
    assert created["configuration_status"] == "draft"

    activated = client.patch(url, json={"configuration_status": "active"})
    assert activated.status_code == 200
    assert activated.json()["configuration_status"] == "active"

    deactivated = client.patch(url, json={"configuration_status": "inactive"})
    assert deactivated.status_code == 200
    assert deactivated.json()["configuration_status"] == "inactive"

    reactivated = client.patch(url, json={"configuration_status": "active"})
    assert reactivated.json()["configuration_status"] == "active"


# ---- error contract -------------------------------------------------------
def test_unknown_route_uses_error_format(client: TestClient) -> None:
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_internal_database_errors_are_not_exposed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: ClientService, *args: Any, **kwargs: Any) -> None:
        raise SQLAlchemyError("secret internal detail: SELECT * FROM clients")

    monkeypatch.setattr(ClientService, "get", boom)
    response = client.get(f"{API}/{uuid.uuid4()}")
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "An internal error occurred."}
    }
    assert "secret" not in response.text
