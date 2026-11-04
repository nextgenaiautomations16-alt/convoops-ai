"""Step 4 tests: conversation creation, listing, retrieval and message history.

Each test gets a fresh in-memory database (see tests/conftest.py); the real
convoops.db is never touched. Messages have no creation endpoint yet, so tests
seed them directly through `db_session` using the existing sequence allocator.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import (
    Conversation,
    ConversationStatus,
    Message,
    MessageDirection,
    MessageProcessingStatus,
    SenderType,
    allocate_message_sequence,
)
from app.services.conversation_service import ConversationService

CLIENTS = "/api/v1/clients"


# ---- helpers -----------------------------------------------------------------
def make_client(api: TestClient, slug: str = "demo-clinic", status: str = "active") -> dict[str, Any]:
    response = api.post(
        CLIENTS,
        json={
            "business_name": f"Demo Clinic {slug}",
            "slug": slug,
            "industry": "healthcare",
            "timezone": "America/New_York",
            "business_description": "Synthetic demo clinic. No real patient data.",
            "configuration_status": status,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def conv_url(client_id: str, conversation_id: str | None = None, suffix: str = "") -> str:
    base = f"{CLIENTS}/{client_id}/conversations"
    return f"{base}/{conversation_id}{suffix}" if conversation_id else base


def make_conversation(api: TestClient, client_id: str, customer_id: str = "cust_1001") -> dict[str, Any]:
    response = api.post(conv_url(client_id), json={"customer_id": customer_id})
    assert response.status_code == 201, response.text
    return response.json()


def seed_messages(db: Session, conversation_id: str, count: int, text: str = "synthetic text") -> None:
    """Insert alternating customer/ai messages using the real sequence allocator."""
    cid = uuid.UUID(conversation_id)
    for i in range(count):
        inbound = i % 2 == 0
        db.add(
            Message(
                conversation_id=cid,
                sequence=allocate_message_sequence(db, cid),
                sender_type=SenderType.CUSTOMER if inbound else SenderType.AI,
                direction=MessageDirection.INBOUND if inbound else MessageDirection.OUTBOUND,
                content=f"{text} {i + 1}",
                processing_status=(
                    MessageProcessingStatus.PROCESSED if inbound else MessageProcessingStatus.GENERATED
                ),
            )
        )
    db.commit()


def count_conversations(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(Conversation)) or 0


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# =============================== creation ======================================
def test_create_conversation_for_active_client(app_client: TestClient, db_session: Session) -> None:
    client = make_client(app_client)
    response = app_client.post(conv_url(client["id"]), json={"customer_id": "cust_1001"})

    assert response.status_code == 201
    body = response.json()
    uuid.UUID(body["id"])
    assert response.headers["Location"] == conv_url(client["id"], body["id"])
    assert body["client_id"] == client["id"]
    assert body["customer_id"] == "cust_1001"
    assert body["channel"] == "sms"  # default
    assert body["status"] == "open"
    assert body["last_message_at"] is None
    assert body["created_at"] == body["updated_at"]
    assert parse_ts(body["created_at"]).tzinfo is not None
    assert "last_message_sequence" not in body  # internal counter is not exposed
    assert count_conversations(db_session) == 1  # actually persisted


def test_create_conversation_with_explicit_sms_channel(app_client: TestClient) -> None:
    client = make_client(app_client)
    response = app_client.post(conv_url(client["id"]), json={"customer_id": "c1", "channel": "sms"})
    assert response.status_code == 201
    assert response.json()["channel"] == "sms"


def test_same_customer_can_have_several_conversations(app_client: TestClient) -> None:
    client = make_client(app_client)
    first = make_conversation(app_client, client["id"], "cust_1001")
    second = make_conversation(app_client, client["id"], "cust_1001")
    assert first["id"] != second["id"]


def test_create_conversation_rejected_for_draft_client(app_client: TestClient, db_session: Session) -> None:
    client = make_client(app_client, status="draft")
    response = app_client.post(conv_url(client["id"]), json={"customer_id": "cust_1001"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "client_not_active"
    assert count_conversations(db_session) == 0


def test_create_conversation_rejected_for_inactive_client(
    app_client: TestClient, db_session: Session
) -> None:
    client = make_client(app_client)
    deactivated = app_client.patch(f"{CLIENTS}/{client['id']}", json={"configuration_status": "inactive"})
    assert deactivated.status_code == 200

    response = app_client.post(conv_url(client["id"]), json={"customer_id": "cust_1001"})
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "client_not_active"
    assert "inactive" in error["message"]
    assert count_conversations(db_session) == 0


def test_create_conversation_for_nonexistent_client(app_client: TestClient, db_session: Session) -> None:
    response = app_client.post(conv_url(str(uuid.uuid4())), json={"customer_id": "cust_1001"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert count_conversations(db_session) == 0


def test_create_conversation_with_invalid_client_id(app_client: TestClient) -> None:
    response = app_client.post(conv_url("not-a-uuid"), json={"customer_id": "cust_1001"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "body",
    [
        {},  # customer_id missing
        {"customer_id": ""},
        {"customer_id": "   "},
        {"customer_id": "x" * 129},
        {"customer_id": "has space"},
        {"customer_id": "john@example.com"},  # emails are not accepted as identifiers
        {"customer_id": "+15551234567"},  # nor phone numbers
        {"customer_id": "../etc"},
        {"customer_id": "line\nbreak"},
        {"customer_id": "_leading-underscore"},
        {"customer_id": 12345},
        {"customer_id": None},
        {"customer_id": "cust_1", "channel": "email"},
        {"customer_id": "cust_1", "channel": "SMS"},
        {"customer_id": "cust_1", "channel": None},
        {"customer_id": "cust_1", "status": "resolved"},  # status cannot be chosen on creation
        {"customer_id": "cust_1", "unexpected": True},
    ],
)
def test_create_conversation_validation_errors(
    app_client: TestClient, db_session: Session, body: dict[str, Any]
) -> None:
    client = make_client(app_client)
    response = app_client.post(conv_url(client["id"]), json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"]
    assert count_conversations(db_session) == 0  # nothing was stored


def test_create_conversation_error_names_the_bad_field(app_client: TestClient) -> None:
    client = make_client(app_client)
    response = app_client.post(conv_url(client["id"]), json={"customer_id": "cust_1", "channel": "email"})
    assert [d["field"] for d in response.json()["error"]["details"]] == ["body.channel"]


def test_create_conversation_with_non_json_body(app_client: TestClient) -> None:
    client = make_client(app_client)
    response = app_client.post(
        conv_url(client["id"]), content="not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_customer_id_boundaries_and_trimming(app_client: TestClient) -> None:
    client = make_client(app_client)
    longest = "a" * 128
    assert app_client.post(conv_url(client["id"]), json={"customer_id": longest}).status_code == 201
    trimmed = app_client.post(conv_url(client["id"]), json={"customer_id": "  cust_9  "})
    assert trimmed.status_code == 201
    assert trimmed.json()["customer_id"] == "cust_9"


def test_failed_commit_returns_generic_500_and_stores_nothing(
    app_client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(app_client)

    def boom(self: Session) -> None:
        raise SQLAlchemyError("secret internal detail: INSERT INTO conversations ...")

    monkeypatch.setattr(Session, "commit", boom)
    response = app_client.post(conv_url(client["id"]), json={"customer_id": "cust_1001"})
    monkeypatch.undo()

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "An internal error occurred."}}
    assert "secret" not in response.text
    assert count_conversations(db_session) == 0


# ================================ listing ======================================
def test_list_conversations_empty(app_client: TestClient) -> None:
    client = make_client(app_client)
    response = app_client.get(conv_url(client["id"]))
    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_list_conversations_pagination_newest_first(app_client: TestClient) -> None:
    client = make_client(app_client)
    created = [make_conversation(app_client, client["id"], f"cust_{i}") for i in range(5)]
    expected_order = [c["id"] for c in reversed(created)]  # newest first

    pages = [
        app_client.get(conv_url(client["id"]), params={"limit": 2, "offset": offset}).json()
        for offset in (0, 2, 4)
    ]
    assert [p["total"] for p in pages] == [5, 5, 5]
    assert [len(p["items"]) for p in pages] == [2, 2, 1]
    assert [item["id"] for p in pages for item in p["items"]] == expected_order
    assert app_client.get(conv_url(client["id"]), params={"offset": 99}).json()["items"] == []


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "x"}])
def test_list_conversations_rejects_bad_pagination(app_client: TestClient, params: dict[str, Any]) -> None:
    client = make_client(app_client)
    response = app_client.get(conv_url(client["id"]), params=params)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_list_conversations_filter_by_status(app_client: TestClient, db_session: Session) -> None:
    client = make_client(app_client)
    open_conv = make_conversation(app_client, client["id"], "cust_open")
    resolved = make_conversation(app_client, client["id"], "cust_done")
    row = db_session.get(Conversation, uuid.UUID(resolved["id"]))
    assert row is not None
    row.status = ConversationStatus.RESOLVED
    db_session.commit()

    only_open = app_client.get(conv_url(client["id"]), params={"status": "open"}).json()
    only_resolved = app_client.get(conv_url(client["id"]), params={"status": "resolved"}).json()
    assert [c["id"] for c in only_open["items"]] == [open_conv["id"]]
    assert [c["id"] for c in only_resolved["items"]] == [resolved["id"]]
    assert only_open["total"] == only_resolved["total"] == 1
    assert app_client.get(conv_url(client["id"]), params={"status": "closed"}).json()["total"] == 0


def test_list_conversations_filter_by_customer_id(app_client: TestClient) -> None:
    client = make_client(app_client)
    make_conversation(app_client, client["id"], "cust_a")
    target = make_conversation(app_client, client["id"], "cust_b")
    body = app_client.get(conv_url(client["id"]), params={"customer_id": "cust_b"}).json()
    assert [c["id"] for c in body["items"]] == [target["id"]]
    assert body["total"] == 1


@pytest.mark.parametrize("params", [{"status": "pending"}, {"customer_id": ""}])
def test_list_conversations_rejects_bad_filters(app_client: TestClient, params: dict[str, Any]) -> None:
    client = make_client(app_client)
    assert app_client.get(conv_url(client["id"]), params=params).status_code == 422


def test_list_conversations_for_nonexistent_client(app_client: TestClient) -> None:
    response = app_client.get(conv_url(str(uuid.uuid4())))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_list_conversations_with_invalid_client_id(app_client: TestClient) -> None:
    assert app_client.get(conv_url("nope")).status_code == 422


def test_list_conversations_only_shows_own_conversations(app_client: TestClient) -> None:
    client_a = make_client(app_client, "clinic-a")
    client_b = make_client(app_client, "clinic-b")
    conv_a = make_conversation(app_client, client_a["id"], "cust_1")
    conv_b = make_conversation(app_client, client_b["id"], "cust_1")  # same customer id, other tenant

    list_a = app_client.get(conv_url(client_a["id"])).json()
    list_b = app_client.get(conv_url(client_b["id"])).json()
    assert [c["id"] for c in list_a["items"]] == [conv_a["id"]] and list_a["total"] == 1
    assert [c["id"] for c in list_b["items"]] == [conv_b["id"]] and list_b["total"] == 1


# =============================== retrieval =====================================
def test_get_conversation(app_client: TestClient) -> None:
    client = make_client(app_client)
    created = make_conversation(app_client, client["id"])
    response = app_client.get(conv_url(client["id"], created["id"]))
    assert response.status_code == 200
    assert response.json() == created


def test_get_nonexistent_conversation(app_client: TestClient) -> None:
    client = make_client(app_client)
    response = app_client.get(conv_url(client["id"], str(uuid.uuid4())))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_get_conversation_with_invalid_ids(app_client: TestClient) -> None:
    client = make_client(app_client)
    assert app_client.get(conv_url(client["id"], "not-a-uuid")).status_code == 422
    assert app_client.get(conv_url("not-a-uuid", str(uuid.uuid4()))).status_code == 422


def test_get_conversation_for_nonexistent_client(app_client: TestClient) -> None:
    client = make_client(app_client)
    created = make_conversation(app_client, client["id"])
    response = app_client.get(conv_url(str(uuid.uuid4()), created["id"]))
    assert response.status_code == 404


def test_conversation_remains_readable_after_client_is_deactivated(app_client: TestClient) -> None:
    client = make_client(app_client)
    created = make_conversation(app_client, client["id"])
    app_client.patch(f"{CLIENTS}/{client['id']}", json={"configuration_status": "inactive"})
    assert app_client.get(conv_url(client["id"], created["id"])).status_code == 200
    assert app_client.get(conv_url(client["id"])).json()["total"] == 1


# ============================ tenant isolation ===================================
def test_cannot_read_another_clients_conversation(app_client: TestClient) -> None:
    client_a = make_client(app_client, "clinic-a")
    client_b = make_client(app_client, "clinic-b")
    conv_a = make_conversation(app_client, client_a["id"])

    cross = app_client.get(conv_url(client_b["id"], conv_a["id"]))
    assert cross.status_code == 404
    assert cross.json()["error"]["code"] == "not_found"
    assert app_client.get(conv_url(client_a["id"], conv_a["id"])).status_code == 200  # owner still can


def test_cross_tenant_lookup_is_indistinguishable_from_a_missing_conversation(
    app_client: TestClient,
) -> None:
    """The 404 must not reveal that the conversation exists under another client."""
    client_a = make_client(app_client, "clinic-a")
    client_b = make_client(app_client, "clinic-b")
    conv_a = make_conversation(app_client, client_a["id"])

    cross = app_client.get(conv_url(client_b["id"], conv_a["id"]))
    missing_id = str(uuid.uuid4())
    missing = app_client.get(conv_url(client_b["id"], missing_id))
    assert cross.status_code == missing.status_code == 404
    assert cross.json()["error"]["message"] == f"Conversation {conv_a['id']} was not found."
    assert cross.json()["error"]["message"].replace(conv_a["id"], "X") == missing.json()["error"][
        "message"
    ].replace(missing_id, "X")


def test_cannot_read_another_clients_messages(app_client: TestClient, db_session: Session) -> None:
    client_a = make_client(app_client, "clinic-a")
    client_b = make_client(app_client, "clinic-b")
    conv_a = make_conversation(app_client, client_a["id"])
    seed_messages(db_session, conv_a["id"], 3, text="tenant-a-private")

    cross = app_client.get(conv_url(client_b["id"], conv_a["id"], "/messages"))
    assert cross.status_code == 404
    assert "tenant-a-private" not in cross.text
    assert app_client.get(conv_url(client_a["id"], conv_a["id"], "/messages")).json()["total"] == 3


def test_service_never_fetches_a_conversation_by_id_alone(app_client: TestClient, db_session: Session) -> None:
    from app.core.exceptions import NotFoundError

    client_a = make_client(app_client, "clinic-a")
    client_b = make_client(app_client, "clinic-b")
    conv_a = make_conversation(app_client, client_a["id"])
    service = ConversationService(db_session)

    assert service.get(uuid.UUID(client_a["id"]), uuid.UUID(conv_a["id"])).id == uuid.UUID(conv_a["id"])
    with pytest.raises(NotFoundError):
        service.get(uuid.UUID(client_b["id"]), uuid.UUID(conv_a["id"]))
    with pytest.raises(NotFoundError):
        service.list_messages(uuid.UUID(client_b["id"]), uuid.UUID(conv_a["id"]), limit=10, offset=0)


# ================================ messages =======================================
def test_messages_empty_for_new_conversation(app_client: TestClient) -> None:
    client = make_client(app_client)
    conv = make_conversation(app_client, client["id"])
    response = app_client.get(conv_url(client["id"], conv["id"], "/messages"))
    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_messages_are_returned_in_chronological_order_with_all_fields(
    app_client: TestClient, db_session: Session
) -> None:
    client = make_client(app_client)
    conv = make_conversation(app_client, client["id"])
    seed_messages(db_session, conv["id"], 4)

    body = app_client.get(conv_url(client["id"], conv["id"], "/messages")).json()
    assert body["total"] == 4
    assert [m["sequence"] for m in body["items"]] == [1, 2, 3, 4]
    assert [m["content"] for m in body["items"]] == [f"synthetic text {i}" for i in (1, 2, 3, 4)]
    assert [m["direction"] for m in body["items"]] == ["inbound", "outbound", "inbound", "outbound"]
    first = body["items"][0]
    assert set(first) == {
        "id", "conversation_id", "sequence", "sender_type", "direction", "content", "intent",
        "intent_confidence", "intent_method", "provider", "model", "processing_status",
        "failure_category", "in_reply_to_id", "idempotency_key", "created_at",
    }  # fmt: skip
    assert first["conversation_id"] == conv["id"]
    assert first["sender_type"] == "customer" and first["processing_status"] == "processed"
    assert first["intent"] is None and first["provider"] is None


def test_messages_pagination(app_client: TestClient, db_session: Session) -> None:
    client = make_client(app_client)
    conv = make_conversation(app_client, client["id"])
    seed_messages(db_session, conv["id"], 5)
    url = conv_url(client["id"], conv["id"], "/messages")

    pages = [app_client.get(url, params={"limit": 2, "offset": o}).json() for o in (0, 2, 4)]
    assert [p["total"] for p in pages] == [5, 5, 5]
    assert [[m["sequence"] for m in p["items"]] for p in pages] == [[1, 2], [3, 4], [5]]
    assert pages[0]["limit"] == 2 and pages[1]["offset"] == 2
    assert app_client.get(url, params={"offset": 50}).json()["items"] == []


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "x"}])
def test_messages_reject_bad_pagination(app_client: TestClient, params: dict[str, Any]) -> None:
    client = make_client(app_client)
    conv = make_conversation(app_client, client["id"])
    response = app_client.get(conv_url(client["id"], conv["id"], "/messages"), params=params)
    assert response.status_code == 422


def test_messages_only_include_this_conversations_messages(
    app_client: TestClient, db_session: Session
) -> None:
    client = make_client(app_client)
    first = make_conversation(app_client, client["id"], "cust_a")
    second = make_conversation(app_client, client["id"], "cust_b")
    seed_messages(db_session, first["id"], 2, text="first-conv")
    seed_messages(db_session, second["id"], 3, text="second-conv")

    body = app_client.get(conv_url(client["id"], first["id"], "/messages")).json()
    assert body["total"] == 2
    assert all(m["content"].startswith("first-conv") for m in body["items"])


def test_messages_for_nonexistent_conversation_or_bad_ids(app_client: TestClient) -> None:
    client = make_client(app_client)
    assert app_client.get(conv_url(client["id"], str(uuid.uuid4()), "/messages")).status_code == 404
    assert app_client.get(conv_url(client["id"], "nope", "/messages")).status_code == 422
    assert app_client.get(conv_url(str(uuid.uuid4()), str(uuid.uuid4()), "/messages")).status_code == 404


def test_message_content_is_not_written_to_logs(
    app_client: TestClient, db_session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    client = make_client(app_client)
    conv = make_conversation(app_client, client["id"])
    seed_messages(db_session, conv["id"], 2, text="PRIVATE-MARKER-TEXT")
    with caplog.at_level(logging.DEBUG):
        response = app_client.get(conv_url(client["id"], conv["id"], "/messages"))
    assert "PRIVATE-MARKER-TEXT" in response.text  # the API does return it...
    assert "PRIVATE-MARKER-TEXT" not in caplog.text  # ...but logs never contain it


# ============================== error handling ===================================
def test_database_errors_are_not_exposed_on_reads(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(app_client)

    def boom(self: ConversationService, *args: Any, **kwargs: Any) -> None:
        raise SQLAlchemyError("secret internal detail: SELECT * FROM conversations")

    monkeypatch.setattr(ConversationService, "list_conversations", boom)
    response = app_client.get(conv_url(client["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "An internal error occurred."}}
    assert "secret" not in response.text


# ============================ Phase 1 contract kept =================================
def test_phase1_client_endpoints_are_unchanged(app_client: TestClient) -> None:
    paths = app_client.get("/openapi.json").json()["paths"]
    assert set(paths[CLIENTS]) == {"get", "post"}
    assert set(paths[f"{CLIENTS}/{{client_id}}"]) == {"get", "patch"}
    assert set(paths[f"{CLIENTS}/{{client_id}}/conversations"]) == {"get", "post"}
    assert "/health" in paths
    assert app_client.get("/health").json()["status"] == "ok"
