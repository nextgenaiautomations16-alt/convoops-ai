# ConvoOps AI

ConvoOps AI is a platform for operating and optimizing LLM-driven business conversations. **This repository contains Phase 1 only: the Client Onboarding and Configuration API.** It registers client businesses (synthetic data only in this repo), stores their configuration, and exposes it over a REST API.

Nothing here is deployed, and there are no SMS, LLM or analytics integrations yet. Those are later phases (see [Roadmap](#roadmap)).

## Architecture

```
HTTP request
   │
   ▼
app/api/clients.py        routing, status codes, query/path parsing
   │  (Depends)
   ▼
app/services/client_service.py   business rules: uniqueness, transactions, 404/409
   │
   ▼
app/models/client.py      SQLAlchemy 2.x ORM model + DB constraints
   │
   ▼
app/db/database.py        engine, session dependency, declarative Base
```

| Path | Responsibility |
|---|---|
| `app/main.py` | App factory: middleware, health route, router wiring. No business logic. |
| `app/core/config.py` | Environment-based settings (`pydantic-settings`). |
| `app/core/logging.py` | Structured `key=value` logging. |
| `app/core/exceptions.py` | Domain errors and the uniform error response format. |
| `app/db/database.py` | Engine/session management; `Base` shared by all future models. |
| `app/models/` | ORM models. |
| `app/schemas/` | Pydantic request/response schemas and validation. |
| `app/api/` | One router module per resource. |
| `app/services/` | One service class per domain area. |

Each future phase adds its own `models/`, `schemas/`, `api/` and `services/` modules next to `client` and references `clients.id`; no existing code needs to change.

### Design decisions

- **Slug is immutable.** It is the stable external identifier later phases (webhooks, knowledge bases) will reference, so `PATCH` rejects it.
- **Uniqueness is enforced twice.** The service pre-checks for friendly 409 messages; database constraints (unique slug, unique case-insensitive business name) are the final guard against concurrent requests.
- **Activation is a status.** `configuration_status` is `draft` → `active` ⇄ `inactive`, changed through `PATCH`. Any transition between the three values is allowed.
- **Errors never leak internals.** Database and unexpected errors are logged server-side and returned as a generic `500 internal_error`.
- **Invalid client IDs return 422**, because the path parameter is typed as a UUID. A well-formed UUID that does not exist returns 404.
- **Schema creation uses `create_all` at startup.** Adopt Alembic once the schema starts changing in Phase 2.

## Installation

Requires Python 3.10+ (developed and tested on 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

### Configuration

All settings come from environment variables or `.env` (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./convoops.db` | SQLAlchemy URL |
| `APP_ENV` | `development` | Environment name |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DB_ECHO` | `false` | Log SQL statements |
| `DEFAULT_PAGE_SIZE` / `MAX_PAGE_SIZE` | `20` / `100` | Pagination limits |

**Switching to PostgreSQL:** install a driver (`pip install "psycopg[binary]"`) and set
`DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/convoops`. No code changes are needed. The model uses portable types only (`Uuid`, `String`, `Text`, timezone-aware timestamps). Only SQLite was exercised in this repo's tests.

## Run the server

```bash
uvicorn app.main:app --reload
```

The SQLite file and tables are created on startup. Interactive docs: <http://127.0.0.1:8000/docs>.

## Run the tests

```bash
python -m pytest -q
```

Each test uses its own in-memory SQLite database, so the tests never touch `convoops.db`.

## API

Base path: `/api/v1`. All errors share this shape:

```json
{"error": {"code": "conflict", "message": "…", "details": [{"field": "…", "message": "…", "type": "…"}]}}
```

`details` appears only on `422 validation_error`.

| Method | Path | Success | Errors |
|---|---|---|---|
| GET | `/health` | 200 (503 if the DB is unreachable) | – |
| POST | `/api/v1/clients` | 201 + `Location` header | 409 duplicate slug/name, 422 validation |
| GET | `/api/v1/clients?limit=&offset=&configuration_status=` | 200 | 422 bad pagination |
| GET | `/api/v1/clients/{client_id}` | 200 | 404, 422 invalid UUID |
| PATCH | `/api/v1/clients/{client_id}` | 200 | 404, 409, 422 |

**Client fields:** `business_name` (≤200, unique ignoring case), `slug` (2–64 chars, lowercase letters/digits/single hyphens, unique), `industry`, `timezone` (IANA name), `business_description`, `contact_email` (optional), `configuration_status` (`draft` default, `active`, `inactive`), plus read-only `id`, `created_at`, `updated_at`.

Listing is ordered newest first and returns `{items, total, limit, offset}`. `PATCH` changes only the fields you send; `contact_email: null` clears the email, other fields cannot be null, and an empty body is rejected.

### Examples

Health:
```bash
curl -s http://127.0.0.1:8000/health
```
```json
{"status":"ok","service":"ConvoOps AI","version":"0.1.0","database":"up"}
```

Create a client (`201 Created`):
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/clients \
  -H 'Content-Type: application/json' \
  -d '{"business_name":"Sunrise Dental Studio","slug":"sunrise-dental","industry":"Dental","timezone":"America/Chicago","business_description":"Synthetic family dental practice used for demos.","contact_email":"frontdesk@example.com"}'
```
```json
{"id":"63f70c80-3787-43c8-851f-054b17fb8d80","business_name":"Sunrise Dental Studio","slug":"sunrise-dental","industry":"Dental","timezone":"America/Chicago","business_description":"Synthetic family dental practice used for demos.","contact_email":"frontdesk@example.com","configuration_status":"draft","created_at":"2026-10-31T15:19:22.597323Z","updated_at":"2026-10-31T15:19:22.597323Z"}
```

Duplicate slug (`409`):
```json
{"error":{"code":"conflict","message":"A client with slug 'sunrise-dental' already exists."}}
```

Missing fields (`422`):
```json
{"error":{"code":"validation_error","message":"Request validation failed.","details":[{"field":"body.slug","message":"Field required","type":"missing"}, …]}}
```

List with pagination:
```bash
curl -s "http://127.0.0.1:8000/api/v1/clients?limit=10&offset=0"
```
```json
{"items":[{"id":"63f70c80-…","slug":"sunrise-dental", "…":"…"}],"total":1,"limit":10,"offset":0}
```

Get one client:
```bash
curl -s http://127.0.0.1:8000/api/v1/clients/63f70c80-3787-43c8-851f-054b17fb8d80
```

Activate (use `"inactive"` to deactivate):
```bash
curl -s -X PATCH http://127.0.0.1:8000/api/v1/clients/63f70c80-3787-43c8-851f-054b17fb8d80 \
  -H 'Content-Type: application/json' -d '{"configuration_status":"active"}'
```
The response contains `"configuration_status":"active"` and a newer `updated_at`.

Not found (`404`):
```json
{"error":{"code":"not_found","message":"Client 00000000-0000-4000-8000-000000000000 was not found."}}
```

Invalid ID (`422`, e.g. `/api/v1/clients/abc`):
```json
{"error":{"code":"validation_error","message":"Request validation failed.","details":[{"field":"path.client_id","message":"Input should be a valid UUID, invalid length: found 3","type":"uuid_parsing"}]}}
```

## Roadmap

Phase 1 (this repo): client onboarding API. Planned, **not implemented**: 2 AI conversation engine · 3 SMS webhook integrations · 4 client-specific knowledge bases · 5 prompt versioning and optimization · 6 conversation monitoring and failure detection · 7 AI evaluation and QA · 8 conversion analytics.
