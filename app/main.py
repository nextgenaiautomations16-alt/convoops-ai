"""Application factory: wiring only, no business logic."""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api import clients, conversations, messages
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.db.database import get_db, init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    logger.info("startup app=%s version=%s env=%s", settings.app_name, settings.app_version, settings.app_env)
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="LLM Conversation Operations platform - Phase 1: client onboarding API.",
        lifespan=lifespan,
    )
    register_exception_handlers(app)

    @app.middleware("http")
    async def request_logging(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/health", tags=["system"], summary="Liveness and database connectivity check")
    def health(db: Session = Depends(get_db)) -> Response:
        try:
            db.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001 - never leak internals from a health probe
            logger.exception("health_check_failed")
            return JSONResponse(status_code=503, content={"status": "unavailable", "database": "down"})
        return JSONResponse(
            content={"status": "ok", "service": settings.app_name, "version": settings.app_version, "database": "up"}
        )

    app.include_router(clients.router, prefix=settings.api_v1_prefix)
    app.include_router(conversations.router, prefix=settings.api_v1_prefix)
    app.include_router(messages.router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
