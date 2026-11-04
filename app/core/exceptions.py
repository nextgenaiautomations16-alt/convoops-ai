"""Domain exceptions and the API error contract.

Every error response has the shape:
    {"error": {"code": "...", "message": "...", "details": [...]}}
Internal exceptions (database errors, bugs) are logged but never returned.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for errors that are safe to show to API consumers."""

    status_code = 400
    code = "bad_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


def _error_response(
    status_code: int, code: str, message: str, details: list[dict[str, Any]] | None = None
) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status_code, content={"error": body})


async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    return _error_response(exc.status_code, exc.code, exc.message)


async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {
            "field": ".".join(str(part) for part in err["loc"]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return _error_response(422, "validation_error", "Request validation failed.", details)


async def _handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = "not_found" if exc.status_code == 404 else "http_error"
    return _error_response(exc.status_code, code, str(exc.detail))


async def _handle_database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.error(
        "database_error method=%s path=%s error_type=%s",
        request.method,
        request.url.path,
        type(exc).__name__,
        exc_info=exc,
    )
    return _error_response(500, "internal_error", "An internal error occurred.")


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_error method=%s path=%s error_type=%s",
        request.method,
        request.url.path,
        type(exc).__name__,
        exc_info=exc,
    )
    return _error_response(500, "internal_error", "An internal error occurred.")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(SQLAlchemyError, _handle_database_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)
