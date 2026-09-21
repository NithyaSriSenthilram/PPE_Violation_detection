"""Uniform error handling.

Requirement (§28/§29): internal detail must never reach a production client.
Every unhandled exception is logged in full server-side with a request id, and
the client receives that id plus a generic message. In development the
exception text is included, because hiding it there just slows debugging down.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.config import settings
from backend.logging_conf import get_logger

logger = get_logger(__name__)


def install_error_handlers(app: FastAPI) -> None:
    """Attach the handlers to an app instance."""

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail if isinstance(exc.detail, str) else "Request failed",
                "detail": None,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Surface *which* field failed — that is not sensitive and it is what
        # the caller needs.
        problems = [
            {
                "field": ".".join(str(p) for p in error.get("loc", ())[1:]) or "body",
                "message": error.get("msg", "invalid value"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "Validation failed", "detail": problems},
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        request_id = str(uuid.uuid4())[:8]
        logger.error(
            "Database error [%s] on %s %s: %s",
            request_id, request.method, request.url.path, exc,
            exc_info=not settings.is_production,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "The database is unavailable. Please retry.",
                "detail": None if settings.is_production else str(exc)[:400],
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(uuid.uuid4())[:8]
        logger.error(
            "Unhandled error [%s] on %s %s",
            request_id, request.method, request.url.path, exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                # Never a traceback in production — see §28.
                "detail": None if settings.is_production else f"{type(exc).__name__}: {exc}",
                "request_id": request_id,
            },
        )
