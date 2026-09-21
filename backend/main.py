"""Application entry point.

    uvicorn backend.main:app --reload                      # development
    uvicorn backend.main:app --host 0.0.0.0 --port $PORT   # production

Startup order matters and is explicit: configure logging, prepare the
database, resolve the inference backend (so the first request does not pay for
model load), calibrate the Mojo kernels, then bring up cameras. Camera startup
is last and is best-effort — an unreachable stream must not prevent the API
and dashboard from serving.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from backend.api import api_router, ws_router
from backend.api.errors import install_error_handlers
from backend.config import settings
from backend.logging_conf import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
AI-powered CCTV surveillance and safety-monitoring platform.

**Detection**: person detection and tracking, PPE compliance (helmet / safety
vest), restricted-area intrusion, loitering, abnormal movement, possible
falls, and crowd anomalies.

**Honesty about inference**: `GET /api/diagnostics` reports the inference
backend that is *actually* running, why every other backend is unavailable,
whether the Mojo kernels are in use (with the benchmark that decided it), and
whether PPE assessment is model-based or the colour heuristic. Nothing is
reported as active unless it is.

Events flagged `advisory: true` — possible falls and abnormal movement — are
AI inferences requiring human verification, not determinations.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bring the platform up and take it down cleanly."""
    configure_logging()
    logger.info(
        "%s v%s starting (%s)",
        settings.app_name, settings.app_version, settings.app_env,
    )

    # ── Database ─────────────────────────────────────────────────────────
    from backend.db.base import init_db

    try:
        init_db()
    except Exception:
        # Serve a useful error page rather than dying at import time.
        logger.exception("Database initialisation failed — API will report degraded")
    else:
        # Jobs run in-process; whatever was mid-flight when the previous
        # process died cannot be resumed and must not show as RUNNING forever.
        from backend.jobs.video_jobs import reconcile_interrupted_jobs

        try:
            reconcile_interrupted_jobs()
        except Exception as exc:
            logger.warning("Could not reconcile interrupted video jobs: %s", exc)

    # ── Event bus needs the running loop to hand messages to sockets ─────
    from backend.events.bus import get_bus

    get_bus().bind_loop(asyncio.get_running_loop())

    # ── Inference: resolve now so the first frame is not slowed by it ────
    from backend.inference.mojo.bridge import get_bridge
    from backend.inference.registry import active_backend_name, resolve_detector

    try:
        await asyncio.to_thread(resolve_detector)
        logger.info(
            "AI inference backend: %s", active_backend_name(),
            extra={"backend": active_backend_name()},
        )
    except Exception:
        logger.exception("Inference backend resolution failed")

    bridge = get_bridge()
    if bridge.active:
        active = [k for k, v in bridge.enabled.items() if v]
        logger.info("Mojo acceleration active for: %s", ", ".join(active))
    else:
        logger.info("Mojo acceleration inactive: %s", bridge.load_error or "numpy faster")

    # ── Severity overrides from the settings table ───────────────────────
    from backend.db.repository import load_severity_overrides
    from backend.events.engine import get_event_engine

    try:
        overrides = load_severity_overrides()
        if overrides:
            get_event_engine().set_severity_overrides(overrides)
            logger.info("Applied %d severity override(s)", len(overrides))
    except Exception as exc:
        logger.warning("Could not load severity overrides: %s", exc)

    # ── Cameras (best effort) ────────────────────────────────────────────
    from backend.video.manager import get_manager

    manager = get_manager()
    if settings.auto_start_cameras:
        try:
            results = await asyncio.to_thread(manager.start_all)
            failures = {k: v for k, v in results.items() if v.startswith(("failed", "error"))}
            if failures:
                logger.warning(
                    "%d camera(s) did not start; the platform is running without them",
                    len(failures),
                )
        except Exception:
            logger.exception("Camera startup failed")
    else:
        logger.info("Camera auto-start disabled (AUTO_START_CAMERAS=false)")

    logger.info(
        "Ready — API on http://%s:%d/api, docs at /api/docs",
        settings.app_host, settings.app_port,
    )

    try:
        yield
    finally:
        logger.info("Shutting down ...")
        from backend.events.evidence import get_evidence_writer
        from backend.jobs.video_jobs import get_job_runner

        for label, action in (
            ("cameras", manager.stop_all),
            ("video jobs", get_job_runner().shutdown),
            ("evidence writer", get_evidence_writer().shutdown),
        ):
            try:
                await asyncio.to_thread(action)
            except Exception as exc:
                logger.warning("Error stopping %s: %s", label, exc)
        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Build the application. A factory keeps tests able to make isolated apps."""
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS is explicit — never a wildcard, since credentials may be sent.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
        max_age=3600,
    )

    install_error_handlers(app)
    app.include_router(api_router)
    app.include_router(ws_router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/api/docs")

    @app.get("/api", include_in_schema=False)
    def api_index() -> JSONResponse:
        return JSONResponse(
            {
                "name": settings.app_name,
                "version": settings.app_version,
                "docs": "/api/docs",
                "health": "/api/health",
                "diagnostics": "/api/diagnostics",
                "websocket": "/ws",
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=not settings.is_production,
        log_config=None,  # our structured logging owns the output
    )
