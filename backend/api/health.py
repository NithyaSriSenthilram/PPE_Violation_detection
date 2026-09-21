"""Health and diagnostics endpoints.

`/api/health` is deliberately cheap — a load balancer can poll it. It reports
`degraded` (still HTTP 200) when the platform is running but not fully
functional, e.g. on synthetic detections or with no camera online.

`/api/diagnostics` is the honest, detailed picture required by §24: which
inference backend is *actually* active, why every other one is not, and
whether the Mojo kernels are genuinely in use.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from sqlalchemy import select, text

from backend.api.deps import DbSession, Manager, uptime_seconds
from backend.config import settings
from backend.db.models import Camera
from backend.events.bus import get_bus
from backend.events.engine import get_event_engine
from backend.inference.ppe import ppe_diagnostics
from backend.inference.registry import (
    active_backend_name,
    describe_backends,
    get_detector,
    inference_info,
    mojo_info,
    platform_info,
    video_output_info,
)
from backend.schemas import DiagnosticsOut, HealthOut

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthOut, summary="Liveness and readiness")
def health(session: DbSession, manager: Manager) -> HealthOut:
    database_ok = True
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        database_ok = False

    total = online = 0
    if database_ok:
        try:
            cameras = session.execute(select(Camera)).scalars().all()
            total = len(cameras)
            online = sum(1 for c in cameras if c.status == "online")
        except Exception:
            database_ok = False

    detector = get_detector()
    backend = active_backend_name()
    # Synthetic detections mean the platform is up but not doing real work.
    degraded = not database_ok or backend in ("none", "mock")

    return HealthOut(
        status="degraded" if degraded else "ok",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        uptime_seconds=round(uptime_seconds(), 1),
        database=database_ok,
        inference_backend=backend,
        model_loaded=bool(detector and detector.is_loaded),
        cameras_online=max(online, manager.online_count),
        cameras_total=total,
        timestamp=datetime.now(UTC),
    )


@router.get(
    "/diagnostics",
    response_model=DiagnosticsOut,
    summary="Full system diagnostics: real backend state, Mojo status, config",
)
def diagnostics(session: DbSession, manager: Manager) -> DiagnosticsOut:
    import sys

    database: dict[str, object] = {"url_scheme": settings.database_url.split("://", 1)[0]}
    try:
        session.execute(text("SELECT 1"))
        database["connected"] = True
        from backend.db.models import Event, VideoJob, Zone

        for label, column in (
            ("cameras", Camera.camera_id),
            ("zones", Zone.zone_id),
            ("events", Event.event_id),
            ("jobs", VideoJob.job_id),
        ):
            from sqlalchemy import func

            database[label] = int(session.execute(select(func.count(column))).scalar_one())
    except Exception as exc:
        database["connected"] = False
        database["error"] = str(exc)[:200]

    ppe = ppe_diagnostics()

    return DiagnosticsOut(
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        uptime_seconds=round(uptime_seconds(), 1),
        platform=platform_info(),
        python_version=sys.version.split()[0],
        database=database,
        inference=inference_info(),
        backends=describe_backends(),
        mojo=mojo_info(),
        ppe=ppe,
        video_output=video_output_info(),
        pipeline={
            **manager.info(),
            "event_engine": get_event_engine().stats(),
            "event_bus": get_bus().stats(),
        },
        config={
            "target_fps": settings.target_fps,
            "detect_every_n_frames": settings.detect_every_n_frames,
            "confidence_threshold": settings.confidence_threshold,
            "nms_iou_threshold": settings.nms_iou_threshold,
            "inference_size": [settings.inference_width, settings.inference_height],
            "ppe_model_path": str(settings.ppe_model_file),
            "ppe_confidence_threshold": settings.ppe_confidence_threshold,
            "ppe_allow_heuristic": settings.ppe_allow_heuristic,
            "required_ppe": settings.required_ppe_items,
            "loitering_threshold": settings.loitering_threshold,
            "running_threshold": settings.running_threshold,
            "crowd_threshold": settings.crowd_threshold,
            "event_cooldown_seconds": settings.event_cooldown_seconds,
            "clip_enabled": settings.clip_enabled,
            "clip_buffer_seconds": settings.clip_buffer_seconds,
            "max_upload_mb": settings.max_upload_mb,
            "api_key_required": bool(settings.api_key),
            "cors_origins": settings.cors_origin_list,
        },
        timestamp=datetime.now(UTC),
    )
