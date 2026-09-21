"""HTTP and WebSocket API.

Routers are aggregated here so `main.py` mounts one object and the URL prefix
lives in exactly one place.
"""

from fastapi import APIRouter

from backend.api import (
    cameras,
    events,
    evidence,
    health,
    statistics,
    videos,
    ws,
    zones,
)

#: Everything under /api
api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(cameras.router)
api_router.include_router(zones.router)
api_router.include_router(events.router)
api_router.include_router(statistics.router)
api_router.include_router(videos.router)
api_router.include_router(evidence.router)

#: WebSocket lives at /ws (no /api prefix — it is not REST)
ws_router = ws.router

__all__ = ["api_router", "ws_router"]
