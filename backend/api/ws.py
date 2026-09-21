"""WebSocket endpoint for realtime updates.

One socket carries every stream the dashboard needs — detections, events,
camera status and job progress — because a browser holding six sockets to the
same origin is worse for everyone. Clients narrow the feed with query
parameters or a `subscribe` message.

Design notes:

* The socket is **write-mostly**. A reader task drains inbound frames so ping/
  pong and close handshakes are processed, but the only commands accepted are
  `ping` and `subscribe`.
* Back-pressure is handled in :mod:`backend.events.bus` by dropping stale
  messages, never by blocking the pipeline that produced them.
* Authentication, when enabled, is by `?api_key=` because browsers cannot set
  headers on a WebSocket handshake.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from backend.config import settings
from backend.events.bus import Subscription, get_bus
from backend.logging_conf import get_logger

logger = get_logger(__name__)
router = APIRouter()

#: Topics a client may subscribe to.
VALID_TOPICS = {"detections", "event", "camera_status", "job_progress", "stats"}

#: Server-side keepalive. Well under typical proxy idle timeouts.
PING_INTERVAL = 25.0


def _parse_set(value: str | None, allowed: set[str] | None = None) -> set[str] | None:
    """Parse a comma-separated filter, returning None for 'everything'."""
    if not value:
        return None
    items = {v.strip() for v in value.split(",") if v.strip()}
    if allowed is not None:
        items &= allowed
    return items or None


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    topics: str | None = Query(
        None, description="Comma-separated topics; omit for all"
    ),
    cameras: str | None = Query(
        None, description="Comma-separated camera ids; omit for all"
    ),
    api_key: str | None = Query(None, description="Required when API_KEY is set"),
) -> None:
    if settings.api_key:
        import hmac

        if not api_key or not hmac.compare_digest(api_key, settings.api_key):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await websocket.accept()
    bus = get_bus()
    bus.bind_loop(asyncio.get_running_loop())

    subscription = bus.subscribe(
        topics=_parse_set(topics, VALID_TOPICS), camera_ids=_parse_set(cameras)
    )

    from backend.inference.registry import active_backend_name

    await websocket.send_json(
        {
            "type": "hello",
            "payload": {
                "app": settings.app_name,
                "version": settings.app_version,
                "backend": active_backend_name(),
                "topics": sorted(subscription.topics or VALID_TOPICS),
                "cameras": sorted(subscription.camera_ids or []) or "all",
                "ping_interval": PING_INTERVAL,
            },
            "ts": datetime.now(UTC).isoformat(),
        }
    )

    reader = asyncio.create_task(_read_commands(websocket, subscription))
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    subscription.queue.get(), timeout=PING_INTERVAL
                )
            except TimeoutError:
                # Idle keepalive so intermediaries do not close the socket.
                await websocket.send_json(
                    {
                        "type": "pong",
                        "payload": {"idle": True},
                        "ts": datetime.now(UTC).isoformat(),
                    }
                )
                continue
            await websocket.send_json(message)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except RuntimeError:
        # Socket closed underneath us mid-send.
        pass
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
    finally:
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reader
        bus.unsubscribe(subscription)


async def _read_commands(websocket: WebSocket, subscription: Subscription) -> None:
    """Drain inbound frames and handle the small command set."""
    while True:
        try:
            data: Any = await websocket.receive_json()
        except (WebSocketDisconnect, asyncio.CancelledError):
            return
        except Exception:
            # Non-JSON payload: ignore it rather than tearing down the socket.
            continue

        if not isinstance(data, dict):
            continue
        action = data.get("action")

        if action == "ping":
            with contextlib.suppress(Exception):
                await websocket.send_json(
                    {
                        "type": "pong",
                        "payload": {},
                        "ts": datetime.now(UTC).isoformat(),
                    }
                )
        elif action == "subscribe":
            topics = data.get("topics")
            cameras = data.get("cameras")
            if isinstance(topics, list):
                narrowed = {str(t) for t in topics} & VALID_TOPICS
                subscription.topics = narrowed or None
            if isinstance(cameras, list):
                subscription.camera_ids = {str(c) for c in cameras} or None
