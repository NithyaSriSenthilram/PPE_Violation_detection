"""In-process publish/subscribe bridge to the WebSocket layer.

The video pipeline runs on plain threads (OpenCV capture blocks, so it cannot
live on the event loop) while WebSockets live on asyncio. This bus is the
crossing point: :meth:`EventBus.publish` is thread-safe and non-blocking, and
each subscriber gets its own bounded queue.

Bounded queues matter. A browser tab that stops reading must not grow a queue
until the process dies, and a slow consumer must not throttle the capture
thread. When a subscriber's queue is full its oldest message is dropped and
the drop is counted, so the pipeline's frame rate is never held hostage to a
websocket.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from backend.logging_conf import get_logger, throttled

logger = get_logger(__name__)

#: Per-subscriber queue depth. Small: these are live updates, and a stale
#: detection frame is worthless.
QUEUE_SIZE = 64


# eq=False keeps the default identity-based __hash__: subscriptions are held
# in a set and are identified by object identity, not by field equality. A
# plain @dataclass sets __hash__ to None and makes them unhashable.
@dataclass(eq=False)
class Subscription:
    """One consumer's channel."""

    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_SIZE))
    topics: set[str] | None = None       # None = everything
    camera_ids: set[str] | None = None   # None = all cameras
    dropped: int = 0
    delivered: int = 0

    def wants(self, topic: str, camera_id: str | None) -> bool:
        if self.topics is not None and topic not in self.topics:
            return False
        return not (
            self.camera_ids is not None
            and camera_id is not None
            and camera_id not in self.camera_ids
        )


class EventBus:
    """Thread-safe fan-out to asyncio subscribers."""

    def __init__(self) -> None:
        self._subscribers: set[Subscription] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._published = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop that owns the subscriber queues."""
        self._loop = loop

    # ── subscription ─────────────────────────────────────────────────────
    def subscribe(
        self, topics: set[str] | None = None, camera_ids: set[str] | None = None
    ) -> Subscription:
        subscription = Subscription(topics=topics, camera_ids=camera_ids)
        with self._lock:
            self._subscribers.add(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._subscribers.discard(subscription)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # ── publishing ───────────────────────────────────────────────────────
    def publish(
        self, topic: str, payload: dict[str, Any], camera_id: str | None = None
    ) -> None:
        """Broadcast a message. Safe to call from any thread; never blocks."""
        with self._lock:
            targets = [s for s in self._subscribers if s.wants(topic, camera_id)]
        if not targets:
            return

        message = {
            "type": topic,
            "payload": payload,
            "ts": datetime.now(UTC).isoformat(),
        }
        self._published += 1

        loop = self._loop
        if loop is None or loop.is_closed():
            return
        # The loop can shut down between the check above and this call.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._deliver, targets, message, topic)

    def _deliver(
        self, targets: list[Subscription], message: dict[str, Any], topic: str
    ) -> None:
        """Runs on the event loop thread."""
        for subscription in targets:
            try:
                subscription.queue.put_nowait(message)
                subscription.delivered += 1
            except asyncio.QueueFull:
                # Drop the oldest so the newest still gets through.
                try:
                    subscription.queue.get_nowait()
                    subscription.queue.put_nowait(message)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
                subscription.dropped += 1
                throttled(
                    logger,
                    "ws-backpressure",
                    "A websocket subscriber is not keeping up; dropping stale "
                    "messages (this does not affect detection)",
                    interval=30.0,
                )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            subscribers = list(self._subscribers)
        return {
            "subscribers": len(subscribers),
            "published": self._published,
            "delivered": sum(s.delivered for s in subscribers),
            "dropped": sum(s.dropped for s in subscribers),
        }


_bus: EventBus | None = None


def get_bus() -> EventBus:
    """Process-wide event bus."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
