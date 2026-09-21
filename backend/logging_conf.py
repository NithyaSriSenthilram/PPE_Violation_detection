"""Structured logging for SentinelVision AI.

Every record carries `timestamp`, `level`, `module` and — where the caller
supplies them — `camera`, `event` and `person`. Two renderers are available:
a colourised console format for development and line-delimited JSON for
production log shipping (`LOG_JSON=true`).

Also provides :func:`throttled` so per-frame code paths can log without
flooding: identical keys are emitted at most once per interval.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Mapping
from typing import Any

from backend.config import settings

# Extra fields promoted onto the record by :class:`ContextFilter`.
_CONTEXT_FIELDS = ("camera", "event", "person", "backend", "job")

# A custom level between WARNING and ERROR for security events, so operators
# can grep/route alerts without parsing message text.
ALERT_LEVEL = 35
logging.addLevelName(ALERT_LEVEL, "ALERT")


class ContextFilter(logging.Filter):
    """Guarantee the context attributes exist on every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for field in _CONTEXT_FIELDS:
            if not hasattr(record, field):
                setattr(record, field, None)
        return True


class ConsoleFormatter(logging.Formatter):
    """Human-readable, aligned console output."""

    COLOURS = {
        "DEBUG": "\033[38;5;244m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ALERT": "\033[38;5;201m",
        "ERROR": "\033[38;5;196m",
        "CRITICAL": "\033[1;38;5;196m",
    }
    RESET = "\033[0m"

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.use_colour = use_colour and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, self.datefmt)
        level = record.levelname
        tag = f"[{level}]"
        if self.use_colour:
            tag = f"{self.COLOURS.get(level, '')}{tag}{self.RESET}"

        bits: list[str] = []
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value:
                bits.append(f"{field}={value}")
        context = f" ({', '.join(bits)})" if bits else ""

        module = record.name.replace("backend.", "")
        line = f"{ts} {tag:<20} {module:<26} {record.getMessage()}{context}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(record.created)
            )
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def configure_logging(level: str | None = None, json_output: bool | None = None) -> None:
    """Install handlers on the root logger. Idempotent."""
    global _configured
    if _configured:
        return

    resolved_level = (level or settings.log_level).upper()
    use_json = settings.log_json if json_output is None else json_output

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if use_json else ConsoleFormatter())
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)

    # Third-party noise reduction — uvicorn duplicates access logs we do not
    # need at INFO, and multipart logs every chunk boundary at DEBUG.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("python_multipart").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Fetch a configured logger."""
    configure_logging()
    return logging.getLogger(name)


def log_alert(
    logger: logging.Logger, message: str, **context: Any
) -> None:  # pragma: no cover - thin wrapper
    """Emit at the custom ALERT level, e.g. on a confirmed security event."""
    logger.log(ALERT_LEVEL, message, extra=context)


# ── Throttling ────────────────────────────────────────────────────────────
_last_emit: dict[str, float] = {}


def throttled(
    logger: logging.Logger,
    key: str,
    message: str,
    *,
    level: int = logging.INFO,
    interval: float = 10.0,
    context: Mapping[str, Any] | None = None,
) -> bool:
    """Log `message` at most once per `interval` seconds for a given `key`.

    Used on per-frame paths (decode failures, dropped frames, model warnings)
    so a persistent condition produces a readable trickle, not thousands of
    identical lines. Returns True when the record was emitted.
    """
    now = time.monotonic()
    previous = _last_emit.get(key)
    if previous is not None and (now - previous) < interval:
        return False
    _last_emit[key] = now
    logger.log(level, message, extra=dict(context or {}))
    return True


def reset_throttle() -> None:
    """Clear throttle state (tests)."""
    _last_emit.clear()
