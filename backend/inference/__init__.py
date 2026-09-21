"""AI inference: backend-agnostic detection.

Consumers import :class:`Detector`, :class:`Detection` and
:func:`resolve_detector`; nothing else should reach into a specific backend.
"""

from backend.inference.base import (
    BackendUnavailable,
    Detection,
    Detector,
    InferenceResult,
)
from backend.inference.registry import (
    active_backend_name,
    describe_backends,
    get_detector,
    inference_info,
    mojo_info,
    platform_info,
    reset_detector,
    resolve_detector,
)

__all__ = [
    "BackendUnavailable",
    "Detection",
    "Detector",
    "InferenceResult",
    "active_backend_name",
    "describe_backends",
    "get_detector",
    "inference_info",
    "mojo_info",
    "platform_info",
    "reset_detector",
    "resolve_detector",
]
