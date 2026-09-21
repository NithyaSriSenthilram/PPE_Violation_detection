"""Event taxonomy: types, severities, statuses and their default mapping.

Extension point — adding a detection category means adding one
:class:`EventType` member and one row in :data:`DEFAULT_SEVERITY`. Nothing
else in the pipeline, API or database needs to change: severities are stored
as strings, and the frontend renders unknown types from their label.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class EventType(StrEnum):
    """Detection categories the system can raise."""

    PPE_VIOLATION = "PPE_VIOLATION"
    MISSING_HELMET = "MISSING_HELMET"
    MISSING_VEST = "MISSING_VEST"
    RESTRICTED_AREA = "RESTRICTED_AREA"
    LOITERING = "LOITERING"
    ABNORMAL_MOVEMENT = "ABNORMAL_MOVEMENT"
    POSSIBLE_FALL = "POSSIBLE_FALL"
    CROWD_ANOMALY = "CROWD_ANOMALY"


class Severity(StrEnum):
    """Alert severity ladder."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EventStatus(StrEnum):
    """Operator workflow state for an event."""

    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


SEVERITY_ORDER: Final[dict[str, int]] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}

#: Default severity per event type. Overridable at runtime via the
#: `severity_overrides` system setting (see backend/events/engine.py), so
#: sites can promote e.g. MISSING_VEST to HIGH without a code change.
DEFAULT_SEVERITY: Final[dict[str, str]] = {
    EventType.PPE_VIOLATION: Severity.HIGH,
    EventType.MISSING_HELMET: Severity.HIGH,
    EventType.MISSING_VEST: Severity.MEDIUM,
    EventType.RESTRICTED_AREA: Severity.HIGH,
    EventType.LOITERING: Severity.MEDIUM,
    EventType.ABNORMAL_MOVEMENT: Severity.LOW,
    EventType.POSSIBLE_FALL: Severity.HIGH,
    EventType.CROWD_ANOMALY: Severity.MEDIUM,
}

#: Human-readable labels used in log lines, descriptions and the UI.
EVENT_LABELS: Final[dict[str, str]] = {
    EventType.PPE_VIOLATION: "PPE Violation",
    EventType.MISSING_HELMET: "Missing Helmet",
    EventType.MISSING_VEST: "Missing Safety Vest",
    EventType.RESTRICTED_AREA: "Restricted Area Intrusion",
    EventType.LOITERING: "Loitering",
    EventType.ABNORMAL_MOVEMENT: "Abnormal Movement",
    EventType.POSSIBLE_FALL: "Possible Fall",
    EventType.CROWD_ANOMALY: "Crowd Anomaly",
}

#: Events whose output is an inference, not a determination. The UI shows an
#: advisory notice for these and wording stays deliberately non-committal —
#: `POSSIBLE_FALL` is an AI detection, never a medical or legal conclusion.
ADVISORY_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        EventType.POSSIBLE_FALL,
        EventType.ABNORMAL_MOVEMENT,
    }
)

ADVISORY_NOTICE: Final[str] = (
    "AI-generated detection. Requires human verification; not a diagnosis "
    "or a determination of intent."
)


def label_for(event_type: str) -> str:
    """Human label for an event type, tolerant of future/unknown values."""
    return EVENT_LABELS.get(event_type, event_type.replace("_", " ").title())


def default_severity(event_type: str) -> str:
    """Fallback severity for an event type not present in the override map."""
    return DEFAULT_SEVERITY.get(event_type, Severity.MEDIUM)


def is_advisory(event_type: str) -> bool:
    """True when results of this type must be presented as advisory."""
    return event_type in ADVISORY_EVENT_TYPES
