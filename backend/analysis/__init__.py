"""Behaviour analysis: turns tracks into event candidates.

Every rule is an :class:`~backend.analysis.base.Analyser`. Add a detection
category by writing one class and appending it to :func:`build_analysers` —
the pipeline, event engine, API and dashboard need no changes.
"""

from backend.analysis.base import Analyser, EventCandidate, FrameContext
from backend.analysis.crowd import CrowdAnalyser
from backend.analysis.fall import FallAnalyser
from backend.analysis.intrusion import IntrusionAnalyser
from backend.analysis.loitering import LoiteringAnalyser
from backend.analysis.movement import MovementAnalyser, measure_speed
from backend.analysis.ppe_rules import PPEAnalyser
from backend.analysis.zones import ResolvedZone, validate_polygon

#: Detection profiles for uploaded-video analysis (see §23 of the spec).
PROFILE_ANALYSERS: dict[str, tuple[str, ...]] = {
    "fast": ("intrusion", "crowd"),
    "standard": ("ppe", "intrusion", "loitering", "movement", "crowd"),
    "thorough": ("ppe", "intrusion", "loitering", "movement", "fall", "crowd"),
    "ppe_only": ("ppe",),
    "security_only": ("intrusion", "loitering", "movement", "fall", "crowd"),
}


def build_analysers(profile: str = "thorough") -> list[Analyser]:
    """Instantiate the analysers enabled by a detection profile."""
    enabled = PROFILE_ANALYSERS.get(profile, PROFILE_ANALYSERS["thorough"])
    factories = {
        "ppe": PPEAnalyser,
        "intrusion": IntrusionAnalyser,
        "loitering": LoiteringAnalyser,
        "movement": MovementAnalyser,
        "fall": FallAnalyser,
        "crowd": CrowdAnalyser,
    }
    return [factories[name]() for name in enabled if name in factories]


__all__ = [
    "Analyser",
    "CrowdAnalyser",
    "EventCandidate",
    "FallAnalyser",
    "FrameContext",
    "IntrusionAnalyser",
    "LoiteringAnalyser",
    "MovementAnalyser",
    "PPEAnalyser",
    "PROFILE_ANALYSERS",
    "ResolvedZone",
    "build_analysers",
    "measure_speed",
    "validate_polygon",
]
