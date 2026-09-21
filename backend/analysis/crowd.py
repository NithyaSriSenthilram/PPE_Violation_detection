"""Crowd density and anomaly detection.

Two independent signals per region:

* **Absolute threshold** — head count above the configured limit.
* **Baseline deviation** — count well above the rolling average for that
  region, which catches a gathering forming somewhere that is normally
  sparse even when it never reaches the absolute limit.

Regions are `crowd`-type zones when any are configured; otherwise the whole
frame is treated as one region, so crowd monitoring works out of the box on a
camera with no zones drawn.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from backend.analysis.base import Analyser, EventCandidate, FrameContext
from backend.analysis.zones import zones_for_points
from backend.config import settings
from backend.events.types import EventType

#: Sentinel region id used when no crowd zones exist.
WHOLE_FRAME = "__frame__"

#: Multiple of the rolling baseline that counts as an anomalous spike.
BASELINE_MULTIPLIER = 2.5

#: Minimum count before a baseline spike is worth reporting — stops a jump
#: from 1 person to 3 registering as a crowd.
MIN_SPIKE_COUNT = 6

#: Samples required before the baseline is trusted.
MIN_BASELINE_SAMPLES = 20


@dataclass
class _RegionState:
    #: (timestamp, count) samples inside the baseline window.
    samples: deque = field(default_factory=lambda: deque(maxlen=4096))
    alerted: bool = False
    peak: int = 0


class CrowdAnalyser(Analyser):
    """Raises CROWD_ANOMALY on absolute or relative crowding."""

    name = "crowd"

    def __init__(
        self,
        default_threshold: int | None = None,
        baseline_window: float | None = None,
    ) -> None:
        self.default_threshold = (
            settings.crowd_threshold if default_threshold is None else default_threshold
        )
        self.baseline_window = (
            settings.crowd_baseline_window if baseline_window is None else baseline_window
        )
        self._regions: dict[str, _RegionState] = {}

    def reset(self) -> None:
        self._regions.clear()

    def counts(self) -> dict[str, int]:
        """Latest count per region — surfaced to the dashboard."""
        return {
            region: (state.samples[-1][1] if state.samples else 0)
            for region, state in self._regions.items()
        }

    def baseline(self, region: str) -> float:
        state = self._regions.get(region)
        if not state or len(state.samples) < MIN_BASELINE_SAMPLES:
            return 0.0
        return float(np.mean([c for _, c in state.samples]))

    def analyse(self, context: FrameContext) -> list[EventCandidate]:
        crowd_zones = [
            z
            for z in context.zones
            if z.enabled and z.is_valid and z.zone_type == "crowd"
        ]

        # Build {region_id: (count, zone_or_None)}.
        regions: dict[str, tuple[int, object]] = {}
        if crowd_zones and context.tracks:
            points = np.array([t.foot_point for t in context.tracks], dtype=np.float32)
            memberships = zones_for_points(
                points, crowd_zones, context.frame_width, context.frame_height
            )
            tally: dict[str, int] = {z.zone_id: 0 for z in crowd_zones}
            for zone_ids in memberships:
                for zone_id in zone_ids:
                    tally[zone_id] += 1
            for zone in crowd_zones:
                regions[zone.zone_id] = (tally[zone.zone_id], zone)
        elif crowd_zones:
            for zone in crowd_zones:
                regions[zone.zone_id] = (0, zone)
        else:
            regions[WHOLE_FRAME] = (len(context.tracks), None)

        now = context.timestamp
        candidates: list[EventCandidate] = []

        for region_id, (count, zone) in regions.items():
            state = self._regions.setdefault(region_id, _RegionState())
            state.samples.append((now, count))
            state.peak = max(state.peak, count)
            # Trim outside the baseline window.
            while state.samples and now - state.samples[0][0] > self.baseline_window:
                state.samples.popleft()

            threshold = (
                zone.crowd_threshold
                if zone is not None and getattr(zone, "crowd_threshold", None)
                else self.default_threshold
            )
            baseline = self.baseline(region_id)

            over_absolute = count >= threshold
            over_baseline = (
                baseline > 0
                and count >= MIN_SPIKE_COUNT
                and count >= baseline * BASELINE_MULTIPLIER
            )

            if not (over_absolute or over_baseline):
                # Re-arm once the crowd meaningfully disperses.
                if count < threshold * 0.7:
                    state.alerted = False
                continue
            if state.alerted:
                continue
            state.alerted = True

            region_name = zone.name if zone is not None else "whole frame"
            trigger = "absolute_threshold" if over_absolute else "baseline_deviation"
            if over_absolute:
                description = (
                    f"{count} people detected in '{region_name}' "
                    f"(threshold {threshold})"
                )
            else:
                description = (
                    f"{count} people detected in '{region_name}' — "
                    f"{count / baseline:.1f}x the recent average of {baseline:.1f}"
                )

            candidates.append(
                EventCandidate(
                    event_type=EventType.CROWD_ANOMALY,
                    confidence=min(
                        0.95,
                        0.55
                        + (
                            (count / threshold - 1) * 0.3
                            if over_absolute
                            else (count / max(baseline, 1) - BASELINE_MULTIPLIER) * 0.1
                        ),
                    ),
                    description=description,
                    zone_id=None if zone is None else zone.zone_id,
                    zone_name=region_name,
                    metadata={
                        "people_count": count,
                        "threshold": threshold,
                        "baseline": round(baseline, 2),
                        "peak": state.peak,
                        "trigger": trigger,
                        "samples": len(state.samples),
                        "region": region_id,
                    },
                )
            )
        return candidates
