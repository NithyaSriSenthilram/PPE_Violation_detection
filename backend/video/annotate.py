"""Detection overlay rendering.

Used for evidence snapshots and for the annotated output of uploaded-video
analysis. The live dashboard does **not** use this: it receives normalised
coordinates over the WebSocket and draws them in the browser, which keeps the
video path free of per-frame drawing work and lets the operator toggle layers
without re-encoding anything.

What gets drawn is the PPE, not the people. A rectangle around a whole body
carries one bit — *someone is here* — and at any distance it swallows the head
and torso the finding actually concerns. So each item is drawn where it lives:
the helmet on the head, the vest on the torso, boots at the feet. The person
box is still computed, tracked and associated against; it is simply not the
thing on screen (``SHOW_PERSON_BOXES=true`` brings it back for debugging).

The palette matches the frontend's severity colours so a snapshot and the UI
read the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from backend.analysis.zones import ResolvedZone
from backend.config import settings
from backend.inference.ppe import PPEAssessment, PPEItemStatus
from backend.inference.ppe_taxonomy import body_region_box
from backend.tracking import Track

# BGR, matching the frontend tokens.
COLOUR_OK = (129, 199, 132)          # green   — compliant
COLOUR_VIOLATION = (68, 68, 244)     # red     — PPE violation
COLOUR_WARNING = (0, 170, 245)       # amber   — advisory
COLOUR_TRACK = (222, 196, 125)       # cyan    — normal track
COLOUR_ZONE = (94, 63, 244)          # rose    — restricted zone
COLOUR_TEXT = (255, 255, 255)
COLOUR_PANEL = (28, 24, 20)

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: A box as drawn: integer pixels, ``(x1, y1, x2, y2)``.
Rect = tuple[int, int, int, int]

#: One line of a label stack: text, chip colour, text scale.
Line = tuple[str, tuple[int, int, int], float]


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    """'#rrggbb' → BGR tuple, falling back to the zone default."""
    try:
        text = value.lstrip("#")
        r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)
    except (ValueError, IndexError):
        return COLOUR_ZONE


def draw_zones(
    frame: np.ndarray, zones: Sequence[ResolvedZone], alpha: float = 0.18
) -> np.ndarray:
    """Draw filled, labelled zone polygons."""
    if not zones:
        return frame
    height, width = frame.shape[:2]
    overlay = frame.copy()

    for zone in zones:
        if not zone.enabled or not zone.is_valid:
            continue
        polygon = zone.pixel_polygon(width, height).astype(np.int32)
        colour = _hex_to_bgr(zone.colour)
        cv2.fillPoly(overlay, [polygon], colour)
        cv2.polylines(frame, [polygon], True, colour, 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Labels go on last so the fill never washes them out.
    for zone in zones:
        if not zone.enabled or not zone.is_valid:
            continue
        polygon = zone.pixel_polygon(width, height).astype(np.int32)
        anchor = polygon[polygon[:, 1].argmin()]
        _label(
            frame,
            f"{zone.name}  [{zone.zone_type.upper()}]",
            (int(anchor[0]), max(16, int(anchor[1]) - 8)),
            _hex_to_bgr(zone.colour),
            scale=0.45,
        )
    return frame


def _label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Text on a filled chip, clamped inside the frame."""
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = origin
    x = max(0, min(x, frame.shape[1] - text_w - 8))
    y = max(text_h + 6, min(y, frame.shape[0] - 4))
    cv2.rectangle(
        frame, (x, y - text_h - baseline - 2), (x + text_w + 8, y + 2), colour, -1
    )
    cv2.putText(
        frame, text, (x + 4, y - 2), FONT, scale, COLOUR_TEXT, thickness, cv2.LINE_AA
    )


# ══════════════════════════════════════════════════════════════════════════
#  Label placement
# ══════════════════════════════════════════════════════════════════════════
CHIP_PAD_X, CHIP_PAD_Y, CHIP_GAP = 5, 3, 3


def _line_metrics(text: str, scale: float) -> tuple[int, int, int]:
    """Chip width, chip height, and the cap height the text sits on."""
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, 1)
    return text_w + CHIP_PAD_X * 2, text_h + baseline + CHIP_PAD_Y * 2, text_h


def _stack_size(lines: Sequence[Line]) -> tuple[int, int]:
    """Bounding size of a label stack, before it is placed anywhere."""
    sizes = [_line_metrics(text, scale) for text, _, scale in lines]
    return (max((w for w, _, _ in sizes), default=0), sum(h for _, h, _ in sizes))


def _draw_stack(
    frame: np.ndarray, lines: Sequence[Line], top_left: tuple[int, int]
) -> None:
    """Draw a label stack from its top-left corner, one filled chip per line."""
    x, y = top_left
    for text, colour, scale in lines:
        chip_w, chip_h, text_h = _line_metrics(text, scale)
        cv2.rectangle(frame, (x, y), (x + chip_w, y + chip_h), colour, -1)
        cv2.putText(
            frame, text, (x + CHIP_PAD_X, y + CHIP_PAD_Y + text_h),
            FONT, scale, COLOUR_TEXT, 1, cv2.LINE_AA,
        )
        y += chip_h


def _as_rect(box: Sequence[float]) -> Rect:
    x1, y1, x2, y2 = (int(v) for v in box)
    return (x1, y1, x2, y2)


def _overlap(a: Rect, b: Rect) -> int:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0


class LabelPlacer:
    """Finds each label a spot: inside the frame, off the HUD, off other labels.

    A PPE label that lands on the telemetry panel, on an event banner, or on the
    label of the person standing next to its owner costs both readings. So every
    placed label is remembered and the next one is steered around it — above the
    box first, since that is where the eye looks for it, then below, then inside
    the top of the box, then beside it, then stacked clear.
    """

    def __init__(self, frame_shape: tuple[int, ...], reserved: Sequence[Rect] = ()) -> None:
        self.height, self.width = int(frame_shape[0]), int(frame_shape[1])
        self.taken: list[Rect] = [_as_rect(r) for r in reserved]

    def place(self, size: tuple[int, int], anchor: Rect) -> tuple[int, int]:
        """Top-left corner for a label of `size` belonging to box `anchor`."""
        w, h = size
        x1, y1, x2, y2 = anchor
        step = h + CHIP_GAP
        candidates: list[tuple[int, int]] = [
            (x1, y1 - h - CHIP_GAP),            # above — the preferred read
            (x1, y2 + CHIP_GAP),                # below
            (x1, y1 + CHIP_GAP),                # inside the top of the box
            (x2 + CHIP_GAP, y1),                # beside it, right
            (x1 - w - CHIP_GAP, y1),            # beside it, left
        ]
        candidates += [(x1, y1 - h - CHIP_GAP - k * step) for k in range(1, 4)]
        candidates += [(x1, y2 + CHIP_GAP + k * step) for k in range(1, 4)]

        # A candidate that only fits once it has been shoved back inside the
        # frame is not really that candidate any more — clamping "above" into
        # a frame with no room above it silently returns the box's own top
        # corner. Such candidates are considered only as a last resort.
        best: Rect | None = None
        best_overlap = 0
        for cx, cy in candidates:
            if not self._in_frame(cx, cy, w, h):
                continue
            rect = (int(cx), int(cy), int(cx) + w, int(cy) + h)
            overlap = sum(_overlap(rect, other) for other in self.taken)
            if overlap == 0:
                best = rect
                break
            if best is None or overlap < best_overlap:
                best, best_overlap = rect, overlap

        if best is None:
            # Nowhere clean and nowhere that fits: clamp the least-bad
            # candidate into the frame rather than drawing off the edge.
            clamped = [self._fit(cx, cy, w, h) for cx, cy in candidates]
            best = min(
                clamped,
                key=lambda rect: sum(_overlap(rect, other) for other in self.taken),
            )
        self.taken.append(best)
        return (best[0], best[1])

    def _in_frame(self, x: int, y: int, w: int, h: int) -> bool:
        return x >= 0 and x + w <= self.width and y >= 0 and y + h <= self.height

    def _fit(self, x: int, y: int, w: int, h: int) -> Rect:
        """Clamp a candidate so the whole label stays inside the frame."""
        x = max(0, min(int(x), self.width - w))
        y = max(0, min(int(y), self.height - h))
        return (x, y, x + w, y + h)


# ══════════════════════════════════════════════════════════════════════════
#  PPE box geometry
# ══════════════════════════════════════════════════════════════════════════
#: How much taller than its body region a model's own PPE box may be — as a
#: fraction of person height — before it is treated as un-localised.
#:
#: Not a guess: measured against the installed model, whose helmet boxes run
#: 0.09–0.15 of person height and whose vest / no-vest boxes run 0.38–0.55.
#: Genuine boxes clear this comfortably; the failure mode it catches is a model
#: that answers "no vest" with a copy of the person box, which lands at ~1.0.
REGION_HEIGHT_SLACK = 0.25

#: Minimum drawable side. Below this a rectangle is a smudge, not a box.
MIN_BOX_SIDE = 3


def _intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def clip_to_frame(
    box: tuple[float, float, float, float], frame_shape: tuple[int, ...]
) -> Rect | None:
    """Integer box clamped inside the frame, or None if nothing is left.

    Scalar `min`/`max` per coordinate — never fancy-indexed array clipping,
    which is what silently mangled boxes here once before.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1 = max(0, min(int(round(box[0])), width - 1))
    y1 = max(0, min(int(round(box[1])), height - 1))
    x2 = max(0, min(int(round(box[2])), width - 1))
    y2 = max(0, min(int(round(box[3])), height - 1))
    if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        return None
    return (x1, y1, x2, y2)


def resolve_ppe_box(
    item: str,
    status: PPEItemStatus,
    person_bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, ...],
) -> tuple[Rect, bool] | None:
    """Where one PPE item is drawn, and whether the model itself said so.

    Three cases, in order of how much they can be trusted:

    * the detector gave a box and it is the size of the thing it names — draw
      exactly that, untouched;
    * the detector gave a box the size of the whole person (some models answer
      an absence class that way) — keep its horizontal placement but constrain
      it to the body region, so "no vest" lands on a torso rather than a body;
    * the detector gave no box at all — an absence deduced from omission, or a
      heuristic estimate — so the body region *is* the box.

    Returns ``(rect, from_model)``; the flag drives the dashed outline that
    marks a derived box as derived. ``None`` when nothing is drawable.
    """
    region = body_region_box(person_bbox, item)
    box = status.bbox
    from_model = False

    if box is not None:
        person_height = max(float(person_bbox[3]) - float(person_bbox[1]), 1e-6)
        region_fraction = (region[3] - region[1]) / person_height
        box_fraction = (float(box[3]) - float(box[1])) / person_height
        if box_fraction <= region_fraction + REGION_HEIGHT_SLACK:
            from_model = True
        else:
            constrained = _intersect(
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])), region
            )
            box = constrained if constrained is not None else region
    else:
        box = region

    rect = clip_to_frame(box, frame_shape)
    return None if rect is None else (rect, from_model)


def _dashed_rect(
    frame: np.ndarray, rect: Rect, colour: tuple[int, int, int], thickness: int
) -> None:
    """A dashed outline: the box was derived from the person, not detected."""
    x1, y1, x2, y2 = rect
    dash = max(6, (x2 - x1) // 8)
    for x in range(x1, x2, dash * 2):
        end = min(x + dash, x2)
        cv2.line(frame, (x, y1), (end, y1), colour, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y2), (end, y2), colour, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash * 2):
        end = min(y + dash, y2)
        cv2.line(frame, (x1, y), (x1, end), colour, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y), (x2, end), colour, thickness, cv2.LINE_AA)


def _item_colour(item: str, present: bool | None) -> tuple[int, int, int]:
    """Green worn, red missing-and-required, amber missing-but-advisory."""
    if present:
        return COLOUR_OK
    return (
        COLOUR_VIOLATION if item in settings.required_ppe_items else COLOUR_WARNING
    )


def _item_text(item: str, status: PPEItemStatus) -> str:
    prefix = "" if status.present else "NO "
    return f"{prefix}{item.upper()} {status.confidence * 100:.0f}%"


def draw_ppe_items(
    frame: np.ndarray,
    track: Track,
    assessment: PPEAssessment,
    placer: LabelPlacer,
    *,
    show_track_id: bool = False,
) -> int:
    """Draw every determined PPE item for one person. Returns how many.

    Only the items with a verdict are drawn: *undetermined* is not a finding,
    and a box asserting one would be a lie about what the detector saw.
    """
    if assessment.method == "disabled":
        return 0

    width = frame.shape[1]
    scale = max(0.42, min(0.70, width / 2000.0))
    thickness = max(2, round(width / 960.0))

    drawn: list[tuple[Rect, bool, str, PPEItemStatus]] = []
    for item, status in assessment.items.items():
        if status.present is None:
            continue
        resolved = resolve_ppe_box(item, status, track.bbox, frame.shape)
        if resolved is None:
            continue
        rect, from_model = resolved
        drawn.append((rect, from_model, item, status))

    # Top-down, so the helmet label claims the space above the head before the
    # vest label starts looking for somewhere to go.
    drawn.sort(key=lambda entry: entry[0][1])

    for rect, from_model, item, status in drawn:
        colour = _item_colour(item, status.present)
        if from_model:
            cv2.rectangle(frame, rect[:2], rect[2:], colour, thickness, cv2.LINE_AA)
        else:
            _dashed_rect(frame, rect, colour, thickness)

        lines: list[Line] = [(_item_text(item, status), colour, scale)]
        if show_track_id:
            lines.append((f"ID #{track.track_id:03d}", COLOUR_PANEL, scale * 0.8))
        _draw_stack(frame, lines, placer.place(_stack_size(lines), rect))

    return len(drawn)


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[Track],
    ppe: Mapping[int, PPEAssessment] | None = None,
    speeds: Mapping[int, float] | None = None,
    avoid: Rect | Sequence[Rect] | None = None,
    zones_by_track: Mapping[int, list[str]] | None = None,
    show_person_boxes: bool | None = None,
    show_ppe_boxes: bool | None = None,
    show_track_id: bool | None = None,
) -> np.ndarray:
    """Draw the PPE findings for each tracked person.

    By default that is *only* the PPE: a tight box on the head for a helmet, on
    the torso for a vest, each labelled with the item and its confidence. The
    person box behind them is what associated the item to this person, and it
    stays off screen unless `show_person_boxes` (``SHOW_PERSON_BOXES``) asks for
    it — at which point the full track readout returns for debugging.

    `avoid` is a rect, or rects, no label may cover: the HUD and any event
    banners already on the frame.
    """
    ppe = ppe or {}
    speeds = speeds or {}
    show_person = settings.show_person_boxes if show_person_boxes is None else show_person_boxes
    show_ppe = settings.show_ppe_boxes if show_ppe_boxes is None else show_ppe_boxes
    show_id = settings.show_track_id if show_track_id is None else show_track_id

    placer = LabelPlacer(frame.shape, _reserved_rects(avoid))

    for track in tracks:
        assessment = ppe.get(track.track_id)

        if show_ppe and assessment is not None:
            draw_ppe_items(
                frame, track, assessment, placer, show_track_id=show_id
            )

        if not show_person:
            continue

        rect = clip_to_frame(track.bbox, frame.shape)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect

        colour = COLOUR_TRACK
        if assessment is not None and assessment.is_violation:
            colour = COLOUR_VIOLATION
        elif assessment is not None and (assessment.helmet or assessment.vest):
            colour = COLOUR_OK

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        # Corner ticks read as a targeting reticle and survive downscaling
        # better than a thin box alone.
        tick = max(8, min(24, (x2 - x1) // 5))
        for cx, cy, dx, dy in (
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1),
        ):
            cv2.line(frame, (cx, cy), (cx + dx * tick, cy), colour, 3, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * tick), colour, 3, cv2.LINE_AA)

        header: list[Line] = [
            (f"Person #{track.track_id:03d}  {track.confidence * 100:.0f}%",
             colour, 0.48)
        ]
        _draw_stack(frame, header, placer.place(_stack_size(header), rect))

        lines: list[Line] = []
        if assessment is not None and assessment.method != "disabled":
            def mark(value: bool | None) -> str:
                return "YES" if value else ("NO" if value is False else "?")

            lines.append(
                (
                    f"Helmet:{mark(assessment.helmet)} Vest:{mark(assessment.vest)}",
                    COLOUR_VIOLATION if assessment.is_violation else COLOUR_OK,
                    0.42,
                )
            )
            # The snapshot is the durable record — the one an investigator sees
            # months later without the dashboard around it. How the verdict was
            # reached is spelled out here rather than encoded as a symbol.
            lines.append(
                (
                    "AI MODEL" if assessment.method == "model" else "HEURISTIC EST.",
                    COLOUR_OK if assessment.method == "model" else COLOUR_WARNING,
                    0.42,
                )
            )
        speed = speeds.get(track.track_id, 0.0)
        if speed > 0.05:
            lines.append(
                (f"{speed:.1f} bh/s",
                 COLOUR_WARNING if speed > 2.0 else COLOUR_TRACK, 0.42)
            )
        if zones_by_track and zones_by_track.get(track.track_id):
            lines.append((f"zone: {len(zones_by_track[track.track_id])}", COLOUR_ZONE, 0.42))

        if lines:
            _draw_stack(frame, lines, placer.place(_stack_size(lines), rect))

    return frame


def _reserved_rects(avoid: Rect | Sequence[Rect] | None) -> list[Rect]:
    """Normalise `avoid` — one rect or several — into a list of rects."""
    if avoid is None:
        return []
    items: list[Any] = list(avoid)
    if len(items) == 4 and all(isinstance(value, int | float) for value in items):
        return [_as_rect(items)]
    return [_as_rect(rect) for rect in items if rect is not None]


def _hud_lines(
    camera_name: str, stats: Mapping[str, Any], timestamp: str
) -> list[str]:
    lines = [
        f"{camera_name}",
        f"FPS {stats.get('fps', 0):.1f}   inference {stats.get('inference_ms', 0):.0f}ms",
        f"people {stats.get('people_count', 0)}   backend {stats.get('backend', '-')}",
    ]
    # The PPE method belongs on the evidence frame itself: a snapshot that
    # leaves the system must carry whether its PPE verdict came from a trained
    # model or the colour fallback.
    ppe_method = str(stats.get("ppe_method") or "")
    if ppe_method and ppe_method != "disabled":
        lines.append(
            f"PPE {'AI MODEL' if ppe_method == 'model' else 'HEURISTIC FALLBACK'}"
        )
    # Only when it is *not* the validator model. A helmet verdict nobody could
    # verify against a real hard hat has to say so on the frame itself, not
    # only in diagnostics — the frame is what leaves the system.
    hardhat_method = str(stats.get("hardhat_method") or "")
    if hardhat_method and hardhat_method != "model":
        lines.append(
            "HARD HAT " + {
                "heuristic": "HEURISTIC",
                "disabled": "UNVALIDATED",
            }.get(hardhat_method, "UNVERIFIED")
        )
    if timestamp:
        lines.append(timestamp)
    return lines


#: Panel metrics, shared so the track labels know what to steer around.
HUD_PAD, HUD_LINE_HEIGHT, HUD_WIDTH, HUD_MARGIN = 8, 20, 300, 10


def hud_rect(
    frame: np.ndarray,
    camera_name: str,
    stats: Mapping[str, Any],
    timestamp: str = "",
) -> tuple[int, int, int, int]:
    """Where the HUD will be drawn, as (x0, y0, x1, y1).

    Exposed so per-person labels can avoid it. A person standing at the bottom
    of frame would otherwise have their PPE verdict rendered directly on top of
    the telemetry panel, leaving both unreadable.
    """
    height = frame.shape[0]
    count = len(_hud_lines(camera_name, stats, timestamp))
    panel_h = HUD_LINE_HEIGHT * count + HUD_PAD
    y0 = height - panel_h - HUD_MARGIN
    return (HUD_MARGIN, y0, HUD_MARGIN + HUD_WIDTH, y0 + panel_h)


def draw_hud(
    frame: np.ndarray,
    camera_name: str,
    stats: Mapping[str, Any],
    timestamp: str = "",
) -> np.ndarray:
    """Bottom-left telemetry panel: fps, latency, counts, backend."""
    lines = _hud_lines(camera_name, stats, timestamp)
    panel_h = HUD_LINE_HEIGHT * len(lines) + HUD_PAD
    y0 = frame.shape[0] - panel_h - HUD_MARGIN
    line_height, panel_w = HUD_LINE_HEIGHT, HUD_WIDTH

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, y0), (10 + panel_w, y0 + panel_h), COLOUR_PANEL, -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (10, y0), (10 + panel_w, y0 + panel_h), (70, 60, 50), 1)

    for index, text in enumerate(lines):
        cv2.putText(
            frame, text, (18, y0 + 16 + index * line_height),
            FONT, 0.44, COLOUR_TEXT, 1, cv2.LINE_AA,
        )
    return frame


#: Severity → banner colour, matching the dashboard's severity scale.
SEVERITY_COLOURS: dict[str, tuple[int, int, int]] = {
    "CRITICAL": (77, 77, 255),
    "HIGH": (61, 138, 255),
    "MEDIUM": (61, 197, 255),
    "LOW": (245, 156, 91),
}


@dataclass(slots=True)
class ActiveEvent:
    """An event being shown on the frame, and until when.

    An event is one instant, but a viewer needs to *see* it: at 25 fps a
    single-frame banner is 40 ms on screen. `expires_at` holds it visible for
    `EVENT_OVERLAY_SECONDS` of video time, so scrubbing to the moment of an
    incident actually shows the incident.
    """

    label: str
    severity: str
    person_id: int | None
    expires_at: float

    @property
    def text(self) -> str:
        who = f"  PERSON #{self.person_id:03d}" if self.person_id is not None else ""
        return f"{self.label.upper()}{who}"


def _banner_layout(
    frame: np.ndarray, events: Sequence[ActiveEvent], max_shown: int = 4
) -> tuple[list[tuple[ActiveEvent, Rect, float, int]], int]:
    """Lay the banners out: ``[(event, rect, scale, thickness)], hidden``.

    Ordered by severity so the worst thing on screen is the first thing read,
    and capped: a pile-up of banners hides the footage they are describing.
    Layout is computed here rather than inside the drawing loop so the label
    placer can be told where the banners will land *before* anything is drawn.
    """
    if not events:
        return [], 0

    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    ranked = sorted(events, key=lambda e: (order.get(e.severity.upper(), 9), e.label))
    shown = ranked[:max_shown]

    width = frame.shape[1]
    scale = max(0.5, min(0.85, width / 1400.0))
    thickness = 2 if width >= 960 else 1
    row_height = int(34 * scale) + 12

    rows: list[tuple[ActiveEvent, Rect, float, int]] = []
    y = 12
    for event in shown:
        (text_w, text_h), baseline = cv2.getTextSize(
            event.text, FONT, scale, thickness
        )
        box_w = text_w + int(58 * scale)
        box_h = text_h + baseline + int(18 * scale)
        x = max(10, (width - box_w) // 2)
        rows.append((event, (x, y, x + box_w, y + box_h), scale, thickness))
        y += row_height

    return rows, len(ranked) - len(shown)


def banner_rect(
    frame: np.ndarray, events: Sequence[ActiveEvent], max_shown: int = 4
) -> Rect | None:
    """Where the event banners will be drawn, as one rect, or None.

    Exposed for the same reason as :func:`hud_rect`: a PPE label rendered under
    a banner is a finding nobody can read.
    """
    rows, hidden = _banner_layout(frame, events, max_shown)
    if not rows:
        return None
    x0 = min(rect[0] for _, rect, _, _ in rows)
    x1 = max(rect[2] for _, rect, _, _ in rows)
    y1 = max(rect[3] for _, rect, _, _ in rows)
    if hidden > 0:
        y1 += int(24 * rows[0][2])
    return (x0, rows[0][1][1], x1, y1)


def draw_event_banners(
    frame: np.ndarray, events: Sequence[ActiveEvent], max_shown: int = 4
) -> np.ndarray:
    """Stack active event banners at the top of the frame."""
    rows, hidden = _banner_layout(frame, events, max_shown)
    if not rows:
        return frame

    for event, (x, y, x_end, y_end), scale, thickness in rows:
        colour = SEVERITY_COLOURS.get(event.severity.upper(), COLOUR_WARNING)

        # Translucent plate so the banner stays readable over bright footage
        # without hiding what is underneath it.
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x_end, y_end), COLOUR_PANEL, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x, y), (x_end, y_end), colour, 2, cv2.LINE_AA)
        # Severity gate: colour is never the only carrier of the distinction,
        # but it makes the scan fast for anyone who can use it.
        cv2.rectangle(frame, (x, y), (x + int(8 * scale), y_end), colour, -1)

        cv2.putText(
            frame, event.text,
            (x + int(22 * scale), y_end - int(10 * scale)),
            FONT, scale, colour, thickness, cv2.LINE_AA,
        )

    if hidden > 0:
        _, last_rect, scale, _ = rows[-1]
        cv2.putText(
            frame, f"+{hidden} more",
            (max(10, frame.shape[1] // 2 - 40), last_rect[3] + int(20 * scale)),
            FONT, scale * 0.8, COLOUR_TEXT, 1, cv2.LINE_AA,
        )
    return frame


def annotate_frame(
    frame: np.ndarray,
    tracks: Sequence[Track],
    zones: Sequence[ResolvedZone] = (),
    ppe: Mapping[int, PPEAssessment] | None = None,
    speeds: Mapping[int, float] | None = None,
    camera_name: str = "",
    stats: Mapping[str, Any] | None = None,
    timestamp: str = "",
    copy: bool = True,
    events: Sequence[ActiveEvent] = (),
) -> np.ndarray:
    """Full overlay pass. Returns a new frame unless `copy=False`.

    Draw order is deliberate: zones sit under the people standing in them,
    PPE boxes and their labels sit above both, and event banners go last so
    nothing can obscure the alert. The HUD and the banners are measured up
    front so no PPE label is placed where one of them is about to land.
    """
    canvas = frame.copy() if copy else frame
    show_hud = bool(camera_name or stats)
    reserved: list[Rect] = []
    if show_hud:
        reserved.append(hud_rect(canvas, camera_name, stats or {}, timestamp))
    banner = banner_rect(canvas, events)
    if banner is not None:
        reserved.append(banner)
    draw_zones(canvas, zones)
    draw_tracks(canvas, tracks, ppe=ppe, speeds=speeds, avoid=reserved)
    if show_hud:
        draw_hud(canvas, camera_name, stats or {}, timestamp)
    draw_event_banners(canvas, events)
    return canvas
