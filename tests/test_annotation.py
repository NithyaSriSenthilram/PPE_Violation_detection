"""Overlay rendering: local PPE boxes, not whole-person rectangles.

The annotated render is what an operator actually looks at, and what it draws
is a product decision with a rule behind it: *the box goes where the finding
is*. A helmet box belongs on a head, a missing-vest box on a torso, and the
body-sized rectangle that used to sit over both belongs off screen — person
detection and tracking still run, they simply are not the picture.

These tests hold that rule in place. They draw onto black frames so "was
anything rendered here?" is a question about pixels rather than about
mocks, and they check the geometry decisions (:func:`resolve_ppe_box`)
directly, because that is where a model's own box is either trusted or
constrained.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import settings
from backend.inference.ppe import PPEAssessment
from backend.inference.ppe_taxonomy import BODY_REGIONS, body_region_box, region_for
from backend.tracking import Track, TrackState
from backend.video.annotate import (
    ActiveEvent,
    LabelPlacer,
    annotate_frame,
    banner_rect,
    clip_to_frame,
    draw_tracks,
    hud_rect,
    resolve_ppe_box,
)

WIDTH, HEIGHT = 960, 540

#: A person standing in the left third of frame: 300 px tall, 100 wide.
PERSON = (200.0, 100.0, 300.0, 400.0)


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════
def blank() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def make_track(track_id: int = 1, bbox: tuple = PERSON) -> Track:
    track = Track(
        track_id=track_id, bbox=bbox, confidence=0.9, state=TrackState.TRACKED
    )
    track.record(0.0)
    return track


def assess(*items, person_id: int = 1) -> PPEAssessment:
    """Build an assessment from ``(item, present, confidence, bbox)`` tuples."""
    assessment = PPEAssessment(person_id=person_id, method="model")
    for item, present, confidence, bbox in items:
        assessment.set(
            item, present, confidence,
            source="detected" if bbox else "inferred", bbox=bbox,
        )
    return assessment


def painted(frame: np.ndarray) -> np.ndarray:
    """Bounding box of every non-black pixel, as (x1, y1, x2, y2)."""
    ys, xs = np.nonzero(frame.any(axis=2))
    assert len(xs), "nothing was drawn at all"
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def is_painted(frame: np.ndarray, x: float, y: float, radius: int = 2) -> bool:
    """Is anything drawn within `radius` px of this point?"""
    x, y = int(x), int(y)
    patch = frame[
        max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1
    ]
    return bool(patch.any())


def region_of(item: str, person: tuple = PERSON) -> tuple[float, float, float, float]:
    return body_region_box(person, item)


def _rects_disjoint(a: tuple, b: tuple) -> bool:
    return min(a[2], b[2]) <= max(a[0], b[0]) or min(a[3], b[3]) <= max(a[1], b[1])


# ══════════════════════════════════════════════════════════════════════════
#  The person box is computed, tracked — and not drawn
# ══════════════════════════════════════════════════════════════════════════
class TestPersonBoxIsHidden:
    def test_it_is_off_by_default(self):
        assert settings.show_person_boxes is False
        assert settings.show_ppe_boxes is True
        assert settings.show_track_id is False

    def test_no_person_rectangle_is_rendered(self):
        frame = blank()
        draw_tracks(
            frame, [make_track()],
            ppe={1: assess(("helmet", True, 0.94, (215.0, 100.0, 285.0, 135.0)))},
        )
        px1, py1, px2, py2 = PERSON
        # The four places a body-sized rectangle would unmistakably show:
        # its bottom edge, and its sides down at knee height.
        assert not is_painted(frame, (px1 + px2) / 2, py2)
        assert not is_painted(frame, px1, py1 + 0.8 * (py2 - py1))
        assert not is_painted(frame, px2, py1 + 0.8 * (py2 - py1))
        assert not is_painted(frame, (px1 + px2) / 2, py1 + 0.9 * (py2 - py1))

    def test_nothing_is_drawn_below_the_ppe_regions(self):
        """Everything rendered sits in the head/torso half of the person."""
        frame = blank()
        draw_tracks(
            frame, [make_track()],
            ppe={
                1: assess(
                    ("helmet", True, 0.94, (215.0, 100.0, 285.0, 135.0)),
                    ("vest", False, 0.88, (205.0, 160.0, 295.0, 270.0)),
                )
            },
        )
        _, _, _, bottom = painted(frame)
        assert bottom < PERSON[1] + 0.75 * (PERSON[3] - PERSON[1])

    def test_the_person_box_comes_back_when_asked_for(self):
        """The data never went away — only the decision to draw it."""
        frame = blank()
        draw_tracks(
            frame, [make_track()], ppe={1: assess()}, show_person_boxes=True
        )
        px1, py1, px2, py2 = PERSON
        assert is_painted(frame, (px1 + px2) / 2, py2)
        assert is_painted(frame, px1, py1 + 0.8 * (py2 - py1))

    def test_a_track_with_no_ppe_verdict_draws_nothing(self):
        """Undetermined is not a finding, and gets no box asserting one."""
        frame = blank()
        draw_tracks(frame, [make_track()], ppe={1: PPEAssessment(person_id=1)})
        assert not frame.any()

    def test_ppe_boxes_can_be_turned_off_too(self):
        frame = blank()
        draw_tracks(
            frame, [make_track()],
            ppe={1: assess(("helmet", True, 0.94, (215.0, 100.0, 285.0, 135.0)))},
            show_ppe_boxes=False,
        )
        assert not frame.any()


# ══════════════════════════════════════════════════════════════════════════
#  Local boxes: the model's own when it is local, the body region when not
# ══════════════════════════════════════════════════════════════════════════
class TestBoxResolution:
    @pytest.mark.parametrize("present", [True, False])
    def test_a_local_helmet_box_is_used_verbatim(self, present):
        model_box = (218.0, 102.0, 282.0, 138.0)
        status = assess(("helmet", present, 0.91, model_box)).status("helmet")
        rect, from_model = resolve_ppe_box("helmet", status, PERSON, (HEIGHT, WIDTH, 3))
        assert from_model is True
        assert rect == (218, 102, 282, 138)

    @pytest.mark.parametrize("present", [True, False])
    def test_a_local_vest_box_is_used_verbatim(self, present):
        model_box = (204.0, 160.0, 296.0, 272.0)
        status = assess(("vest", present, 0.87, model_box)).status("vest")
        rect, from_model = resolve_ppe_box("vest", status, PERSON, (HEIGHT, WIDTH, 3))
        assert from_model is True
        assert rect == (204, 160, 296, 272)

    def test_a_helmet_box_the_size_of_the_person_is_constrained_to_the_head(self):
        """Some models answer 'no helmet' with a copy of the person box."""
        status = assess(("helmet", False, 0.91, PERSON)).status("helmet")
        rect, from_model = resolve_ppe_box("helmet", status, PERSON, (HEIGHT, WIDTH, 3))
        assert from_model is False, "a body-sized box must not be trusted as local"
        height = PERSON[3] - PERSON[1]
        assert rect[3] <= PERSON[1] + 0.31 * height
        assert (rect[3] - rect[1]) / height <= 0.31

    def test_a_vest_box_the_size_of_the_person_is_constrained_to_the_torso(self):
        status = assess(("vest", False, 0.88, PERSON)).status("vest")
        rect, from_model = resolve_ppe_box("vest", status, PERSON, (HEIGHT, WIDTH, 3))
        assert from_model is False
        height = PERSON[3] - PERSON[1]
        assert rect[1] >= PERSON[1] + 0.19 * height
        assert rect[3] <= PERSON[1] + 0.71 * height

    def test_an_absence_with_no_box_falls_back_to_the_body_region(self):
        """Absence inferred from omission carries no box of its own."""
        status = assess(("helmet", False, 0.55, None)).status("helmet")
        assert status.bbox is None
        rect, from_model = resolve_ppe_box("helmet", status, PERSON, (HEIGHT, WIDTH, 3))
        assert from_model is False
        expected = clip_to_frame(region_of("helmet"), (HEIGHT, WIDTH, 3))
        assert rect == expected

    def test_a_box_entirely_outside_the_frame_is_dropped(self):
        status = assess(("helmet", True, 0.9, (-80.0, -60.0, -10.0, -20.0))).status(
            "helmet"
        )
        assert resolve_ppe_box("helmet", status, PERSON, (HEIGHT, WIDTH, 3)) is None

    def test_every_known_item_maps_to_a_defined_region(self):
        for item in ("helmet", "vest", "gloves", "boots", "goggles", "mask", "harness"):
            assert region_for(item) in BODY_REGIONS
            box = body_region_box(PERSON, item)
            assert PERSON[0] <= box[0] < box[2] <= PERSON[2]
            assert PERSON[1] <= box[1] < box[3] <= PERSON[3]

    def test_regions_sit_where_the_body_part_is(self):
        head, torso, feet = (region_of(i) for i in ("helmet", "vest", "boots"))
        assert head[3] <= torso[3] and head[1] <= torso[1]
        assert torso[3] <= feet[1]

    def test_an_unmapped_item_still_gets_a_drawable_region(self):
        box = body_region_box(PERSON, "lanyard")
        assert box == region_of("vest")


# ══════════════════════════════════════════════════════════════════════════
#  What lands on the frame
# ══════════════════════════════════════════════════════════════════════════
class TestRenderedBoxes:
    @pytest.mark.parametrize(
        ("item", "present", "box"),
        [
            ("helmet", True, (218.0, 102.0, 282.0, 138.0)),
            ("helmet", False, (218.0, 102.0, 282.0, 138.0)),
            ("helmet", False, None),
            ("vest", True, (204.0, 160.0, 296.0, 272.0)),
            ("vest", False, (204.0, 160.0, 296.0, 272.0)),
            ("vest", False, None),
        ],
    )
    def test_the_box_is_drawn_on_the_body_part_it_names(self, item, present, box):
        frame = blank()
        assessment = assess((item, present, 0.9, box))
        draw_tracks(frame, [make_track()], ppe={1: assessment})
        rect, _ = resolve_ppe_box(
            item, assessment.status(item), PERSON, (HEIGHT, WIDTH, 3)
        )
        # Corners of the resolved box are painted; the region it does not
        # cover — the legs — is not.
        assert is_painted(frame, rect[0], rect[1])
        assert is_painted(frame, rect[2], rect[3])
        assert not is_painted(frame, (PERSON[0] + PERSON[2]) / 2, PERSON[3] - 5)

    def test_the_box_is_tight_relative_to_the_person(self):
        """A helmet box covers a head, not a body."""
        frame = blank()
        track = make_track()
        draw_tracks(
            frame, [track],
            ppe={1: assess(("helmet", False, 0.91, (218.0, 102.0, 282.0, 138.0)))},
        )
        _, _, _, bottom = painted(frame)
        person_height = PERSON[3] - PERSON[1]
        assert (bottom - PERSON[1]) / person_height < 0.35

    def test_helmet_and_vest_are_two_independent_boxes(self):
        frame = blank()
        helmet_box = (218.0, 102.0, 282.0, 138.0)
        vest_box = (204.0, 170.0, 296.0, 280.0)
        assessment = assess(
            ("helmet", True, 0.94, helmet_box), ("vest", False, 0.86, vest_box)
        )
        draw_tracks(frame, [make_track()], ppe={1: assessment})

        helmet_rect, _ = resolve_ppe_box(
            "helmet", assessment.status("helmet"), PERSON, (HEIGHT, WIDTH, 3)
        )
        vest_rect, _ = resolve_ppe_box(
            "vest", assessment.status("vest"), PERSON, (HEIGHT, WIDTH, 3)
        )
        # Two rectangles, disjoint, each drawn where it belongs — never one
        # rectangle spanning the pair.
        assert helmet_rect[3] < vest_rect[1]
        for rect in (helmet_rect, vest_rect):
            assert is_painted(frame, rect[0], rect[1])
            assert is_painted(frame, rect[2], rect[3])
        # And nothing reaches past the lower of the two, give or take the
        # width of the outline itself.
        assert painted(frame)[3] <= vest_rect[3] + 2

    def test_each_box_stays_with_its_own_person(self):
        left, right = (200.0, 100.0, 300.0, 400.0), (600.0, 100.0, 700.0, 400.0)
        frame = blank()
        draw_tracks(
            frame,
            [make_track(1, left), make_track(2, right)],
            ppe={
                1: assess(("helmet", True, 0.94, (215.0, 102.0, 285.0, 138.0))),
                # No box of its own: derived from *this* person's geometry.
                2: assess(("helmet", False, 0.55, None), person_id=2),
            },
        )
        _, region_top, _, _ = region_of("helmet", right)
        assert is_painted(frame, 615.0, region_top + 2)
        # Person 2's derived box must not have been built from person 1's body.
        for column in (int(left[0]), int(left[2])):
            assert not is_painted(frame, column, right[1] + 200)

    def test_boxes_at_the_frame_edge_are_clipped_not_wrapped(self):
        edge_person = (WIDTH - 60.0, HEIGHT - 120.0, WIDTH + 40.0, HEIGHT + 60.0)
        frame = blank()
        draw_tracks(
            frame, [make_track(7, edge_person)],
            ppe={7: assess(("vest", False, 0.8, None), person_id=7)},
        )
        x1, y1, x2, y2 = painted(frame)
        assert 0 <= x1 <= x2 < WIDTH
        assert 0 <= y1 <= y2 < HEIGHT

    def test_a_box_straddling_the_left_edge_is_clipped(self):
        assert clip_to_frame((-50.0, -30.0, 40.0, 60.0), (HEIGHT, WIDTH, 3)) == (
            0, 0, 40, 60,
        )

    def test_a_degenerate_box_is_not_drawn(self):
        assert clip_to_frame((10.0, 10.0, 11.0, 11.0), (HEIGHT, WIDTH, 3)) is None


# ══════════════════════════════════════════════════════════════════════════
#  Labels
# ══════════════════════════════════════════════════════════════════════════
class TestLabels:
    def _text_rows(self, frame: np.ndarray) -> set[int]:
        return set(np.nonzero(frame.any(axis=2).any(axis=1))[0].tolist())

    def test_the_label_names_the_item_not_the_person(self):
        """Rendered text is checked by geometry elsewhere; here, by construction."""
        from backend.video.annotate import _item_text

        status = assess(("helmet", False, 0.91, None)).status("helmet")
        assert _item_text("helmet", status) == "NO HELMET 91%"
        worn = assess(("vest", True, 0.87, None)).status("vest")
        assert _item_text("vest", worn) == "VEST 87%"

    def test_the_track_id_is_a_secondary_line_only_when_asked_for(self):
        box = (218.0, 102.0, 282.0, 138.0)
        without = blank()
        draw_tracks(
            without, [make_track(10)], ppe={10: assess(("helmet", True, 0.94, box))}
        )
        with_id = blank()
        draw_tracks(
            with_id, [make_track(10)],
            ppe={10: assess(("helmet", True, 0.94, box))},
            show_track_id=True,
        )
        assert int(with_id.any(axis=2).sum()) > int(without.any(axis=2).sum())

    def test_labels_never_leave_the_frame(self):
        """A person hard against the top-left corner still gets a readable label."""
        corner = (0.0, 0.0, 90.0, 260.0)
        frame = blank()
        draw_tracks(
            frame, [make_track(3, corner)],
            ppe={3: assess(("helmet", False, 0.91, None), person_id=3)},
        )
        x1, y1, x2, y2 = painted(frame)
        assert (x1, y1) >= (0, 0)
        assert x2 < WIDTH and y2 < HEIGHT

    def test_two_neighbouring_labels_do_not_overlap(self):
        placer = LabelPlacer((HEIGHT, WIDTH, 3))
        size = (120, 22)
        first = placer.place(size, (200, 100, 260, 140))
        second = placer.place(size, (210, 100, 270, 140))
        a = (first[0], first[1], first[0] + size[0], first[1] + size[1])
        b = (second[0], second[1], second[0] + size[0], second[1] + size[1])
        overlap_x = min(a[2], b[2]) - max(a[0], b[0])
        overlap_y = min(a[3], b[3]) - max(a[1], b[1])
        assert overlap_x <= 0 or overlap_y <= 0, "labels landed on top of each other"

    def test_a_label_is_preferentially_placed_above_its_box(self):
        placer = LabelPlacer((HEIGHT, WIDTH, 3))
        x, y = placer.place((120, 22), (200, 100, 260, 140))
        assert y + 22 <= 100

    def test_a_label_with_no_room_above_goes_below(self):
        placer = LabelPlacer((HEIGHT, WIDTH, 3))
        x, y = placer.place((120, 22), (200, 0, 260, 40))
        assert y >= 40

    def test_a_label_moves_off_a_reserved_region(self):
        """Rendered proof: a label will not be drawn into reserved space."""
        frame = blank()
        helmet_box = (218.0, 110.0, 282.0, 146.0)
        # Exactly where the label would otherwise go — directly above the box.
        reserved = (150, 60, 400, 108)
        draw_tracks(
            frame, [make_track()],
            ppe={1: assess(("helmet", False, 0.91, helmet_box))},
            avoid=reserved,
        )
        rx0, ry0, rx1, ry1 = reserved
        assert not frame[ry0:ry1, rx0:rx1].any(), (
            "a label was drawn into the reserved region"
        )

    def test_labels_keep_clear_of_the_hud(self):
        """A person at the bottom-left, where the telemetry panel lives.

        The PPE box itself stays on the torso — that is the whole point of it —
        but the label that explains it has to end up somewhere readable.
        """
        stats = {"fps": 30.0, "people_count": 1, "backend": "onnx"}
        reserved = hud_rect(blank(), "cam", stats, "t=1.0s")
        placer = LabelPlacer((HEIGHT, WIDTH, 3), [reserved])
        low_left = (20, HEIGHT - 160, 120, HEIGHT - 40)
        x, y = placer.place((150, 22), low_left)
        assert _rects_disjoint((x, y, x + 150, y + 22), reserved)

    def test_labels_keep_clear_of_the_event_banner(self):
        events = [ActiveEvent("Missing Safety Vest", "HIGH", 5, 99.0)]
        reserved = banner_rect(blank(), events)
        assert reserved is not None
        placer = LabelPlacer((HEIGHT, WIDTH, 3), [reserved])
        # Someone's head, directly under the banner.
        head = (WIDTH // 2 - 30, 20, WIDTH // 2 + 30, 90)
        x, y = placer.place((150, 22), head)
        assert _rects_disjoint((x, y, x + 150, y + 22), reserved)


# ══════════════════════════════════════════════════════════════════════════
#  The whole pass
# ══════════════════════════════════════════════════════════════════════════
class TestAnnotateFrame:
    def test_banners_and_hud_coexist_with_local_ppe_boxes(self):
        frame = blank()
        vest_box = (204.0, 170.0, 296.0, 280.0)
        canvas = annotate_frame(
            frame,
            [make_track()],
            ppe={1: assess(("vest", False, 0.86, vest_box))},
            camera_name="factory.mp4",
            stats={"fps": 25.0, "people_count": 1, "backend": "onnx"},
            timestamp="t=12.0s",
            events=[ActiveEvent("Missing Safety Vest", "HIGH", 1, 99.0)],
        )
        # The banner is at the top…
        assert canvas[12:60, :].any()
        # …the HUD at the bottom-left…
        hx0, hy0, hx1, hy1 = hud_rect(canvas, "factory.mp4", {"fps": 25.0}, "t=12.0s")
        assert canvas[hy0:hy1, hx0:hx1].any()
        # …and the violation box is still on the torso, not replaced by either.
        assert is_painted(canvas, vest_box[0], vest_box[3])
        assert not is_painted(canvas, (PERSON[0] + PERSON[2]) / 2, PERSON[3])

    def test_the_source_frame_is_left_alone(self):
        frame = blank()
        annotate_frame(
            frame, [make_track()],
            ppe={1: assess(("helmet", True, 0.94, (218.0, 102.0, 282.0, 138.0)))},
        )
        assert not frame.any()

    def test_settings_drive_the_defaults(self, monkeypatch):
        monkeypatch.setattr(settings, "show_person_boxes", True)
        frame = blank()
        annotate_frame(frame, [make_track()], ppe={1: assess()}, copy=False)
        assert is_painted(frame, (PERSON[0] + PERSON[2]) / 2, PERSON[3])
