"""The annotated render: the primary output of uploaded-video analysis.

These tests run the real pipeline over real video files and then interrogate
the file that comes out — frame count, duration, geometry, codec, decodability
— because "the job said COMPLETED" proves nothing about whether anyone can
watch the result. The failure this suite exists to catch is a render that
exists on disk and will not play.

The job is driven synchronously through ``VideoJobRunner._analyse`` rather than
the thread pool: a test that polls a background worker is a test that flakes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.config import settings
from backend.db.base import session_scope
from backend.db.models import Event, VideoJob
from backend.jobs.video_jobs import JobStatus, VideoJobRunner
from backend.video.writer import (
    AnnotatedVideoWriter,
    VideoWriteError,
    ffmpeg_path,
    probe_media,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE = FIXTURES / "surveillance_sample.mp4"
SAMPLE_WITH_AUDIO = FIXTURES / "sample_with_audio.mp4"


def source_properties(path: Path) -> dict[str, float]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), f"fixture {path} is not decodable"
    info = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return info


def run_job(source: Path, profile: str = "thorough", **fields) -> VideoJob:
    """Create and synchronously analyse a job over `source`."""
    import shutil
    import uuid

    job_id = str(uuid.uuid4())
    settings.ensure_directories()
    stored = settings.upload_root / f"{job_id}{source.suffix}"
    shutil.copy(source, stored)

    with session_scope() as session:
        session.add(
            VideoJob(
                job_id=job_id,
                filename=source.name,
                stored_path=str(stored),
                profile=profile,
                status=JobStatus.QUEUED,
                **fields,
            )
        )

    VideoJobRunner(max_workers=1)._analyse(job_id, profile)

    with session_scope() as session:
        job = session.get(VideoJob, job_id)
        session.expunge(job)
        return job


def render_path(job: VideoJob) -> Path:
    assert job.annotated_path, "job recorded no annotated video"
    from backend.config import PROJECT_ROOT

    path = Path(job.annotated_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _row_values(instance) -> dict:
    """Column values of an ORM instance, detached from its session."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }


@pytest.fixture(scope="module")
def analysed_once() -> dict:
    """Analyse the sample clip once for the whole module.

    Analysis is the expensive part — decoding, inference and encoding every
    frame — so it runs a single time and the resulting rows are captured as
    plain values.
    """
    job = run_job(SAMPLE)
    with session_scope() as session:
        events = session.query(Event).filter(Event.job_id == job.job_id).all()
        return {"job": _row_values(job), "events": [_row_values(e) for e in events]}


@pytest.fixture
def analysed(analysed_once) -> VideoJob:
    """The analysed job, present in the database for this test.

    conftest truncates every table between tests, which is the right default —
    but it also removes the rows this module analysed once and reuses. The
    render on disk survives, so the rows are simply reinstated rather than the
    whole analysis being repeated per test.
    """
    with session_scope() as session:
        if session.get(VideoJob, analysed_once["job"]["job_id"]) is None:
            session.add(VideoJob(**analysed_once["job"]))
            for values in analysed_once["events"]:
                session.add(Event(**values))
    return VideoJob(**analysed_once["job"])


# ══════════════════════════════════════════════════════════════════════════
#  The render
# ══════════════════════════════════════════════════════════════════════════
class TestAnnotatedRender:
    def test_the_job_completes(self, analysed):
        assert analysed.status == JobStatus.COMPLETED, analysed.message
        assert analysed.progress == 1.0

    def test_a_render_is_produced_outside_the_evidence_tree(self, analysed):
        """The full render is a distinct artefact from per-event evidence."""
        path = render_path(analysed)
        assert path.is_file()
        assert path.stat().st_size > 0
        assert path.is_relative_to(settings.processed_root)
        assert not path.is_relative_to(settings.evidence_root)

    def test_the_original_upload_is_not_overwritten(self, analysed):
        source = Path(analysed.stored_path)
        assert source.is_file()
        assert source.resolve() != render_path(analysed).resolve()

    def test_every_source_frame_is_rendered(self, analysed):
        """Frame-for-frame. A short render means drifting annotations."""
        expected = source_properties(SAMPLE)["frames"]
        assert analysed.processed_frames == expected
        assert analysed.output["frames_written"] == expected

        capture = cv2.VideoCapture(str(render_path(analysed)))
        counted = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            counted += 1
        capture.release()
        assert counted == expected

    def test_duration_fps_and_resolution_match_the_source(self, analysed):
        source = source_properties(SAMPLE)
        rendered = source_properties(render_path(analysed))
        assert rendered["width"] == source["width"]
        assert rendered["height"] == source["height"]
        assert rendered["fps"] == pytest.approx(source["fps"], abs=0.05)
        assert analysed.output["resolution"] == f"{source['width']}x{source['height']}"

        expected_seconds = source["frames"] / source["fps"]
        assert analysed.output["duration_seconds"] == pytest.approx(
            expected_seconds, abs=0.25
        )

    def test_the_render_opens_and_decodes(self, analysed):
        capture = cv2.VideoCapture(str(render_path(analysed)))
        assert capture.isOpened()
        ok, frame = capture.read()
        capture.release()
        assert ok and frame is not None and frame.size > 0

    def test_the_codec_is_one_browsers_play(self, analysed):
        assert analysed.output["browser_compatible"] is True
        assert analysed.output["codec"].lower() in {"h264", "avc1", "vp8", "vp9", "av1"}

    @pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not installed")
    def test_the_index_is_at_the_front_so_seeking_works(self, analysed):
        """`moov` before `mdat` — otherwise the browser must fetch it all first."""
        head = render_path(analysed).read_bytes()[:400_000]
        moov, mdat = head.find(b"moov"), head.find(b"mdat")
        assert moov != -1, "no moov atom found in the first 400 KB"
        assert mdat == -1 or moov < mdat

    def test_frames_are_actually_annotated(self, analysed):
        """The render must differ from the source, or nothing was drawn.

        Compares the same frame index in both files. Encoding alone perturbs
        pixels slightly, so the bar is a *structural* difference: a meaningful
        number of pixels changed a meaningful amount.
        """
        index = 12
        original = cv2.VideoCapture(str(SAMPLE))
        rendered = cv2.VideoCapture(str(render_path(analysed)))
        for _ in range(index + 1):
            ok_a, frame_a = original.read()
            ok_b, frame_b = rendered.read()
        original.release()
        rendered.release()
        assert ok_a and ok_b

        difference = cv2.absdiff(frame_a, frame_b).max(axis=2)
        changed = float((difference > 60).mean())
        assert changed > 0.002, (
            f"only {changed:.4%} of pixels differ — the render looks unannotated"
        )

    def test_the_hud_timecode_is_burned_in(self, analysed):
        """The overlay is drawn over the picture, not composited by the player."""
        capture = cv2.VideoCapture(str(render_path(analysed)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 20)
        ok, frame = capture.read()
        capture.release()
        assert ok
        # The HUD plate is a dark panel in the bottom-left corner.
        height, width = frame.shape[:2]
        corner = frame[int(height * 0.78) : height - 4, 8 : min(320, width)]
        assert corner.mean() < 140, "no HUD panel found in the bottom-left corner"


# ══════════════════════════════════════════════════════════════════════════
#  Events, evidence, timeline
# ══════════════════════════════════════════════════════════════════════════
class TestEventsAndEvidence:
    def test_events_carry_a_seekable_source_timestamp(self, analysed):
        with session_scope() as session:
            events = (
                session.query(Event).filter(Event.job_id == analysed.job_id).all()
            )
            positions = [e.video_timestamp for e in events]
            job_frames = analysed.processed_frames
            fps = analysed.fps or 25.0

        for position in positions:
            assert position is not None
            assert 0.0 <= position <= (job_frames / fps) + 1.0

    def test_evidence_is_captured_for_events_only(self, analysed):
        """Snapshots mark incidents; they are not produced per frame."""
        with session_scope() as session:
            events = (
                session.query(Event).filter(Event.job_id == analysed.job_id).all()
            )
            snapshots = [e.snapshot_path for e in events if e.snapshot_path]
            count = len(events)

        assert count < analysed.processed_frames, (
            "one snapshot per frame would mean the render is not the primary output"
        )
        from backend.events.evidence import resolve_evidence_path

        for relative in snapshots:
            assert resolve_evidence_path(relative) is not None

    def test_the_summary_reports_real_counters(self, analysed):
        summary = analysed.summary
        assert summary["frames_analysed"] == analysed.processed_frames
        assert summary["processing_fps"] > 0
        assert "ppe_violations" in summary
        assert summary["backend"]


# ══════════════════════════════════════════════════════════════════════════
#  Audio
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(
    not SAMPLE_WITH_AUDIO.is_file(), reason="no audio fixture available"
)
class TestAudio:
    @pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not installed")
    def test_source_audio_survives_into_the_render(self):
        job = run_job(SAMPLE_WITH_AUDIO, profile="fast")
        assert job.status == JobStatus.COMPLETED, job.message
        assert job.output["has_audio"] is True

        probe = probe_media(render_path(job))
        assert probe.get("has_audio") is True
        assert probe.get("codec", "").lower() == "h264"

    def test_a_silent_source_still_renders_and_says_so(self, analysed):
        assert analysed.output["has_audio"] is False
        assert any("audio" in w.lower() for w in analysed.output["warnings"])


# ══════════════════════════════════════════════════════════════════════════
#  Failure handling
# ══════════════════════════════════════════════════════════════════════════
class TestFailures:
    def test_a_corrupt_file_fails_the_job_rather_than_completing(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"\x00\x01\x02 not a video " * 400)
        job = run_job(broken)
        assert job.status == JobStatus.FAILED
        assert job.annotated_path is None
        assert job.message

    def test_a_missing_upload_fails_cleanly(self):
        import uuid

        job_id = str(uuid.uuid4())
        with session_scope() as session:
            session.add(
                VideoJob(
                    job_id=job_id,
                    filename="gone.mp4",
                    stored_path="/nonexistent/gone.mp4",
                    profile="fast",
                    status=JobStatus.QUEUED,
                )
            )
        VideoJobRunner(max_workers=1)._analyse(job_id, "fast")
        with session_scope() as session:
            job = session.get(VideoJob, job_id)
            assert job.status == JobStatus.FAILED
            assert "disk" in job.message.lower() or "no longer" in job.message.lower()

    def test_a_writer_with_no_frames_refuses_to_report_success(self, tmp_path):
        writer = AnnotatedVideoWriter(tmp_path / "empty.mp4", 25.0, (320, 240))
        writer.open()
        with pytest.raises(VideoWriteError, match="no frames"):
            writer.finalise()

    def test_an_unwritable_destination_is_reported(self, tmp_path):
        blocked = tmp_path / "nope"
        blocked.write_text("this is a file, not a directory")
        writer = AnnotatedVideoWriter(blocked / "out.mp4", 25.0, (320, 240))
        with pytest.raises((VideoWriteError, OSError, NotADirectoryError)):
            writer.open()


# ══════════════════════════════════════════════════════════════════════════
#  The writer in isolation
# ══════════════════════════════════════════════════════════════════════════
class TestWriter:
    def test_it_writes_exactly_the_frames_it_is_given(self, tmp_path):
        destination = tmp_path / "out.mp4"
        writer = AnnotatedVideoWriter(destination, 20.0, (320, 240))
        writer.open()
        for value in range(40):
            writer.write(np.full((240, 320, 3), value * 5 % 255, dtype=np.uint8))
        result = writer.finalise()

        assert result.frames_written == 40
        assert result.duration_seconds == pytest.approx(2.0, abs=0.2)
        assert destination.is_file()

    def test_oversized_sources_are_scaled_with_a_recorded_warning(self, tmp_path):
        writer = AnnotatedVideoWriter(
            tmp_path / "big.mp4", 25.0, (3840, 2160), max_width=1280
        )
        assert writer.width == 1280
        assert writer.height == 720
        assert any("1280x720" in w for w in writer.warnings)

    def test_odd_dimensions_are_made_even_for_h264(self, tmp_path):
        """H.264 chroma subsampling cannot encode an odd width or height."""
        writer = AnnotatedVideoWriter(tmp_path / "odd.mp4", 25.0, (641, 481))
        assert writer.width % 2 == 0
        assert writer.height % 2 == 0

    @pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not installed")
    def test_ffprobe_reports_what_is_actually_in_the_file(self):
        probe = probe_media(SAMPLE)
        assert probe["width"] == 960
        assert probe["height"] == 540
        assert probe["frames"] == 210
        assert probe["has_audio"] is False


# ══════════════════════════════════════════════════════════════════════════
#  API surface
# ══════════════════════════════════════════════════════════════════════════
class TestResultApi:
    def test_result_returns_urls_never_filesystem_paths(self, client, analysed):
        response = client.get(f"/api/videos/jobs/{analysed.job_id}/result")
        assert response.status_code == 200
        body = response.json()

        assert body["annotated_ready"] is True
        assert body["video_url"] == f"/api/videos/jobs/{analysed.job_id}/video"
        assert body["download_url"]
        assert body["frames"] == analysed.processed_frames
        assert body["duration_seconds"] > 0
        assert body["browser_compatible"] is True

        # Nothing in the payload may leak a host path.
        serialised = response.text
        assert str(settings.processed_root) not in serialised
        assert str(settings.upload_root) not in serialised

    def test_the_video_streams_and_supports_range_requests(self, client, analysed):
        """Range support is what lets the browser seek without a full download."""
        whole = client.get(f"/api/videos/jobs/{analysed.job_id}/video")
        assert whole.status_code == 200
        assert whole.headers["content-type"] == "video/mp4"
        assert "inline" in whole.headers.get("content-disposition", "")

        part = client.get(
            f"/api/videos/jobs/{analysed.job_id}/video",
            headers={"Range": "bytes=0-1023"},
        )
        assert part.status_code == 206
        assert part.headers["accept-ranges"] == "bytes"
        assert "content-range" in part.headers
        assert len(part.content) == 1024

    def test_downloads_are_offered_as_attachments(self, client, analysed):
        annotated = client.get(f"/api/videos/jobs/{analysed.job_id}/download")
        assert annotated.status_code == 200
        assert "attachment" in annotated.headers.get("content-disposition", "")
        assert len(annotated.content) == render_path(analysed).stat().st_size

        original = client.get(f"/api/videos/jobs/{analysed.job_id}/original")
        assert original.status_code == 200
        assert "attachment" in original.headers.get("content-disposition", "")

    def test_the_timeline_carries_seek_targets(self, client, analysed):
        body = client.get(f"/api/videos/jobs/{analysed.job_id}/result").json()
        for entry in body["timeline"]:
            assert entry["video_timestamp"] is not None
            assert entry["label"]
            assert entry["severity"]
        positions = [e["video_timestamp"] for e in body["timeline"]]
        assert positions == sorted(positions), "timeline must be in playback order"

    def test_a_running_job_says_the_video_is_not_ready(self, client):
        import uuid

        job_id = str(uuid.uuid4())
        with session_scope() as session:
            session.add(
                VideoJob(
                    job_id=job_id, filename="x.mp4", stored_path="/tmp/x.mp4",
                    profile="fast", status=JobStatus.RUNNING,
                )
            )
        result = client.get(f"/api/videos/jobs/{job_id}/result").json()
        assert result["annotated_ready"] is False
        assert result["video_url"] is None
        assert client.get(f"/api/videos/jobs/{job_id}/video").status_code == 404

    def test_unknown_jobs_404(self, client):
        for suffix in ("result", "video", "download", "original"):
            response = client.get(f"/api/videos/jobs/does-not-exist/{suffix}")
            assert response.status_code == 404


# ══════════════════════════════════════════════════════════════════════════
#  Browser playability, verified with a real decoder
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not installed")
class TestBrowserPlayback:
    def test_the_render_decodes_end_to_end_without_errors(self, analysed):
        """A full decode pass. Catches truncated or corrupt output."""
        completed = subprocess.run(  # noqa: S603
            [
                ffmpeg_path(), "-v", "error", "-i", str(render_path(analysed)),
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=180, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr.strip() == "", completed.stderr

    def test_the_pixel_format_is_the_one_browsers_require(self, analysed):
        completed = subprocess.run(  # noqa: S603
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=pix_fmt", "-of", "csv=p=0",
                str(render_path(analysed)),
            ],
            capture_output=True, text=True, timeout=60, check=False,
        )
        # Safari will not play 4:4:4 or 10-bit H.264.
        assert completed.stdout.strip() == "yuv420p"

    def test_seeking_lands_on_the_expected_frame(self, analysed):
        """Seek to the middle and confirm a frame comes back."""
        capture = cv2.VideoCapture(str(render_path(analysed)))
        target = analysed.processed_frames // 2
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = capture.read()
        landed = capture.get(cv2.CAP_PROP_POS_FRAMES)
        capture.release()
        assert ok and frame is not None
        assert abs(landed - (target + 1)) <= 2


# ══════════════════════════════════════════════════════════════════════════
#  Memory-capped hosts: working resolution and the watchdog
# ══════════════════════════════════════════════════════════════════════════
def _write_clip(path: Path, width: int, height: int, frames: int = 12) -> None:
    """A short synthetic clip at an exact geometry."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    assert writer.isOpened()
    rng = np.random.default_rng(7)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    writer.release()


class _ShapeSpy:
    """Wraps the real detector and records the frames it was handed."""

    def __init__(self, detector) -> None:
        self._detector = detector
        self.shapes: list[tuple[int, int]] = []

    def infer(self, frame):
        self.shapes.append(frame.shape[:2])
        return self._detector.infer(frame)

    def __getattr__(self, name):
        return getattr(self._detector, name)


class TestRenderFreeSafeguards:
    def test_working_size_scales_wide_sources_and_leaves_others_alone(self):
        from backend.jobs.video_jobs import working_size

        assert working_size(3840, 2160, 1280) == (1280, 720)
        assert working_size(1920, 1080, 1280) == (1280, 720)
        assert working_size(1280, 720, 1280) == (1280, 720)
        assert working_size(640, 360, 1280) == (640, 360)
        # Odd dimensions are made even, as H.264 requires; not scaled.
        assert working_size(641, 481, 1280) == (640, 480)

    def test_frames_are_downscaled_before_inference(self, tmp_path, monkeypatch):
        """Detection sees the working frame, not the source frame.

        The point of the cap is memory: a frame that reaches the detector at
        source size has already cost the full-resolution copies the overlay
        pass makes. So the check is on what `infer` receives, not on the file.
        """
        from backend.inference.registry import resolve_detector
        from backend.jobs import video_jobs

        source = tmp_path / "wide.mp4"
        _write_clip(source, 1920, 1080)
        spy = _ShapeSpy(resolve_detector())
        monkeypatch.setattr(video_jobs, "resolve_detector", lambda: spy)
        monkeypatch.setattr(settings, "annotated_max_width", 640)

        job = run_job(source, profile="fast")

        assert job.status == JobStatus.COMPLETED, job.message
        assert spy.shapes and all(shape == (360, 640) for shape in spy.shapes)
        assert job.output["resolution"] == "640x360"
        assert job.resolution == "1920x1080"  # the source is still described
        assert any("640x360" in w for w in job.output["warnings"])
        info = source_properties(render_path(job))
        assert (info["width"], info["height"]) == (640, 360)
        assert info["frames"] == job.processed_frames

    def test_small_sources_are_not_upscaled_or_touched(self, tmp_path, monkeypatch):
        from backend.inference.registry import resolve_detector
        from backend.jobs import video_jobs

        source = tmp_path / "small.mp4"
        _write_clip(source, 320, 240)
        spy = _ShapeSpy(resolve_detector())
        monkeypatch.setattr(video_jobs, "resolve_detector", lambda: spy)
        monkeypatch.setattr(settings, "annotated_max_width", 1280)

        job = run_job(source, profile="fast")

        assert job.status == JobStatus.COMPLETED, job.message
        assert all(shape == (240, 320) for shape in spy.shapes)
        assert job.output["resolution"] == "320x240"
        assert not any("rather than the source" in w for w in job.output["warnings"])

    def test_the_rss_watchdog_fails_the_job_and_cleans_up(self, tmp_path, monkeypatch):
        from backend.jobs import video_jobs

        source = tmp_path / "clip.mp4"
        _write_clip(source, 320, 240, frames=30)
        monkeypatch.setattr(settings, "job_max_rss_mb", 470)
        monkeypatch.setattr(video_jobs, "_process_rss_mb", lambda: 480.0)
        before = set(settings.processed_root.glob("*"))

        job = run_job(source, profile="fast")

        assert job.status == JobStatus.FAILED
        assert "memory" in job.message.lower()
        assert "470" in job.message
        assert job.finished_at is not None
        assert job.annotated_path is None
        # The partial render was removed and nothing else was left behind.
        assert set(settings.processed_root.glob("*")) == before
        # The upload is untouched, so the operator can retry elsewhere.
        assert Path(job.stored_path).is_file()

    def test_the_rss_watchdog_is_off_by_default(self, tmp_path, monkeypatch):
        from backend.jobs import video_jobs

        source = tmp_path / "clip.mp4"
        _write_clip(source, 320, 240)
        monkeypatch.setattr(settings, "job_max_rss_mb", 0)
        monkeypatch.setattr(video_jobs, "_process_rss_mb", lambda: 9999.0)
        job = run_job(source, profile="fast")
        assert job.status == JobStatus.COMPLETED, job.message

    def test_the_wall_clock_limit_fails_the_job_and_cleans_up(self, tmp_path, monkeypatch):
        source = tmp_path / "clip.mp4"
        _write_clip(source, 320, 240, frames=30)
        # Small enough that the first progress tick is already past it.
        monkeypatch.setattr(settings, "job_max_seconds", 1e-6)
        before = set(settings.processed_root.glob("*"))

        job = run_job(source, profile="fast")

        assert job.status == JobStatus.FAILED
        assert "JOB_MAX_SECONDS" in job.message
        assert job.annotated_path is None
        assert set(settings.processed_root.glob("*")) == before

    def test_the_release_is_unaffected_by_a_watchdog_failure(self, tmp_path, monkeypatch):
        """A failed job must not leave the decoder open on the upload.

        `capture.release()` runs in the `finally`; if it did not, the upload
        could not be deleted on Windows and would leak a decoder everywhere.
        """
        from backend.jobs import video_jobs

        released: list[bool] = []

        class _Capture:
            """Delegates to a real capture (subclassing cv2 types is unsafe)."""

            def __init__(self, *args) -> None:
                self._capture = cv2.VideoCapture(*args)

            def release(self) -> None:
                released.append(True)
                self._capture.release()

            def __getattr__(self, name):
                return getattr(self._capture, name)

        class _Cv2:
            VideoCapture = _Capture

            def __getattr__(self, name):
                return getattr(cv2, name)

        monkeypatch.setattr(video_jobs, "cv2", _Cv2())
        source = tmp_path / "clip.mp4"
        _write_clip(source, 320, 240)
        monkeypatch.setattr(settings, "job_max_rss_mb", 1)
        monkeypatch.setattr(video_jobs, "_process_rss_mb", lambda: 2.0)

        job = run_job(source, profile="fast")
        assert job.status == JobStatus.FAILED
        assert released == [True]

    def test_rss_reader_never_raises(self):
        """Returns a number on Linux and None elsewhere; never an exception."""
        from backend.jobs.video_jobs import _process_rss_mb

        value = _process_rss_mb()
        assert value is None or value > 0

    def test_lowering_thread_priority_is_safe_everywhere(self):
        from backend.jobs.video_jobs import _lower_thread_priority

        _lower_thread_priority()  # must not raise on any platform
