"""Video sources, the rolling clip buffer, and pipeline error containment."""

from __future__ import annotations

import numpy as np
import pytest

from backend.video.ring_buffer import FrameRingBuffer
from backend.video.source import SourceStatus, VideoSource, probe_video_file


def frame(value: int = 90, shape=(120, 160)) -> np.ndarray:
    return np.full((*shape, 3), value, np.uint8)


# ══════════════════════════════════════════════════════════════════════════
#  Ring buffer
# ══════════════════════════════════════════════════════════════════════════
class TestRingBuffer:
    def test_eviction_is_time_based(self):
        """The window must mean the same at any frame rate."""
        buffer = FrameRingBuffer(seconds=2.0)
        for i in range(100):
            buffer.append(frame(), i * 0.1)  # 10 fps, 10s of footage
        stats = buffer.stats()
        assert stats["seconds_buffered"] <= 2.1
        assert stats["frames"] <= 22

    def test_frames_decode_back(self):
        buffer = FrameRingBuffer(seconds=5.0, jpeg_quality=95)
        original = frame(123)
        buffer.append(original, 0.0)
        decoded = buffer.snapshot()[0].decode()
        assert decoded is not None
        assert decoded.shape == original.shape
        # JPEG is lossy, so compare approximately.
        assert abs(int(decoded.mean()) - 123) < 4

    def test_memory_stays_bounded(self):
        """Raw frames would cost ~150 MB per camera; JPEG keeps it affordable."""
        buffer = FrameRingBuffer(seconds=8.0)
        for i in range(300):
            buffer.append(frame(shape=(540, 960)), i * 0.08)
        assert buffer.memory_bytes < 30_000_000, buffer.stats()

    def test_snapshot_since_filters(self):
        buffer = FrameRingBuffer(seconds=10.0)
        for i in range(10):
            buffer.append(frame(), i * 0.1)
        assert len(buffer.snapshot(since=0.5)) == 5

    def test_zero_window_stores_nothing(self):
        buffer = FrameRingBuffer(seconds=0.0)
        buffer.append(frame(), 0.0)
        assert buffer.frame_count == 0

    def test_clear(self):
        buffer = FrameRingBuffer(seconds=5.0)
        buffer.append(frame(), 0.0)
        buffer.clear()
        assert buffer.frame_count == 0 and buffer.memory_bytes == 0

    def test_latest_returns_newest(self):
        buffer = FrameRingBuffer(seconds=5.0)
        for i in range(5):
            buffer.append(frame(), i * 0.1)
        latest = buffer.latest()
        assert latest is not None and latest.timestamp == pytest.approx(0.4)


# ══════════════════════════════════════════════════════════════════════════
#  Source error handling (§29)
# ══════════════════════════════════════════════════════════════════════════
class TestSourceErrors:
    @pytest.mark.parametrize(
        "source_type,url",
        [
            ("file", "does-not-exist.mp4"),
            ("webcam", "not-a-number"),
            ("file", "../../etc/passwd"),
            ("file", "/etc/passwd"),
        ],
    )
    def test_bad_sources_fail_without_raising(self, source_type, url):
        """One bad camera must never take the server down."""
        source = VideoSource(source_type, url, camera_id="t", name="bad")
        assert source.open() is False
        assert source.status is SourceStatus.ERROR
        assert source.last_error
        source.close()

    def test_corrupt_file_is_rejected(self, tmp_path):
        path = tmp_path / "broken.mp4"
        path.write_bytes(b"this is definitely not a video container")
        result = probe_video_file(path)
        assert result["ok"] is False and result["error"]

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.mp4"
        path.write_bytes(b"")
        assert probe_video_file(path)["ok"] is False

    def test_backoff_grows_and_is_capped(self):
        from backend.config import settings

        source = VideoSource("rtsp", "rtsp://192.0.2.1:554/x", camera_id="t")
        delays = [source.schedule_retry() for _ in range(8)]
        assert delays == sorted(delays), "backoff must be non-decreasing"
        assert max(delays) <= settings.reconnect_max_backoff_seconds

    def test_retry_only_when_faulted(self):
        source = VideoSource("rtsp", "rtsp://192.0.2.1:554/x", camera_id="t")
        assert source.status is SourceStatus.IDLE
        assert source.should_retry() is False

    def test_info_is_serialisable(self):
        source = VideoSource("file", "x.mp4", camera_id="t", name="n")
        info = source.info()
        for key in ("camera_id", "status", "source_type", "last_error", "loop"):
            assert key in info


# ══════════════════════════════════════════════════════════════════════════
#  Real file playback
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def sample_video():
    """A short synthetic clip written into the upload directory."""
    import cv2

    from backend.config import settings

    settings.ensure_directories()
    path = settings.upload_root / "unit-sample.mp4"
    if not path.exists():
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
        )
        for i in range(30):
            canvas = np.full((120, 160, 3), 40, np.uint8)
            cv2.rectangle(canvas, (i * 4, 40), (i * 4 + 30, 100), (200, 200, 200), -1)
            writer.write(canvas)
        writer.release()
    return path


class TestFilePlayback:
    def test_opens_and_reports_properties(self, sample_video):
        source = VideoSource("file", sample_video.name, camera_id="t", loop=False)
        assert source.open() is True
        assert source.status is SourceStatus.ONLINE
        assert source.total_frames > 0
        assert (source.width, source.height) == (160, 120)
        source.close()

    def test_reads_every_frame_then_ends(self, sample_video):
        """File sources must not drop frames — every frame gets analysed."""
        source = VideoSource("file", sample_video.name, camera_id="t", loop=False)
        assert source.open()
        count = 0
        while (data := source.read()) is not None:
            count += 1
            assert data.frame.shape == (120, 160, 3)
            if count > 200:
                pytest.fail("file source did not terminate")
        assert count >= 25
        assert source.status is SourceStatus.ENDED
        source.close()

    def test_loop_rewinds_instead_of_ending(self, sample_video):
        """File replay is a supported camera mode; it must not stop at the end."""
        source = VideoSource("file", sample_video.name, camera_id="t", loop=True)
        assert source.open()
        for _ in range(80):  # more than the clip contains
            assert source.read() is not None
        assert source.status is SourceStatus.ONLINE
        assert source.loops_completed >= 1
        source.close()

    def test_probe_reports_metadata(self, sample_video):
        result = probe_video_file(sample_video)
        assert result["ok"] is True
        assert result["width"] == 160 and result["height"] == 120
        assert result["total_frames"] > 0


# ══════════════════════════════════════════════════════════════════════════
#  Pipeline
# ══════════════════════════════════════════════════════════════════════════
class TestPipeline:
    def test_runs_end_to_end_and_raises_events(self, sample_video, camera):
        """Full chain on a real file: decode -> detect -> track -> analyse."""
        import time
        import uuid

        from backend.analysis.zones import ResolvedZone
        from backend.events.engine import CooldownRegistry, EventEngine
        from backend.inference.backends.mock_backend import MockDetector
        from backend.inference.ppe import NullPPEDetector
        from backend.video.pipeline import CameraPipeline

        detector = MockDetector(people=3)
        detector.load()
        engine = EventEngine(
            cooldown=CooldownRegistry(default_seconds=0.0, max_per_minute=0),
            persist=False,
            broadcast=False,
            capture_evidence=False,
        )
        zone = ResolvedZone(
            zone_id=str(uuid.uuid4()),
            name="Whole frame",
            zone_type="restricted",
            polygon_norm=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        )
        pipeline = CameraPipeline(
            camera_id=camera.camera_id,
            name="Pipeline Test",
            source_type="file",
            stream_url=sample_video.name,
            detector=detector,
            ppe_detector=NullPPEDetector(),
            engine=engine,
            zones=[zone],
            profile="thorough",
            broadcast=False,
            persist_telemetry=False,
            loop=False,
        )
        pipeline.start()
        deadline = time.monotonic() + 25
        while pipeline.is_running and time.monotonic() < deadline:
            time.sleep(0.2)
        pipeline.stop()

        assert pipeline.stats.frames_processed > 0
        assert pipeline.stats.frames_detected > 0
        # Everyone is inside the whole-frame restricted zone.
        assert pipeline.stats.events_raised > 0
        assert engine.stats()["accepted"] > 0

    def test_unreachable_source_does_not_crash_the_worker(self, camera):
        """Spec §29: a dead stream backs off; it does not kill the thread."""
        import time

        from backend.inference.backends.mock_backend import MockDetector
        from backend.inference.ppe import NullPPEDetector
        from backend.video.pipeline import CameraPipeline

        detector = MockDetector()
        detector.load()
        pipeline = CameraPipeline(
            camera_id=camera.camera_id,
            name="Dead Source",
            source_type="file",
            stream_url="absent-file.mp4",
            detector=detector,
            ppe_detector=NullPPEDetector(),
            broadcast=False,
            persist_telemetry=False,
        )
        pipeline.start()
        time.sleep(1.5)
        assert pipeline.is_running, "worker died on an unreachable source"
        assert pipeline.status in {"error", "offline", "connecting", "idle"}
        pipeline.stop()
        assert not pipeline.is_running

    def test_runtime_snapshot_is_complete(self, camera):
        from backend.video.pipeline import CameraPipeline

        pipeline = CameraPipeline(
            camera_id=camera.camera_id,
            name="Reporting",
            source_type="file",
            stream_url="x.mp4",
            broadcast=False,
            persist_telemetry=False,
        )
        runtime = pipeline.runtime()
        for key in ("camera_id", "status", "fps", "people_count", "backend", "buffer"):
            assert key in runtime
