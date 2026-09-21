"""Background analysis of uploaded video.

Runs the same detect → track → analyse → event chain as a live camera, but
over a file, as fast as the hardware allows.

The **primary output is a full annotated render**: every frame of the source,
in order, with tracks, PPE verdicts, zones and event banners drawn on it, so an
operator can watch the analysed footage as footage. Per-event snapshots and
clips are still captured, but they are secondary evidence — stills cannot show
whether a box actually follows the person it claims to.

Frames are streamed through: read → detect → annotate → write → release. No
frame buffer accumulates, so a two-hour upload costs the same memory as a
two-minute one.

Why a thread pool rather than the request: a 60-second clip takes tens of
seconds to analyse. Doing it inline would hold an HTTP connection open, block
a worker and time out behind a proxy. The upload endpoint returns a `job_id`
immediately; progress is polled from `/api/videos/jobs/{id}` and pushed over
the WebSocket.

Concurrency is capped (`MAX_CONCURRENT_JOBS`, default 2) because each job
holds a decoder and competes with the live cameras for the same inference
session.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from backend.analysis import build_analysers
from backend.analysis.base import FrameContext
from backend.analysis.movement import MovementAnalyser
from backend.config import PROJECT_ROOT, settings
from backend.db.base import session_scope
from backend.db.models import VideoJob
from backend.events.bus import get_bus
from backend.events.engine import get_event_engine
from backend.events.types import label_for
from backend.inference.hardhat import resolve_hardhat_validator
from backend.inference.ppe import resolve_ppe_detector
from backend.inference.registry import resolve_detector
from backend.logging_conf import get_logger
from backend.tracking import ByteTracker
from backend.tracking.recorder import TrackRecorder
from backend.video.annotate import ActiveEvent, annotate_frame
from backend.video.writer import AnnotatedVideoWriter, VideoWriteError

logger = get_logger(__name__)

#: Progress is pushed at most this often, to avoid flooding the WebSocket.
PROGRESS_INTERVAL = 0.75


#: Event types counted as PPE violations in the progress readout.
PPE_EVENT_TYPES = frozenset({"MISSING_HELMET", "MISSING_VEST", "PPE_VIOLATION"})


def _timecode(seconds: float) -> str:
    """`mm:ss` burned into the render, so a frame can be located in the source."""
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _render_name(filename: str, job_id: str) -> str:
    """Output filename: recognisable, unique, and safe to put in a path.

    The operator's own name is kept so a downloaded file is identifiable, but
    it is sanitised and suffixed with the job id — two uploads of `video.mp4`
    must not overwrite each other's renders.
    """
    from backend.events.evidence import sanitise

    stem = sanitise(Path(filename).stem, 60) or "video"
    return f"{stem}_{job_id[:8]}_annotated.mp4"


def _load_zones(camera_id: str | None) -> list:
    """Zones for the camera this footage is being analysed against."""
    if not camera_id:
        return []
    try:
        from backend.db.repository import load_zones_for_camera

        return list(load_zones_for_camera(camera_id))
    except Exception as exc:
        logger.warning("Could not load zones for camera %s: %s", camera_id[:8], exc)
        return []


def _stored_path(path: Path) -> str:
    """Path as stored in the database.

    Relative to the project root when the render lives inside it, which keeps
    a development database portable; absolute otherwise — PROCESSED_DIR on a
    mounted disk (DATA_ROOT=/var/data) is not under the project at all. The
    reader, ``_resolve_render`` in the videos API, accepts both and confines
    either to PROCESSED_DIR before opening it.
    """
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


#: Nice value for the analysis thread. Positive is *lower* priority: the
#: server's event loop and the health probe get the CPU first whenever they
#: want it, and the job takes what is left — which on a throttled host is
#: nearly all of it anyway, because they want it rarely.
WORKER_NICE = 10


def _lower_thread_priority() -> None:
    """Deprioritise the calling thread so the web server stays responsive.

    On a fractional-CPU host the analysis loop can otherwise starve Uvicorn
    long enough for the platform's health check to fail and restart the
    process. Linux schedules threads as tasks, so ``setpriority`` on the
    native thread id affects this thread only; elsewhere (macOS, Windows) the
    call would apply to the whole process or not exist, so it is skipped.
    Best effort: a refusal is logged, never raised.
    """
    if not sys.platform.startswith("linux") or not hasattr(os, "setpriority"):
        return
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), WORKER_NICE)
    except (OSError, AttributeError) as exc:
        logger.debug("Could not lower analysis thread priority: %s", exc)


def _process_rss_mb() -> float | None:
    """Resident set size of this process in MB, or None where unknown.

    Read from ``/proc/self/statm`` — no dependency, and cheap enough to call
    every progress tick. Only Linux has it, which is where the memory-capped
    hosts are.
    """
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def working_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Resolution the whole pipeline runs at for a source of `width`x`height`.

    Sources wider than `max_width` are scaled down proportionally, once, right
    after decode — so detection, tracking, PPE, annotation and the render all
    work on the smaller frame and a 4K upload costs the same memory as a
    1280-wide one. Both dimensions are made even, which is what H.264 chroma
    subsampling requires and what the writer would otherwise do itself.
    """
    if width <= max_width:
        return width - (width % 2), height - (height % 2)
    scale = max_width / width
    return max_width - (max_width % 2), max(2, int(round(height * scale)) & ~1)


class JobStatus:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class VideoJobRunner:
    """Owns the worker pool and the cancellation flags."""

    def __init__(self, max_workers: int | None = None) -> None:
        if max_workers is None:
            max_workers = settings.max_concurrent_jobs
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="videojob"
        )
        self._cancelled: set[str] = set()
        self._active: set[str] = set()
        self._lock = threading.Lock()

    # ── control ──────────────────────────────────────────────────────────
    def submit(self, job_id: str, profile: str = "standard") -> None:
        with self._lock:
            self._active.add(job_id)
        self._pool.submit(self._run, job_id, profile)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._active:
                return False
            self._cancelled.add(job_id)
        return True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def shutdown(self) -> None:
        with self._lock:
            self._cancelled.update(self._active)
        self._pool.shutdown(wait=False)

    # ── persistence helpers ──────────────────────────────────────────────
    @staticmethod
    def _update(job_id: str, **fields: Any) -> None:
        try:
            with session_scope() as session:
                job = session.get(VideoJob, job_id)
                if job is None:
                    return
                for key, value in fields.items():
                    setattr(job, key, value)
        except Exception as exc:
            logger.error("Could not update job %s: %s", job_id[:8], exc)

    @staticmethod
    def _publish(job_id: str, **payload: Any) -> None:
        get_bus().publish("job_progress", {"job_id": job_id, **payload})

    def _fail(self, job_id: str, message: str) -> None:
        logger.warning("Video job %s failed: %s", job_id[:8], message)
        self._update(
            job_id,
            status=JobStatus.FAILED,
            message=message,
            finished_at=datetime.now(UTC),
        )
        self._publish(job_id, status=JobStatus.FAILED, message=message, progress=0.0)

    # ── the work ─────────────────────────────────────────────────────────
    def _run(self, job_id: str, profile: str) -> None:
        _lower_thread_priority()
        try:
            self._analyse(job_id, profile)
        except Exception as exc:
            logger.exception("Video job %s crashed", job_id[:8])
            self._fail(job_id, f"Analysis failed: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._active.discard(job_id)
                self._cancelled.discard(job_id)

    def _analyse(self, job_id: str, profile: str) -> None:
        with session_scope() as session:
            job = session.get(VideoJob, job_id)
            if job is None:
                logger.error("Video job %s not found", job_id[:8])
                return
            source_path = Path(job.stored_path)
            filename = job.filename
            zone_camera_id = job.zone_camera_id

        if not source_path.is_file():
            self._fail(job_id, "Uploaded file is no longer on disk")
            return

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            self._fail(job_id, "The decoder could not open this file")
            return

        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if source_fps <= 0:
                source_fps = 25.0  # some containers omit it
            if width <= 0 or height <= 0:
                self._fail(job_id, "The file reports no valid video dimensions")
                return

            started = datetime.now(UTC)
            self._update(
                job_id,
                status=JobStatus.RUNNING,
                started_at=started,
                total_frames=total_frames,
                fps=round(source_fps, 2),
                resolution=f"{width}x{height}",
                duration_seconds=round(total_frames / source_fps, 2) if total_frames else 0.0,
                message="Analysing",
                progress=0.0,
            )
            self._publish(job_id, status=JobStatus.RUNNING, progress=0.0, message="Analysing")

            detector = resolve_detector()
            ppe_detector = resolve_ppe_detector()
            # Named on every rendered frame when it is anything but the trained
            # validator, so an unverifiable helmet verdict is visible in the
            # video and not only in diagnostics.
            hardhat_method = resolve_hardhat_validator().method
            tracker = ByteTracker()
            analysers = build_analysers(profile)
            # Uploaded-video people are recorded against the job, not a camera,
            # so the People Detected KPI counts them too.
            recorder = TrackRecorder(camera_id=f"upload:{job_id[:8]}", job_id=job_id)
            # Uploaded-video analysis is deliberately deduplicated the same way
            # as live: an operator reviewing results wants incidents, not frames.
            engine = get_event_engine()

            # ── the primary output ───────────────────────────────────────
            # A render that cannot be produced is a failed analysis, not a
            # quiet downgrade: the annotated video is what the operator came
            # for, and reporting "complete" without one would be a lie.
            # Everything downstream of the decoder — detection, tracking,
            # PPE, annotation, evidence and the render — runs at this size.
            # Scaling once here, rather than in the writer at the end, is
            # what keeps a 1080p or 4K upload inside a 512 MB host: every
            # full-frame copy the overlay pass makes is of the small frame.
            work_width, work_height = working_size(
                width, height, settings.annotated_max_width
            )
            downscale = (work_width, work_height) != (width, height)

            annotated_path = settings.processed_root / _render_name(filename, job_id)
            writer: AnnotatedVideoWriter | None = None
            if settings.annotated_video_enabled:
                # The writer is handed the working size, so it never resizes
                # again; the note it would have recorded is kept in the output.
                writer = AnnotatedVideoWriter(
                    annotated_path, source_fps, (work_width, work_height)
                )
                if width > settings.annotated_max_width:
                    writer.warnings.append(
                        f"Rendered at {work_width}x{work_height} rather than the "
                        f"source {width}x{height} "
                        f"(ANNOTATED_MAX_WIDTH={settings.annotated_max_width})"
                    )
                try:
                    writer.open()
                except VideoWriteError as exc:
                    self._fail(job_id, f"Could not start the annotated video: {exc}")
                    return

            # Uploaded video has no camera of its own, so zone geometry only
            # exists if the upload named a camera to borrow zones from. Without
            # one, the zone-dependent rules (intrusion, per-zone loitering)
            # correctly find nothing rather than inventing a boundary.
            zones = _load_zones(zone_camera_id)
            if zone_camera_id and not zones:
                logger.info(
                    "Job %s: camera %s has no usable zones",
                    job_id[:8], str(zone_camera_id)[:8],
                )
            job_scope = f"upload-{job_id[:8]}"

            frame_index = 0
            processed = 0
            events_created = 0
            people_seen: set[int] = set()
            type_counts: dict[str, int] = {}
            last_progress = 0.0
            detect_every = max(1, settings.detect_every_n_frames)
            tracks: list = []
            ppe_results: dict = {}
            # Banners currently on screen, and when each stops being drawn.
            active_events: list[ActiveEvent] = []
            ppe_violations = 0
            inference_ms = 0.0
            wall_start = time.monotonic()

            while True:
                if self.is_cancelled(job_id):
                    self._update(
                        job_id,
                        status=JobStatus.CANCELLED,
                        message="Cancelled by operator",
                        finished_at=datetime.now(UTC),
                    )
                    self._publish(job_id, status=JobStatus.CANCELLED, message="Cancelled")
                    return

                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                if downscale:
                    # The full-size decode is dropped as soon as this returns;
                    # nothing below ever sees it.
                    frame = cv2.resize(
                        frame, (work_width, work_height), interpolation=cv2.INTER_AREA
                    )

                # Virtual clock from the source timeline, so time-based rules
                # (loitering, fall persistence) measure *video* seconds — not
                # how long the analysis took.
                video_time = frame_index / source_fps

                if frame_index % detect_every == 0:
                    result = detector.infer(frame)
                    inference_ms = result.inference_ms
                    people = [d for d in result.detections if d.label == "person"]
                    tracks = tracker.update(people, timestamp=video_time)
                    people_seen.update(t.track_id for t in tracks)
                    if tracks and ppe_detector is not None:
                        try:
                            ppe_results = ppe_detector.assess(
                                frame, tracks, people, scope=job_scope
                            )
                        except Exception as exc:
                            logger.debug("PPE failed on job frame: %s", exc)

                    recorder.observe(
                        tracks,
                        timestamp=video_time,
                        speeds={
                            t.track_id: a.speed_for(t)
                            for a in analysers
                            if isinstance(a, MovementAnalyser)
                            for t in tracks
                        },
                        ppe=ppe_results,
                    )
                    recorder.flush({t.track_id for t in tracker.tracks})

                    context = FrameContext(
                        camera_id=f"upload:{job_id[:8]}",
                        frame_index=frame_index,
                        timestamp=video_time,
                        wall_time=time.time(),
                        frame_width=work_width,
                        frame_height=work_height,
                        tracks=tracks,
                        zones=zones,
                        ppe=ppe_results,
                    )
                    candidates = []
                    for analyser in analysers:
                        try:
                            candidates.extend(analyser.analyse(context))
                        except Exception as exc:
                            logger.debug("Analyser %s failed: %s", analyser.name, exc)

                    if candidates:
                        speeds = next(
                            (
                                {t.track_id: a.speed_for(t) for t in tracks}
                                for a in analysers
                                if isinstance(a, MovementAnalyser)
                            ),
                            {},
                        )
                        annotated = annotate_frame(
                            frame, tracks, zones, ppe_results, speeds,
                            camera_name=filename,
                            stats={"fps": source_fps, "people_count": len(tracks),
                                   "backend": detector.name},
                            timestamp=f"t={video_time:.1f}s",
                        )
                        accepted = engine.submit(
                            candidates,
                            camera_id=None,
                            scope=job_scope,
                            frame=frame,
                            annotated_frame=annotated,
                            buffer=None,     # no rolling clip for file analysis
                            fps=source_fps,
                            monotonic_time=video_time,
                            job_id=job_id,
                            video_timestamp=round(video_time, 3),
                        )
                        events_created += len(accepted)
                        recorder.note_events([r.person_id for r in accepted])
                        for record in accepted:
                            type_counts[record.event_type] = (
                                type_counts.get(record.event_type, 0) + 1
                            )
                            if record.event_type in PPE_EVENT_TYPES:
                                ppe_violations += 1
                            # Hold the banner on screen for a watchable
                            # interval of *video* time, so it survives into
                            # the render at the moment it describes.
                            active_events.append(
                                ActiveEvent(
                                    label=label_for(record.event_type),
                                    severity=record.severity,
                                    person_id=record.person_id,
                                    expires_at=video_time
                                    + settings.event_overlay_seconds,
                                )
                            )

                # Every source frame produces exactly one output frame, in
                # order — including the frames between detections, which reuse
                # the current tracks so the render never freezes or drifts out
                # of sync with the source timeline.
                if writer is not None:
                    active_events = [
                        e for e in active_events if e.expires_at > video_time
                    ]
                    speeds = next(
                        (
                            {t.track_id: a.speed_for(t) for t in tracks}
                            for a in analysers
                            if isinstance(a, MovementAnalyser)
                        ),
                        {},
                    )
                    # `copy=False`: detection and event capture are both done
                    # with this frame, so drawing on it directly saves a
                    # full-frame copy on every frame of the video.
                    writer.write(
                        annotate_frame(
                            frame, tracks, zones, ppe_results, speeds,
                            camera_name=filename,
                            stats={
                                "fps": source_fps,
                                "inference_ms": inference_ms,
                                "people_count": len(tracks),
                                "backend": detector.name,
                                "ppe_method": ppe_detector.method if ppe_detector else "",
                                "hardhat_method": hardhat_method,
                            },
                            timestamp=_timecode(video_time),
                            copy=False,
                            events=active_events,
                        )
                    )

                processed += 1
                now = time.monotonic()
                if now - last_progress >= PROGRESS_INTERVAL:
                    last_progress = now
                    progress = (
                        min(0.99, frame_index / total_frames) if total_frames else 0.0
                    )
                    # Real rates from real counters. A progress bar that
                    # interpolates is worse than none: it hides a stall.
                    spent = now - wall_start
                    rate = processed / spent if spent > 0 else 0.0
                    remaining = (
                        round((total_frames - frame_index) / rate, 1)
                        if total_frames and rate > 0
                        else None
                    )
                    rss = _process_rss_mb()
                    logger.info(
                        "Job %s: %.1f%% frame %d/%d, %.1f fps, rss %s MB",
                        job_id[:8], progress * 100, frame_index, total_frames, rate,
                        f"{rss:.0f}" if rss is not None else "n/a",
                    )
                    # Watchdog. Failing one job here, with a reason in the
                    # record, beats the host killing the whole process with
                    # nothing in the log and no job left to show for it.
                    limit_mb = settings.job_max_rss_mb
                    if limit_mb and rss is not None and rss > limit_mb:
                        if writer is not None:
                            writer.abort()
                        self._fail(
                            job_id,
                            f"Stopped at {progress * 100:.0f}%: memory use "
                            f"({rss:.0f} MB) approached the host's limit "
                            f"(JOB_MAX_RSS_MB={limit_mb}). Try a shorter or "
                            f"lower-resolution video.",
                        )
                        return
                    limit_s = settings.job_max_seconds
                    if limit_s and spent > limit_s:
                        if writer is not None:
                            writer.abort()
                        self._fail(
                            job_id,
                            f"Stopped at {progress * 100:.0f}%: analysis exceeded "
                            f"the {limit_s:.0f}s limit (JOB_MAX_SECONDS) at "
                            f"{rate:.1f} fps. Try a shorter video.",
                        )
                        return
                    self._update(
                        job_id, progress=round(progress, 4), processed_frames=processed,
                        events_created=events_created,
                        people_detected=len(people_seen),
                    )
                    self._publish(
                        job_id,
                        status=JobStatus.RUNNING,
                        progress=round(progress, 4),
                        processed_frames=processed,
                        total_frames=total_frames,
                        events_created=events_created,
                        people_detected=len(people_seen),
                        ppe_violations=ppe_violations,
                        processing_fps=round(rate, 1),
                        eta_seconds=remaining,
                        message="Analysing",
                    )

            recorder.flush(force=True)
            elapsed = time.monotonic() - wall_start

            # Finish the render before declaring the job complete. If this
            # fails there is no annotated video, and the job failed — §20: a
            # job must never report success over an output nobody can play.
            render = None
            if writer is not None:
                try:
                    render = writer.finalise(source=source_path)
                except VideoWriteError as exc:
                    writer.abort()
                    self._fail(
                        job_id, f"The annotated video could not be produced: {exc}"
                    )
                    return
                except Exception as exc:
                    writer.abort()
                    logger.exception("Job %s: render finalisation crashed", job_id[:8])
                    self._fail(job_id, f"Annotated video encoding failed: {exc}")
                    return

                if render.frames_written != processed:
                    # Frame-for-frame is the contract: a render short of the
                    # source has annotations drifting against the timeline.
                    logger.error(
                        "Job %s: wrote %d frames for %d source frames",
                        job_id[:8], render.frames_written, processed,
                    )
                    render.warnings.append(
                        f"Render has {render.frames_written} frames for "
                        f"{processed} source frames — timing may be off"
                    )
                for warning in render.warnings:
                    logger.warning("Job %s render: %s", job_id[:8], warning)

            summary = {
                "events_by_type": type_counts,
                "frames_analysed": processed,
                "detection_frames": processed // detect_every if detect_every else processed,
                "wall_seconds": round(elapsed, 2),
                "realtime_factor": (
                    round((processed / source_fps) / elapsed, 2) if elapsed > 0 else 0.0
                ),
                "processing_fps": round(processed / elapsed, 2) if elapsed > 0 else 0.0,
                "backend": detector.name,
                "ppe_method": ppe_detector.method if ppe_detector else "disabled",
                "hardhat_method": hardhat_method,
                "ppe_violations": ppe_violations,
                "zones_applied": len(zones),
                "profile": profile,
            }
            relative_annotated = _stored_path(render.path) if render is not None else None

            self._update(
                job_id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                processed_frames=processed,
                events_created=events_created,
                people_detected=len(people_seen),
                annotated_path=relative_annotated,
                output=render.as_dict() if render is not None else None,
                summary=summary,
                message=(
                    f"Analysed {processed} frames in {elapsed:.1f}s — "
                    f"{events_created} incident(s), {len(people_seen)} people tracked"
                ),
                finished_at=datetime.now(UTC),
            )
            self._publish(
                job_id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                processed_frames=processed,
                total_frames=total_frames,
                events_created=events_created,
                people_detected=len(people_seen),
                ppe_violations=ppe_violations,
                processing_fps=summary["processing_fps"],
                message="Complete",
                summary=summary,
                # The frontend reveals the player off this flag rather than
                # polling for a file to appear.
                annotated_ready=render is not None,
            )
            logger.info(
                "Video job complete: %s — %d frames, %d events, %d people, "
                "%.1fs (%.1fx realtime)",
                filename, processed, events_created, len(people_seen),
                elapsed, summary["realtime_factor"],
                extra={"job": job_id[:8]},
            )
        finally:
            capture.release()


_runner: VideoJobRunner | None = None


def get_job_runner() -> VideoJobRunner:
    """Process-wide job runner."""
    global _runner
    if _runner is None:
        _runner = VideoJobRunner()
    return _runner


def reconcile_interrupted_jobs() -> int:
    """Fail jobs left QUEUED/RUNNING by a previous process.

    Jobs run on an in-process thread pool, so a restart or redeploy (routine
    on a hosted platform) kills them mid-flight while the database still says
    they are running. Without this they would show as RUNNING forever. They
    are marked FAILED with a clear message so the operator can re-upload;
    nothing is re-run automatically because the upload may have been partial.
    Returns the number of jobs reconciled.
    """
    from sqlalchemy import select

    interrupted = 0
    with session_scope() as session:
        stale = session.execute(
            select(VideoJob).where(
                VideoJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING))
            )
        ).scalars().all()
        for job in stale:
            job.status = JobStatus.FAILED
            job.message = "Interrupted by a server restart — please upload again"
            job.finished_at = datetime.now(UTC)
            interrupted += 1
    if interrupted:
        logger.warning("Marked %d interrupted video job(s) as FAILED", interrupted)
    return interrupted
