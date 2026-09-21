"""Writing the full annotated render of an analysed video.

This is the primary output of uploaded-video analysis: the whole source clip,
every frame, with detections and events drawn on it — not a set of stills. It
is deliberately separate from evidence capture, which samples single incidents.

Three things make the difference between a file that exists and a file an
operator can actually watch:

**Codec.** OpenCV will happily open a writer for a fourcc the platform cannot
encode and then produce an empty or undecodable file, so each candidate is
tried in turn and the result is verified by reopening it afterwards.

**Front-loaded index.** OpenCV writes the MP4 index (`moov`) at the *end* of
the file. A browser cannot seek — often cannot start playing — until it has
that, so it downloads the entire file first. `-movflags +faststart` moves it to
the front, which is what makes scrubbing through a long render feel instant.

**Audio.** `cv2.VideoWriter` has no concept of audio, so the source track is
muxed back in afterwards. This is a convenience, never a requirement: if it
cannot be done the render still succeeds and says so.

All three finishing steps need ffmpeg. Without it the raw OpenCV file is still
returned and still plays; the warnings say what was given up.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.config import settings
from backend.logging_conf import get_logger

logger = get_logger(__name__)

#: Tried in order. `avc1` is H.264 — the one browsers universally play.
#: `mp4v` (MPEG-4 Part 2) is the usual fallback and is *not* reliably playable
#: in Chrome or Safari, which is exactly why the ffmpeg pass below exists.
CODEC_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("avc1", "H.264"),
    ("mp4v", "MPEG-4 Part 2"),
)

#: Codecs every current browser decodes. Anything else gets transcoded.
BROWSER_SAFE_CODECS = frozenset({"h264", "avc1", "vp8", "vp9", "av1"})

FFMPEG_TIMEOUT = 900.0


class VideoWriteError(RuntimeError):
    """The annotated render could not be produced."""


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def probe_media(path: Path) -> dict[str, Any]:
    """Codec, geometry, duration and audio presence, via ffprobe.

    Returns ``{}`` when ffprobe is unavailable — callers must treat an empty
    result as *unknown*, never as *absent*.
    """
    binary = ffprobe_path()
    if binary is None or not path.is_file():
        return {}
    try:
        completed = subprocess.run(  # noqa: S603
            [
                binary, "-v", "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,nb_frames,avg_frame_rate",
                "-show_entries", "format=duration,size",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if completed.returncode != 0:
            return {}
        import json

        data = json.loads(completed.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug("ffprobe failed on %s: %s", path.name, exc)
        return {}

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    def _float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    rate = str(video.get("avg_frame_rate") or "0/1")
    try:
        numerator, _, denominator = rate.partition("/")
        fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return {
        "codec": video.get("codec_name", ""),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "frames": int(video.get("nb_frames") or 0),
        "fps": round(fps, 3),
        "duration_seconds": round(_float(fmt.get("duration")), 3),
        "size_bytes": int(_float(fmt.get("size"))),
        "has_audio": audio is not None,
        "audio_codec": (audio or {}).get("codec_name", ""),
    }


@dataclass
class RenderResult:
    """What was actually produced, for the job record and the API."""

    path: Path
    frames_written: int
    width: int
    height: int
    fps: float
    codec: str = ""
    size_bytes: int = 0
    duration_seconds: float = 0.0
    has_audio: bool = False
    browser_compatible: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_written": self.frames_written,
            "resolution": f"{self.width}x{self.height}",
            "fps": round(self.fps, 3),
            "codec": self.codec,
            "size_bytes": self.size_bytes,
            "duration_seconds": self.duration_seconds,
            "has_audio": self.has_audio,
            "browser_compatible": self.browser_compatible,
            "warnings": list(self.warnings),
        }


class AnnotatedVideoWriter:
    """Streams annotated frames to disk, one at a time.

    Frames are written and released as they come; nothing accumulates in
    memory, so render cost is flat regardless of how long the source is.
    """

    def __init__(
        self,
        destination: Path,
        fps: float,
        source_size: tuple[int, int],
        max_width: int | None = None,
    ) -> None:
        self.destination = destination
        self.fps = fps if fps > 0 else 25.0
        source_width, source_height = source_size

        limit = settings.annotated_max_width if max_width is None else max_width
        if source_width > limit:
            # Annotations are drawn at source scale and scaled with the frame,
            # so they stay correctly positioned — no coordinate maths needed.
            scale = limit / source_width
            # Even dimensions: H.264 chroma subsampling requires them.
            self.width = limit - (limit % 2)
            self.height = max(2, int(round(source_height * scale)) & ~1)
            self.downscaled = True
        else:
            self.width = source_width - (source_width % 2)
            self.height = source_height - (source_height % 2)
            self.downscaled = False

        self._resize_needed = (self.width, self.height) != (source_width, source_height)
        self._writer: cv2.VideoWriter | None = None
        self._codec = ""
        self._frames = 0
        self.warnings: list[str] = []
        if self.downscaled:
            self.warnings.append(
                f"Rendered at {self.width}x{self.height} rather than the source "
                f"{source_width}x{source_height} (ANNOTATED_MAX_WIDTH={limit})"
            )

    # ── lifecycle ────────────────────────────────────────────────────────
    def open(self) -> None:
        """Open the encoder, trying each codec until one really works."""
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        attempts: list[str] = []
        for fourcc, label in CODEC_CANDIDATES:
            candidate = cv2.VideoWriter(
                str(self.destination),
                cv2.VideoWriter_fourcc(*fourcc),
                self.fps,
                (self.width, self.height),
            )
            if candidate.isOpened():
                self._writer = candidate
                self._codec = fourcc
                logger.info(
                    "Annotated render: %s @ %dx%d %.2ffps via %s",
                    self.destination.name, self.width, self.height, self.fps, label,
                )
                return
            candidate.release()
            attempts.append(f"{fourcc} ({label})")

        raise VideoWriteError(
            "No usable video codec on this system; tried "
            f"{', '.join(attempts)}. Install ffmpeg or an OpenCV build with "
            "H.264 support."
        )

    def write(self, frame: np.ndarray) -> None:
        """Write one annotated frame. Never buffers."""
        if self._writer is None:
            raise VideoWriteError("writer is not open")
        if self._resize_needed:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        self._writer.write(frame)
        self._frames += 1

    @property
    def frames_written(self) -> int:
        return self._frames

    def abort(self) -> None:
        """Release the encoder and remove a partial file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self.destination.unlink(missing_ok=True)

    # ── finishing ────────────────────────────────────────────────────────
    def finalise(self, source: Path | None = None) -> RenderResult:
        """Close the encoder, verify the file, then make it browser-friendly.

        Raises :class:`VideoWriteError` if no playable file was produced — a
        job must never report success over a render nobody can open.
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self._frames == 0:
            self.destination.unlink(missing_ok=True)
            raise VideoWriteError("no frames were written to the annotated video")
        if not self.destination.is_file() or self.destination.stat().st_size == 0:
            raise VideoWriteError(
                "the encoder produced no output file — the codec accepted "
                "frames but wrote nothing"
            )

        # Verify by reopening: an encoder can accept every frame and still
        # leave a file no decoder will touch.
        check = cv2.VideoCapture(str(self.destination))
        readable = check.isOpened() and check.read()[0]
        check.release()
        if not readable:
            raise VideoWriteError(
                "the annotated video was written but cannot be decoded"
            )

        result = RenderResult(
            path=self.destination,
            frames_written=self._frames,
            width=self.width,
            height=self.height,
            fps=self.fps,
            codec=self._codec,
            size_bytes=self.destination.stat().st_size,
            duration_seconds=round(self._frames / self.fps, 3),
            warnings=list(self.warnings),
        )

        probe = probe_media(self.destination)
        if probe:
            result.codec = probe.get("codec") or result.codec
            result.duration_seconds = probe.get("duration_seconds") or result.duration_seconds

        if settings.annotated_ffmpeg_finalise:
            self._finalise_with_ffmpeg(result, source)
        else:
            result.warnings.append(
                "ANNOTATED_FFMPEG_FINALISE=false — the file has no front-loaded "
                "index, so the browser must download it fully before seeking"
            )

        result.browser_compatible = (result.codec or "").lower() in BROWSER_SAFE_CODECS
        if not result.browser_compatible:
            result.warnings.append(
                f"Codec '{result.codec}' may not play in Chrome or Safari; "
                f"install ffmpeg so the render can be transcoded to H.264"
            )
        return result

    def _finalise_with_ffmpeg(self, result: RenderResult, source: Path | None) -> None:
        """Transcode/remux for browser playback, seeking and audio.

        Every failure here is recorded as a warning and swallowed: the
        unprocessed render is still a working video, and losing an entire
        analysis because audio could not be copied would be absurd.
        """
        binary = ffmpeg_path()
        if binary is None:
            result.warnings.append(
                "ffmpeg is not installed — audio preservation unavailable, and "
                "the file has no front-loaded index so seeking requires a full "
                "download. The annotated video itself is unaffected."
            )
            return

        source_probe = probe_media(source) if source else {}
        want_audio = bool(source_probe.get("has_audio"))
        # Copy the video stream when it is already H.264; only re-encode when
        # the codec would not play. Copying a 200 MB render is seconds' work
        # against minutes for a re-encode, at identical quality.
        already_h264 = (result.codec or "").lower() in {"h264", "avc1"}

        output = self.destination.with_name(f"{self.destination.stem}.final.mp4")
        command = [binary, "-y", "-loglevel", "error", "-i", str(self.destination)]
        if want_audio and source is not None:
            command += ["-i", str(source)]

        command += ["-map", "0:v:0"]
        if want_audio and source is not None:
            # `?` makes the audio mapping optional: a source whose audio stream
            # ffprobe saw but ffmpeg cannot read must not abort the whole mux.
            command += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k"]

        if already_h264:
            command += ["-c:v", "copy"]
        else:
            command += [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", str(settings.annotated_crf),
                # 4:2:0 is what browsers and hardware decoders expect; ffmpeg
                # defaults to 4:4:4 from some inputs, which Safari will not play.
                "-pix_fmt", "yuv420p",
            ]
            if settings.annotated_ffmpeg_threads > 0:
                command += ["-threads", str(settings.annotated_ffmpeg_threads)]
        command += ["-movflags", "+faststart", "-shortest", str(output)]

        try:
            completed = subprocess.run(  # noqa: S603
                command, capture_output=True, text=True,
                timeout=FFMPEG_TIMEOUT, check=False,
            )
        except subprocess.TimeoutExpired:
            output.unlink(missing_ok=True)
            result.warnings.append(
                f"ffmpeg timed out after {FFMPEG_TIMEOUT:.0f}s; the render is "
                f"unmodified and still playable"
            )
            return
        except OSError as exc:
            output.unlink(missing_ok=True)
            result.warnings.append(f"ffmpeg could not be run: {exc}")
            return

        if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            detail = (completed.stderr or "").strip().splitlines()
            result.warnings.append(
                "ffmpeg finalisation failed, keeping the unprocessed render: "
                + (detail[-1] if detail else f"exit code {completed.returncode}")
            )
            if want_audio:
                result.warnings.append("Audio preservation unavailable")
            return

        output.replace(self.destination)
        final = probe_media(self.destination)
        if final:
            result.codec = final.get("codec") or result.codec
            result.size_bytes = final.get("size_bytes") or self.destination.stat().st_size
            result.duration_seconds = final.get("duration_seconds") or result.duration_seconds
            result.has_audio = bool(final.get("has_audio"))
        else:
            result.size_bytes = self.destination.stat().st_size

        if want_audio and not result.has_audio:
            result.warnings.append("Audio preservation unavailable")
        elif not want_audio:
            result.warnings.append("Source has no audio track")
