"""Central configuration for SentinelVision AI.

All tunables live here and are sourced from environment variables / `.env`.
Nothing in the application should hard-code a threshold, path or URL — import
`settings` instead.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root — resolved from this file so the app works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

InferenceBackendName = Literal["auto", "max", "onnx", "ultralytics", "mock"]
MojoMode = Literal["auto", "on", "off"]


def _resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative configured path against the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class Settings(BaseSettings):
    """Typed application settings.

    Field names map to upper-case environment variables (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=os.getenv("SENTINEL_ENV_FILE", str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    app_host: str = "0.0.0.0"
    #: `PORT` is what hosting platforms (Render, Heroku, Fly, Cloud Run) inject;
    #: `APP_PORT` is the local override. Either name works, `PORT` wins.
    app_port: int = Field(8008, validation_alias=AliasChoices("PORT", "APP_PORT"))
    log_level: str = "INFO"
    log_json: bool = False
    app_name: str = "SentinelVision AI"
    app_version: str = "1.0.0"

    # ── Runtime storage ──────────────────────────────────────────────────
    #: One directory for everything the server writes at runtime: uploads,
    #: processed renders, evidence and the SQLite database. On a hosted
    #: deployment this is the persistent-disk mount (Render: `/var/data`), so
    #: that nothing is lost on a restart or redeploy. When set, it supplies
    #: the default for every path below that is not set explicitly; the
    #: individual variables still win when both are given.
    data_root: Path | None = None

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/surveillance.db"

    # ── Inference ────────────────────────────────────────────────────────
    inference_backend: InferenceBackendName = "auto"
    model_path: Path = Path("./models/yolov8n.onnx")
    model_labels_path: Path = Path("./models/coco.names")
    ppe_model_path: Path = Path("./models/ppe.onnx")
    ppe_model_labels_path: Path = Path("./models/ppe.names")

    confidence_threshold: float = Field(0.35, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    inference_width: int = Field(640, ge=64, le=2048)
    inference_height: int = Field(640, ge=64, le=2048)
    inference_threads: int = Field(4, ge=1, le=64)

    # ── PPE ──────────────────────────────────────────────────────────────
    #: PPE detection runs its own threshold: the PPE model is a different
    #: network trained on a different dataset, and tying it to the person
    #: detector's threshold would force a compromise on one of the two.
    #:
    #: This is the *presence* threshold — the bar a `helmet` or `vest` box has
    #: to clear. Measured on the reference footage: helmets score a median 0.56
    #: and vests 0.77, but the lower quartile of each falls to 0.32 and 0.43,
    #: and those are real items on real people — a saturated orange vest scores
    #: 0.37 where a yellow-green one scores 0.84, because the training sets are
    #: dominated by yellow-green. A 0.50 bar therefore drops roughly a third of
    #: genuine PPE and reports it as undetermined. 0.35 keeps it. Helmets carry
    #: no extra risk at this bar because every one of them is re-checked by the
    #: hard-hat validator below, which is what actually rejects caps.
    ppe_confidence_threshold: float = Field(0.35, ge=0.0, le=1.0)
    #: The bar for the *absence* classes (`no-helmet`, `no-vest`), which is a
    #: different question and needs a different number. Absence is the harder
    #: call and the model scores it accordingly: across the reference footage
    #: `no-helmet` never once exceeds 0.54 and its median is 0.34, so a 0.50
    #: bar discards 97% of the class and silently switches it off. What keeps
    #: the lower bar honest is not the score but PPE_MIN_CONSECUTIVE_FRAMES:
    #: a violation still has to hold for several consecutive frames before it
    #: fires, so a single weak reading raises nothing.
    ppe_absence_confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    #: Floor the PPE model itself decodes at. Deliberately below both bars
    #: above: the model has to emit a box before the assessment layer — the
    #: only layer that knows whether a class means presence or absence — can
    #: judge it against the right one. Raising this to the presence threshold
    #: would filter out the absence detections before they are ever seen.
    ppe_intake_confidence_threshold: float = Field(0.20, ge=0.0, le=1.0)
    ppe_allow_heuristic: bool = False
    ppe_every_n_detections: int = Field(1, ge=1, le=30)
    #: Items a person must be wearing. Anything in ITEM_SPECS is valid, so a
    #: site needing gloves and boots sets REQUIRED_PPE=helmet,vest,gloves,boots
    #: and supplies a model that emits those classes.
    required_ppe: str = "helmet,vest"
    #: Optional per-model class mapping, e.g. `hat=helmet:present,worker=person`.
    #: Empty means the built-in synonym table decides.
    ppe_class_map: str = ""
    #: Fraction of a PPE box that must fall inside a person box before the item
    #: is attributed to that person.
    ppe_association_threshold: float = Field(0.55, ge=0.0, le=1.0)
    #: Person boxes shorter than this carry too few pixels for a PPE reading.
    ppe_min_person_height: int = Field(96, ge=0, le=2048)
    #: Consecutive frames a missing-PPE finding must hold before it fires.
    ppe_min_consecutive_frames: int = Field(5, ge=1, le=120)

    # ── Hard-hat validation ──────────────────────────────────────────────
    #: Second-stage check on every helmet the PPE detector proposes. PPE models
    #: are trained to find *head coverings*, and they call a baseball cap a
    #: helmet — measured at 27 of 28 caps on the reference footage. Only an
    #: industrial hard hat is valid PPE, so a helmet detection is not a helmet
    #: until this stage agrees.
    hardhat_validation_enabled: bool = True
    hardhat_model_path: Path = Path("./models/hardhat_validator.onnx")
    #: Class prompt embeddings that go with the validator, written by
    #: `scripts/fetch_models.py --hardhat` alongside the model.
    hardhat_prompts_path: Path = Path("./models/hardhat_prompts.npz")
    #: Probability the head crop is an industrial hard hat, below which the
    #: helmet is rejected. 0.5 is measured, not assumed: on the reference
    #: crops it gives 0% cap-false-positive at 75% hard-hat recall, and
    #: raising it to 0.7 costs 44 points of recall for no precision gain.
    hardhat_confidence_threshold: float = Field(0.50, ge=0.0, le=1.0)
    #: Permit the documented structural fallback when no validator model is
    #: installed. Off by default: a heuristic that wrongly passes a cap reports
    #: false compliance, which is the failure this whole stage exists to stop.
    hardhat_allow_heuristic: bool = False
    #: Detection frames between re-validations of the same tracked person.
    #: Headwear does not change frame to frame, and the check costs ~55 ms per
    #: head, so the verdict is held (and averaged) in between.
    hardhat_validate_every_n_detections: int = Field(6, ge=1, le=120)
    #: Validations averaged per person before the verdict is read. Smooths a
    #: single bad look without hiding a real change of headwear.
    hardhat_smoothing_window: int = Field(3, ge=1, le=15)

    # ── Mojo ─────────────────────────────────────────────────────────────
    mojo_enabled: MojoMode = "auto"
    mojo_lib_path: Path = Path("./backend/inference/mojo/build/libsv_kernels.dylib")

    # ── Video pipeline ───────────────────────────────────────────────────
    target_fps: float = Field(12.0, ge=0.0, le=120.0)
    detect_every_n_frames: int = Field(2, ge=1, le=30)
    max_stream_width: int = Field(960, ge=160, le=3840)
    stream_open_timeout_seconds: float = Field(6.0, ge=1.0, le=60.0)
    reconnect_backoff_seconds: float = Field(3.0, ge=0.1)
    reconnect_max_backoff_seconds: float = Field(60.0, ge=1.0)
    frame_queue_size: int = Field(2, ge=1, le=64)
    #: Start every enabled camera in the database when the API boots. Off by
    #: default: the primary workflow is uploaded-video analysis, which needs no
    #: live camera, and a stored camera whose source is unreachable would
    #: otherwise reconnect in a loop from the moment the server starts.
    #: Cameras can still be started explicitly through the cameras API.
    auto_start_cameras: bool = False

    # ── Tracking ─────────────────────────────────────────────────────────
    track_high_threshold: float = Field(0.5, ge=0.0, le=1.0)
    track_low_threshold: float = Field(0.1, ge=0.0, le=1.0)
    track_match_iou: float = Field(0.8, ge=0.0, le=1.0)
    track_max_age_frames: int = Field(30, ge=1)
    track_min_hits: int = Field(3, ge=1)

    # ── Behaviour analysis ───────────────────────────────────────────────
    loitering_threshold: float = Field(60.0, ge=1.0)
    running_threshold: float = Field(2.2, ge=0.1)
    fall_aspect_ratio: float = Field(1.35, ge=0.5)
    fall_vertical_drop: float = Field(0.28, ge=0.01, le=1.0)
    fall_confirm_seconds: float = Field(1.0, ge=0.0)
    crowd_threshold: int = Field(15, ge=1)
    crowd_baseline_window: float = Field(120.0, ge=10.0)

    # ── Events / evidence ────────────────────────────────────────────────
    event_cooldown_seconds: float = Field(30.0, ge=0.0)
    event_max_per_minute: int = Field(12, ge=0, le=600)
    event_min_confidence: float = Field(0.4, ge=0.0, le=1.0)
    evidence_dir: Path = Path("./evidence")
    #: Full annotated renders of uploaded video. Deliberately separate from the
    #: evidence tree: evidence is per-incident and retention-sensitive, while a
    #: processed render is a derived copy of an upload that can be regenerated.
    processed_dir: Path = Path("./processed")
    clip_buffer_seconds: float = Field(8.0, ge=0.0, le=60.0)
    clip_post_seconds: float = Field(4.0, ge=0.0, le=60.0)
    clip_enabled: bool = True
    snapshot_jpeg_quality: int = Field(85, ge=40, le=100)

    # ── Uploads ──────────────────────────────────────────────────────────
    # ── Annotated video output ───────────────────────────────────────────
    annotated_video_enabled: bool = True
    #: Seconds an event banner stays on screen after the event fires. One frame
    #: at 25 fps is 40 ms — invisible. This is what makes an event *watchable*.
    event_overlay_seconds: float = Field(2.5, ge=0.0, le=30.0)
    #: Cap on the rendered width. Source resolution is preserved below this;
    #: above it the annotated frame is scaled down (annotations scale with it)
    #: so a 4K upload does not produce an unplayable multi-gigabyte render.
    annotated_max_width: int = Field(1920, ge=320, le=7680)
    #: Re-encode the finished render with ffmpeg for browser compatibility and
    #: a front-loaded index, and mux the original audio back in. Without it the
    #: file still plays, but seeking needs the whole file downloaded first.
    annotated_ffmpeg_finalise: bool = True
    #: CRF for the ffmpeg pass. Lower is better quality and a larger file.
    annotated_crf: int = Field(23, ge=0, le=51)

    # ── Overlay rendering ────────────────────────────────────────────────
    #: Draw the whole-person rectangle. Off by default: a body-sized box says
    #: only "a person is here", while the PPE boxes below say *what is wrong
    #: and where*. Person detection and tracking are unaffected — they still
    #: drive association, zones, loitering and every event. Turn this on to
    #: debug tracking, or on a deployment whose events are mostly zone- and
    #: movement-based, where the person marker is the point.
    show_person_boxes: bool = False
    #: Draw the tight PPE boxes — helmet on the head, vest on the torso.
    show_ppe_boxes: bool = True
    #: Add a small "ID #010" line under each PPE label. Off by default: the
    #: track number is internal bookkeeping, and repeating it beside every item
    #: crowds the labels that carry the finding.
    show_track_id: bool = False

    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = Field(512, ge=1, le=8192)
    allowed_video_extensions: str = ".mp4,.avi,.mov,.mkv,.webm"

    # ── Security ─────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    api_key: str = ""

    # ── Validators ───────────────────────────────────────────────────────
    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_level(cls, v: str) -> str:
        level = str(v).upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid LOG_LEVEL: {v}")
        return level

    @model_validator(mode="after")
    def _apply_data_root(self) -> Settings:
        """Derive the runtime paths from DATA_ROOT unless set individually."""
        if self.data_root is None:
            return self
        root = _resolve(self.data_root)
        explicit = self.model_fields_set
        if "upload_dir" not in explicit:
            self.upload_dir = root / "uploads"
        if "processed_dir" not in explicit:
            self.processed_dir = root / "processed"
        if "evidence_dir" not in explicit:
            self.evidence_dir = root / "evidence"
        if "database_url" not in explicit:
            self.database_url = f"sqlite:///{root / 'db' / 'surveillance.db'}"
        return self

    # ── Derived helpers ──────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extension_set(self) -> set[str]:
        return {
            e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
            for e in self.allowed_video_extensions.split(",")
            if e.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def required_ppe_items(self) -> list[str]:
        """REQUIRED_PPE as a validated list, unknown names dropped with a log."""
        from backend.inference.ppe_taxonomy import ITEM_SPECS

        items: list[str] = []
        for raw in self.required_ppe.split(","):
            name = raw.strip().lower()
            if not name:
                continue
            if name not in ITEM_SPECS:
                import logging

                logging.getLogger("sentinel.config").warning(
                    "REQUIRED_PPE lists unknown item %r; known items are %s",
                    name, ", ".join(ITEM_SPECS),
                )
                continue
            if name not in items:
                items.append(name)
        return items

    # Absolute paths ------------------------------------------------------
    @property
    def model_file(self) -> Path:
        return _resolve(self.model_path)

    @property
    def model_labels_file(self) -> Path:
        return _resolve(self.model_labels_path)

    @property
    def ppe_model_file(self) -> Path:
        return _resolve(self.ppe_model_path)

    @property
    def ppe_labels_file(self) -> Path:
        return _resolve(self.ppe_model_labels_path)

    @property
    def hardhat_model_file(self) -> Path:
        return _resolve(self.hardhat_model_path)

    @property
    def hardhat_prompts_file(self) -> Path:
        return _resolve(self.hardhat_prompts_path)

    @property
    def mojo_lib_file(self) -> Path:
        return _resolve(self.mojo_lib_path)

    @property
    def evidence_root(self) -> Path:
        return _resolve(self.evidence_dir)

    @property
    def snapshot_dir(self) -> Path:
        return self.evidence_root / "snapshots"

    @property
    def clip_dir(self) -> Path:
        return self.evidence_root / "clips"

    @property
    def processed_root(self) -> Path:
        return _resolve(self.processed_dir)

    @property
    def upload_root(self) -> Path:
        return _resolve(self.upload_dir)

    @property
    def resolved_database_url(self) -> str:
        """Rewrite a relative SQLite path to an absolute one.

        Without this, running uvicorn from a different cwd silently creates a
        second, empty database.
        """
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            raw = self.database_url[len(prefix) :]
            if not raw.startswith("/"):
                return f"{prefix}{_resolve(raw)}"
        return self.database_url

    @property
    def sqlite_file(self) -> Path | None:
        """Filesystem path of the SQLite database, or None for other engines."""
        url = self.resolved_database_url
        prefix = "sqlite:///"
        if not url.startswith(prefix) or ":memory:" in url:
            return None
        return Path(url[len(prefix) :].split("?", 1)[0])

    def ensure_directories(self) -> None:
        """Create every directory the runtime writes into."""
        dirs = [
            self.evidence_root,
            self.snapshot_dir,
            self.clip_dir,
            self.processed_root,
            self.upload_root,
            _resolve("./models"),
            _resolve("./data"),
        ]
        # The SQLite file may live on a mounted disk (DATA_ROOT) whose
        # subdirectory does not exist yet; SQLite will not create it.
        if (db := self.sqlite_file) is not None:
            dirs.append(db.parent)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


settings = get_settings()
