# syntax=docker/dockerfile:1.7
# ── SentinelVision AI backend — production image ──────────────────────────
#
# FastAPI + Uvicorn + ONNX Runtime (CPU) + OpenCV + FFmpeg. Built for Render's
# Docker web service, but plain `docker build` works anywhere.
#
# Inference backend is pinned to ONNX Runtime: it is the dependable CPU path
# and the only one this image ships. MAX/Mojo and Ultralytics are not
# installed, and /api/diagnostics will report exactly that.
#
# Runtime data (uploads, processed renders, evidence, SQLite) lives under
# DATA_ROOT. On Render Free there is no persistent disk, so /var/data is
# ephemeral container storage: it is wiped on every deploy and restart. Mount
# a disk there on a paid plan and nothing else needs to change.

FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # glibc grows one malloc arena per thread; the job pool, ORT and uvicorn
    # together can hold tens of MB of fragmented free space. Two is plenty.
    MALLOC_ARENA_MAX=2 \
    # Pin the mmap threshold. By default glibc raises it every time a large
    # block is freed, so after the first few frames every multi-MB frame and
    # ORT scratch buffer comes off the brk heap, where freed space is never
    # returned and RSS ramps for the length of a job. At a fixed 128 KB those
    # buffers are mmapped and go straight back to the OS on free.
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TRIM_THRESHOLD_=131072 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies: ffmpeg/ffprobe for the finalising transcode + audio
# mux, libglib/libgomp for opencv-python-headless and onnxruntime, curl for
# the model download below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      libglib2.0-0 \
      libgomp1 \
      curl \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies first so the layer is cached across code changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code, label files and the two small detector models
# (yolov8n.onnx, ppe.onnx) and the hard-hat prompt bank, which are committed.
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY pyproject.toml ./

# The hard-hat validator (CLIP ViT-B/32 image encoder) is too large for the
# git repository, so it is fetched from its public source at build time. Same
# file `scripts/fetch_models.py --hardhat` downloads locally.
#   fp32  — 351 MB; reference calibration, fastest on CPU. Measured at ~585 MB
#           of RSS once ONNX Runtime has optimised it: on its own it exceeds
#           Render Free's 512 MB, so it is not the image default.
#   q4f16 — 53 MB, same measured accuracy, ~2x slower per crop; ~80 MB RSS.
#           Default. Whole stack (both YOLO detectors + validator + FastAPI +
#           OpenCV) measures ~270 MB with models loaded, ~400 MB peak during
#           a 720p analysis and ~465 MB at 1080p (CPU-only, arena off).
# Override with `--build-arg HARDHAT_VARIANT=fp32` on a host with the RAM.
ARG HARDHAT_VARIANT=q4f16
ARG HARDHAT_REPO=Xenova/clip-vit-base-patch32
RUN set -eu; \
    case "$HARDHAT_VARIANT" in \
      fp32)  f=onnx/vision_model.onnx ;; \
      fp16)  f=onnx/vision_model_fp16.onnx ;; \
      q4f16) f=onnx/vision_model_q4f16.onnx ;; \
      *) echo "unknown HARDHAT_VARIANT=$HARDHAT_VARIANT" >&2; exit 2 ;; \
    esac; \
    if [ ! -s models/hardhat_validator.onnx ]; then \
      curl -fsSL --retry 5 --retry-delay 3 \
        "https://huggingface.co/${HARDHAT_REPO}/resolve/main/${f}" \
        -o models/hardhat_validator.onnx; \
    fi; \
    test -s models/hardhat_prompts.npz; \
    python -c "import onnxruntime as ort; ort.InferenceSession('models/hardhat_validator.onnx', providers=['CPUExecutionProvider']); print('hard-hat validator OK')"

# Production defaults. Every one of these can be overridden by the platform's
# environment (Render: render.yaml / dashboard). No .env file is copied in.
ENV APP_ENV=production \
    APP_HOST=0.0.0.0 \
    PORT=8008 \
    LOG_JSON=true \
    INFERENCE_BACKEND=onnx \
    MOJO_ENABLED=off \
    AUTO_START_CAMERAS=false \
    INFERENCE_THREADS=1 \
    ONNX_CPU_MEM_ARENA=false \
    ANNOTATED_FFMPEG_THREADS=1 \
    MAX_CONCURRENT_JOBS=1 \
    ANNOTATED_MAX_WIDTH=1280 \
    # Fail a job cleanly (with a reason in its record) before the host
    # SIGKILLs the process at 512 MB with nothing in the log.
    JOB_MAX_RSS_MB=470 \
    JOB_MAX_SECONDS=7200 \
    DATA_ROOT=/var/data

# Ephemeral by default (no VOLUME): Render Free has no persistent disks.
RUN mkdir -p /var/data

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# One worker only: analysis jobs and camera pipelines are in-process threads.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
