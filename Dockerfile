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
# DATA_ROOT, which on Render is the persistent disk mounted at /var/data.

FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
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

# The hard-hat validator (CLIP ViT-B/32 image encoder, 351 MB fp32) is too
# large for the git repository, so it is fetched from its public source at
# build time. Same file `scripts/fetch_models.py --hardhat` downloads locally.
#   fp32  — reference calibration, fastest on CPU (default)
#   q4f16 — 53 MB, same measured accuracy, ~2x slower per crop
ARG HARDHAT_VARIANT=fp32
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
    DATA_ROOT=/var/data

RUN mkdir -p /var/data
VOLUME ["/var/data"]

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# One worker only: analysis jobs and camera pipelines are in-process threads.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
