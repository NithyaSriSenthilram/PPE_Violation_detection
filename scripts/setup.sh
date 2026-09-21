#!/usr/bin/env bash
# One-shot development setup for SentinelVision AI.
#
#   ./scripts/setup.sh              # backend + frontend + model
#   ./scripts/setup.sh --with-mojo  # also build the Mojo kernels
#
# Every optional component degrades cleanly: if the Mojo toolchain or a model
# export is unavailable the platform still runs and says so in
# GET /api/diagnostics.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_MOJO=0
[[ "${1:-}" == "--with-mojo" ]] && WITH_MOJO=1

info()  { printf '\033[38;5;39m▸\033[0m %s\n' "$*"; }
ok()    { printf '\033[38;5;40m✓\033[0m %s\n' "$*"; }
warn()  { printf '\033[38;5;214m!\033[0m %s\n' "$*"; }

# ── Python ────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version=$("$candidate" -c 'import sys; print(sys.version_info >= (3, 11))')
      [[ "$version" == "True" ]] && PYTHON="$candidate" && break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "ERROR: Python 3.11 or newer is required." >&2
  exit 1
fi
info "Python: $($PYTHON --version)"

if [[ ! -d .venv ]]; then
  info "Creating .venv"
  "$PYTHON" -m venv .venv
fi
./.venv/bin/python -m pip install --quiet --upgrade pip wheel
info "Installing backend dependencies"
./.venv/bin/python -m pip install --quiet -r requirements-dev.txt
ok "Backend dependencies installed"

# ── Configuration ─────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  cp .env.example .env
  ok "Created .env from .env.example"
else
  info ".env already exists — leaving it alone"
fi

mkdir -p data/uploads evidence/snapshots evidence/clips models

# ── Model ─────────────────────────────────────────────────────────────────
if [[ -f models/yolov8n.onnx ]]; then
  info "Detector already present: models/yolov8n.onnx"
else
  info "Exporting a detector (installs ultralytics once, for the export only)"
  ./.venv/bin/python -m pip install --quiet ultralytics || warn "ultralytics install failed"
  if ./.venv/bin/python scripts/fetch_models.py; then
    ok "Detector ready"
  else
    warn "Model export failed — the platform will start on synthetic detections."
    warn "See GET /api/diagnostics, then re-run: python scripts/fetch_models.py"
  fi
fi

# ── Mojo (optional) ───────────────────────────────────────────────────────
if [[ "$WITH_MOJO" == "1" ]]; then
  if [[ ! -x .venv-mojo/bin/mojo ]]; then
    info "Installing the Mojo toolchain into .venv-mojo (a few hundred MB)"
    "$PYTHON" -m venv .venv-mojo
    ./.venv-mojo/bin/python -m pip install --quiet --upgrade pip
    ./.venv-mojo/bin/python -m pip install --quiet modular || warn "modular install failed"
  fi
  ./backend/inference/mojo/build.sh || warn "Mojo build failed — numpy fallbacks will be used"
else
  info "Skipping Mojo (pass --with-mojo to build the accelerated kernels)"
fi

# ── Frontend ──────────────────────────────────────────────────────────────
if command -v npm >/dev/null 2>&1; then
  info "Installing frontend dependencies"
  (cd frontend && npm install --silent)
  ok "Frontend dependencies installed"
else
  warn "npm not found — skipping the frontend"
fi

cat <<'MSG'

────────────────────────────────────────────────────────────────────────────
Setup complete.

  Terminal 1   make backend     # API on http://127.0.0.1:8008
  Terminal 2   make frontend    # UI  on http://localhost:5173

  Verify        curl -s localhost:8008/api/diagnostics | python3 -m json.tool
  Tests         make test
────────────────────────────────────────────────────────────────────────────
MSG
