#!/usr/bin/env bash
# Build the Mojo acceleration kernels into a shared library.
#
# The Mojo toolchain is optional. If it is missing this script explains how to
# get it and exits 0 — the application runs on the numpy fallbacks and reports
# `mojo.active: false` in GET /api/diagnostics.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BUILD_DIR="$HERE/build"
SRC="$HERE/kernels.mojo"

case "$(uname -s)" in
  Darwin) LIB_NAME="libsv_kernels.dylib" ;;
  *)      LIB_NAME="libsv_kernels.so" ;;
esac
OUT="$BUILD_DIR/$LIB_NAME"

# Locate the mojo compiler: explicit override, project toolchain venv, PATH.
MOJO_BIN="${MOJO_BIN:-}"
if [[ -z "$MOJO_BIN" ]]; then
  for candidate in "$ROOT/.venv-mojo/bin/mojo" "$ROOT/.venv/bin/mojo"; do
    [[ -x "$candidate" ]] && MOJO_BIN="$candidate" && break
  done
fi
[[ -z "$MOJO_BIN" ]] && MOJO_BIN="$(command -v mojo 2>/dev/null || true)"

if [[ -z "$MOJO_BIN" ]]; then
  cat <<'MSG'
── Mojo toolchain not found ─────────────────────────────────────────────
SentinelVision AI will run normally using its numpy fallback kernels.

To enable the Mojo kernels:
    python3.11 -m venv .venv-mojo
    .venv-mojo/bin/pip install modular
    ./backend/inference/mojo/build.sh

Or set MOJO_BIN=/path/to/mojo.
─────────────────────────────────────────────────────────────────────────
MSG
  exit 0
fi

echo "Mojo compiler : $MOJO_BIN ($("$MOJO_BIN" --version 2>/dev/null | head -1))"
mkdir -p "$BUILD_DIR"

# Deprecation warnings are expected on Mojo 1.0 (see the header comment in
# kernels.mojo); surface errors only.
if "$MOJO_BIN" build --emit shared-lib "$SRC" -o "$OUT" 2>&1 \
     | grep -E "error|ERROR" ; then :; fi

if [[ -f "$OUT" ]]; then
  echo "Built         : ${OUT#$ROOT/} ($(wc -c < "$OUT" | tr -d ' ') bytes)"
  echo "Exported      : $(nm -gU "$OUT" 2>/dev/null | grep -c ' T _sv_') symbols"
  exit 0
fi

echo "ERROR: Mojo build failed — the application will use numpy fallbacks." >&2
exit 1
