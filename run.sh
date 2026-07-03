#!/usr/bin/env bash
# Minimal alias for the release wrapper.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "${ROOT}/run_lmms_eval.sh" "$@"
