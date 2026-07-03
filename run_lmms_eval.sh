#!/usr/bin/env bash
# Release-friendly VSI-Super-Wild lmms-eval wrapper.
# Usage: bash run_lmms_eval.sh [model] [limit]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
MODEL="${1:-qwen2_5_vl}"
LIMIT="${2:-10}"
export VSI_SUPER_WILD_VIDEO_ROOT="${VSI_SUPER_WILD_VIDEO_ROOT:-${ROOT}/data}"
MODEL_ARGS="${MODEL_ARGS:-pretrained=Qwen/Qwen2.5-VL-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
cmd=(python -m lmms_eval eval
  --model "$MODEL"
  --model_args "$MODEL_ARGS"
  --tasks vsi_super_wild
  --include_path "${ROOT}/tasks"
  --batch_size "$BATCH_SIZE"
  --device "$DEVICE"
  --log_samples
  --limit "$LIMIT")
printf '[ENV] VSI_SUPER_WILD_VIDEO_ROOT=%s\n' "$VSI_SUPER_WILD_VIDEO_ROOT"
printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
"${cmd[@]}"
