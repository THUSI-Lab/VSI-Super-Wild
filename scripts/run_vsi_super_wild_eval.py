#!/usr/bin/env python3
"""Unified VSI-Super-Wild lmms-eval runner."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
TASKS_DIR = PROJECT_ROOT / "tasks"
TASK_DATA_FILE = TASKS_DIR / "vsi_super_wild" / "data" / "vsi_super_wild_qa.jsonl"
TASK_NAME = "vsi_super_wild"
DEFAULT_VIDEO_ROOT = os.environ.get("VSI_SUPER_WILD_VIDEO_ROOT") or str(PROJECT_ROOT / "data")

MODEL_CONFIGS = {
    "qwen2_vl_2b": {"model": "qwen2_vl", "model_args": "pretrained=Qwen/Qwen2-VL-2B-Instruct"},
    "qwen2_vl_7b": {"model": "qwen2_vl", "model_args": "pretrained=Qwen/Qwen2-VL-7B-Instruct"},
    "qwen2_vl_72b": {"model": "qwen2_vl", "model_args": "pretrained=Qwen/Qwen2-VL-72B-Instruct"},
    "qwen_vl_2_5_3b": {"model": "qwen2_5_vl", "model_args": "pretrained=Qwen/Qwen2.5-VL-3B-Instruct"},
    "qwen_vl_2_5_7b": {"model": "qwen2_5_vl", "model_args": "pretrained=Qwen/Qwen2.5-VL-7B-Instruct"},
    "qwen_vl_2_5_72b": {"model": "qwen2_5_vl", "model_args": "pretrained=Qwen/Qwen2.5-VL-72B-Instruct"},
    "llava_onevision_7b": {"model": "llava_onevision", "model_args": "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov"},
    "internvl2_5_8b": {"model": "internvl2", "model_args": "pretrained=OpenGVLab/InternVL2_5-8B"},
    "cambrian_s_7b": {"model": "cambrian_s", "model_args": "pretrained=nyu-visionx/Cambrian-S-7B"},
}


def ensure_data_available() -> bool:
    if TASK_DATA_FILE.exists():
        print(f"[INFO] QA file found: {TASK_DATA_FILE}")
        return True
    print(f"[ERROR] QA file not found: {TASK_DATA_FILE}")
    print("[INFO] Place vsi_super_wild_qa.jsonl under tasks/vsi_super_wild/data/ before evaluation.")
    return False


def resolve_model(model: str, model_args: str | None) -> tuple[str, str]:
    if model in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model]
        return cfg["model"], cfg["model_args"]
    if not model_args:
        raise ValueError(
            f"Unknown model alias '{model}'. Provide --model_args, or use one of: {sorted(MODEL_CONFIGS)}"
        )
    return model, model_args


def run_eval(
    model: str,
    model_args: str | None = None,
    limit: int | None = None,
    batch_size: int = 1,
    device: str = "cuda",
    output_path: str | None = None,
    max_num_frames: int | None = None,
    fps: float | None = None,
    video_root: str = DEFAULT_VIDEO_ROOT,
    dry_run: bool = False,
) -> int:
    if not ensure_data_available():
        return 1

    try:
        lmms_model, lmms_model_args = resolve_model(model, model_args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    extra_args = []
    if max_num_frames is not None:
        extra_args.append(f"max_num_frames={max_num_frames}")
    if fps is not None:
        extra_args.append(f"fps={fps}")
    if extra_args:
        lmms_model_args = ",".join([lmms_model_args, *extra_args]) if lmms_model_args else ",".join(extra_args)

    cmd = [
        sys.executable,
        "-m",
        "lmms_eval",
        "eval",
        "--model",
        lmms_model,
        "--model_args",
        lmms_model_args,
        "--tasks",
        TASK_NAME,
        "--include_path",
        str(TASKS_DIR),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--log_samples",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if output_path:
        cmd.extend(["--output_path", output_path])

    print("\n" + "=" * 80)
    print("[INFO] Starting VSI-Super-Wild evaluation")
    print(f"       Model alias/input: {model}")
    print(f"       lmms-eval model:   {lmms_model}")
    print(f"       Task:              {TASK_NAME}")
    print(f"       Tasks dir:         {TASKS_DIR}")
    print("=" * 80 + "\n")
    print(f"[CMD] {' '.join(cmd)}\n")
    print(f"[ENV] VSI_SUPER_WILD_VIDEO_ROOT={video_root}\n")

    if dry_run:
        return 0

    env = os.environ.copy()
    env["VSI_SUPER_WILD_VIDEO_ROOT"] = str(video_root)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VSI-Super-Wild unified QA evaluation script for lmms-eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_vsi_super_wild_eval.py --model qwen_vl_2_5_7b --limit 10
  python scripts/run_vsi_super_wild_eval.py --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --dry_run
        """,
    )
    parser.add_argument("--model", type=str, required=True, help="Model alias or lmms-eval model name")
    parser.add_argument("--model_args", type=str, default=None, help="lmms-eval model args")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_path", type=str, default=None, help="Output path")
    parser.add_argument("--dry_run", action="store_true", help="Print command and environment without running evaluation")
    parser.add_argument("--video_root", type=str, default=DEFAULT_VIDEO_ROOT, help="Root directory containing videos")
    parser.add_argument("--max_num_frames", "--max_frames", dest="max_num_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    os.environ["VSI_SUPER_WILD_ROOT"] = str(REPO_ROOT)
    sys.exit(
        run_eval(
            args.model,
            args.model_args,
            args.limit,
            args.batch_size,
            args.device,
            args.output_path,
            args.max_num_frames,
            args.fps,
            args.video_root,
            args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
