# VSI-Super-Wild task utils for lmms-eval: doc_to_text, doc_to_visual, process_results.
# Logic aligned with the VSI-Super-Wild evaluation logic (clean_question, parse_sequence, extract_letter, MRA).
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    import av
except ImportError:  # pragma: no cover - optional video backend
    av = None

try:
    from decord import VideoReader, cpu
except ImportError:  # pragma: no cover - optional fast path
    VideoReader = None
    cpu = None

FRAME_RECALL_CONTEXT_FRAMES = 12
FRAME_RECALL_LABELS = ("A", "B", "C", "D")


def _as_float_or_none(value: Any) -> float | None:
    """Best-effort cast to float."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_accuracy_value(item: Any) -> float | None:
    """Handle lmms-eval passing dicts, bools, or raw numeric metric values."""
    if isinstance(item, dict):
        if "accuracy" in item:
            return _as_float_or_none(item.get("accuracy"))
        if "correct" in item:
            return 1.0 if bool(item.get("correct")) else 0.0
        return None
    if isinstance(item, bool):
        return 1.0 if item else 0.0
    return _as_float_or_none(item)


def _extract_mra_value(item: Any) -> float | None:
    """Handle lmms-eval passing dicts or raw numeric metric values."""
    if isinstance(item, dict):
        if "mra" in item:
            return _as_float_or_none(item.get("mra"))
        return None
    return _as_float_or_none(item)


def _normalize_frame_indices(frame_indices: Any) -> Tuple[int, ...]:
    if not frame_indices:
        return ()
    normalized = []
    for value in frame_indices:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(normalized)


@lru_cache(maxsize=256)
def _probe_total_frames(video_path: str) -> int:
    if not video_path or not os.path.exists(video_path):
        return 0

    if VideoReader is not None:
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=int(os.getenv("LMMS_VIDEO_DECORD_THREADS", "2")))
            return len(vr)
        except Exception:
            pass

    if av is None:
        return 0

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)
        if total_frames > 0:
            return total_frames

        total_frames = 0
        for total_frames, _ in enumerate(container.decode(video=0), start=1):
            pass
        return total_frames
    finally:
        container.close()


@lru_cache(maxsize=4096)
def _load_frames(video_path: str, frame_indices: Tuple[int, ...]) -> Tuple[Image.Image, ...]:
    if not video_path or not os.path.exists(video_path) or not frame_indices:
        return ()

    total_frames = _probe_total_frames(video_path)
    if total_frames <= 0:
        return ()

    clamped = tuple(min(max(0, idx), total_frames - 1) for idx in frame_indices)

    if VideoReader is not None:
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=int(os.getenv("LMMS_VIDEO_DECORD_THREADS", "2")))
            frames = vr.get_batch(list(clamped)).asnumpy()
            return tuple(Image.fromarray(frame) for frame in frames)
        except Exception:
            pass

    unique_targets = sorted(set(clamped))
    frames_by_index: dict[int, Image.Image] = {}

    if av is None:
        return ()

    container = av.open(video_path)
    try:
        last_idx = unique_targets[-1]
        target_set = set(unique_targets)
        for current_idx, frame in enumerate(container.decode(video=0)):
            if current_idx > last_idx:
                break
            if current_idx in target_set and current_idx not in frames_by_index:
                frames_by_index[current_idx] = Image.fromarray(frame.to_ndarray(format="rgb24"))
        return tuple(frames_by_index[idx] for idx in clamped if idx in frames_by_index)
    finally:
        container.close()


def _clone_images(images: Tuple[Image.Image, ...]) -> List[Image.Image]:
    return [image.copy() for image in images]


def _format_frame_label_list(count: int) -> str:
    labels = [f"Frame {label}" for label in FRAME_RECALL_LABELS[:count]]
    if not labels:
        return "the next images"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return " and ".join(labels)
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _build_frame_recall_intro(video_path: str, frame_indices: Tuple[int, ...]) -> str:
    total_frames = _probe_total_frames(video_path)
    context_count = min(FRAME_RECALL_CONTEXT_FRAMES, total_frames) if total_frames > 0 else FRAME_RECALL_CONTEXT_FRAMES
    question_count = len(frame_indices)
    frame_labels = _format_frame_label_list(question_count)
    return (
        f"The first {context_count} images are uniformly sampled from the video; "
        f"the next {question_count} are {frame_labels} (in order).\n\n"
    )


def _build_frame_recall_visuals(video_path: str, frame_indices: Tuple[int, ...]) -> List[Any]:
    total_frames = _probe_total_frames(video_path)
    if total_frames <= 0:
        return []

    context_count = min(FRAME_RECALL_CONTEXT_FRAMES, total_frames)
    context_indices = tuple(np.linspace(0, total_frames - 1, context_count, dtype=int).tolist())
    visuals = _clone_images(_load_frames(video_path, context_indices))
    visuals.extend(_clone_images(_load_frames(video_path, frame_indices)))
    return visuals


def clean_question_for_prompt(question: str) -> str:
    """Clean question text for model prompt."""
    q = re.sub(
        r"\.\s*At each checkpoint,\s*report the cumulative count up to that point as a single integer\.",
        ". Report the total count up to this point as a single integer.",
        question,
    )
    q = re.sub(
        r"\.\s*At each checkpoint,\s*arrange all seen objects",
        ". Arrange all seen objects",
        q,
    )
    q = q.replace("From the video you have watched so far, here are", "Here are")
    return q


def parse_sequence(text: str) -> str:
    """Parse sequence answer like 'ABCD' from text."""
    text = text.upper().replace("\u2192", "").replace("->", "").replace(" ", "").replace(",", "")
    match = re.search(r"[ABCD]{2,4}", text)
    return match.group(0) if match else ""


def extract_letter(text: str) -> str:
    """
    Extract valid letter (A-D) from text.
    Prioritizes explicit answer patterns like "answer is B" or "(B)".
    """
    text = text.upper().strip()

    patterns = [
        r"ANSWER\s*(?:IS|:)?\s*[\"']?([A-D])[\"']?",
        r"\bOPTION\s*([A-D])\b",
        r"\bCHOICE\s*([A-D])\b",
        r"\(([A-D])\)",
        r"^[\s\[\(\{]*([A-D])[\s\]\)\}\. ]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    match = re.search(r"\b([A-D])[\.\)]", text)
    if match:
        return match.group(1)

    for letter in ["A", "B", "C", "D"]:
        if letter in text:
            return letter

    return "A"



def _task_family(doc: Dict[str, Any]) -> str:
    """Return the release task family while staying compatible with legacy names."""
    task_type = (doc.get("task_type") or "").upper()
    if task_type in {"VPO", "VMR", "VOO", "VOC"}:
        return task_type
    legacy = task_type.lower()
    if "frame_recall" in legacy:
        return "VPO"
    if "motion" in legacy:
        return "VMR"
    if "appearance" in legacy:
        return "VOO"
    if "count" in legacy:
        return "VOC"
    return task_type or "UNKNOWN"


@lru_cache(maxsize=4096)
def _resolve_video_path_from_name(video_name: str, video_root: str) -> str:
    name = str(video_name or "").strip()
    if not name:
        return ""
    if os.path.isabs(name):
        return name
    candidates = []
    base = name if name.endswith(".mp4") else f"{name}.mp4"
    roots = [video_root or os.getcwd()]
    subdirs = ("", "videos", "long_video_persp", "new_long_video_persp", "top20merge_0207_persp")
    for root in roots:
        for subdir in subdirs:
            candidates.append(os.path.join(root, subdir, base) if subdir else os.path.join(root, base))
    for path in candidates:
        if os.path.exists(path):
            return path

    # Release QA uses clean names like Z7ta3z5qcMA_back.mp4, while some
    # mirrored video folders keep semantic prefixes such as
    # IndustrialFacilitiesdatacenter_Z7ta3z5qcMA_back.mp4.
    suffix_matches = []
    for root in roots:
        for subdir in subdirs:
            search_dir = os.path.join(root, subdir) if subdir else root
            if not os.path.isdir(search_dir):
                continue
            try:
                for filename in os.listdir(search_dir):
                    if filename.endswith("_" + base):
                        suffix_matches.append(os.path.join(search_dir, filename))
            except OSError:
                continue
    if suffix_matches:
        return sorted(suffix_matches)[0]

    return candidates[0]


def resolve_doc_video_path(doc: Dict[str, Any]) -> str:
    """Resolve video path from legacy video_path or release video_name + VSI_SUPER_WILD_VIDEO_ROOT/current working directory."""
    legacy_path = doc.get("video_path")
    if legacy_path:
        return str(legacy_path)
    video_root = os.getenv("VSI_SUPER_WILD_VIDEO_ROOT") or os.getcwd()
    return _resolve_video_path_from_name(str(doc.get("video_name") or ""), video_root)


# =============================================================================
# lmms-eval Standard Interface Functions
# =============================================================================

def doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Convert document to text prompt for the model.
    Standard lmms-eval interface function.
    """
    question = clean_question_for_prompt(doc.get("question", ""))
    family = _task_family(doc)
    frame_indices = _normalize_frame_indices(doc.get("frame_indices"))

    if doc.get("options"):
        options_str = "\n".join(doc["options"])
        prefix = ""
        if family == "VPO" and frame_indices:
            prefix = _build_frame_recall_intro(resolve_doc_video_path(doc), frame_indices)
        return f"{prefix}{question}\n\nOptions:\n{options_str}\n\nAnswer with only the letter (A, B, C, or D)."

    if family == "VOC":
        return f"{question}\n\nAnswer with only a single integer number."

    if family == "VOO":
        return f"{question}\n\nAnswer with only the letter sequence (e.g., ABC), nothing else."

    return question


def doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    """
    Extract visual content from document.
    - VPO/place-ordering tasks: 12 sampled context frames + checkpoint frames as images
    - other tasks: full video path resolved from video_name at runtime
    """
    video_path = resolve_doc_video_path(doc)
    family = _task_family(doc)
    frame_indices = _normalize_frame_indices(doc.get("frame_indices"))

    if family == "VPO" and frame_indices:
        visuals = _build_frame_recall_visuals(video_path, frame_indices)
        if visuals:
            return visuals

    return [video_path] if video_path else []


def doc_to_target(doc: Dict[str, Any]) -> Any:
    """
    Extract ground truth answer from document.
    Standard lmms-eval interface function.
    """
    return doc.get("answer")


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process model output and compute metrics.
    Standard lmms-eval interface function.

    Returns dict with keys for aggregation:
    - accuracy: binary correct/incorrect
    - mra: mean relative accuracy for counting tasks
    """
    text = (results[0] or "").strip() if results else ""
    family = _task_family(doc)
    gt = doc.get("answer")

    out = {
        "doc_id": doc.get("doc_id"),
        "gt": gt,
        "pred_raw": text,
        "task_type": doc.get("task_type") or "unknown",
    }

    if family == "VOO" and not doc.get("options"):
        pred = parse_sequence(text)
        out["pred"] = pred
        out["correct"] = pred == gt
        out["accuracy"] = 1.0 if pred == gt else 0.0
        return out

    if doc.get("options") is not None or family in {"VPO", "VMR"}:
        pred = extract_letter(text)
        out["pred"] = pred
        out["correct"] = pred == gt
        out["accuracy"] = 1.0 if pred == gt else 0.0
        return out

    if family == "VOC":
        nums = re.findall(r"\d+", text)
        pred_val = int(nums[0]) if nums else 0
        gt_val = gt if isinstance(gt, int) else int(gt)
        mra = max(0.0, 1.0 - abs(pred_val - gt_val) / max(gt_val, 1))
        out["pred"] = pred_val
        out["mra"] = mra
        out["accuracy"] = 1.0 if abs(pred_val - gt_val) / max(gt_val, 1) <= 0.1 else 0.0
        return out

    return out

# =============================================================================
# lmms-eval Aggregation Functions
# =============================================================================

def aggregate_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate accuracy metric across all samples.
    Standard lmms-eval aggregation function.
    Returns: accuracy percentage (0-100)
    """
    if not results:
        return 0.0

    vals = []
    for item in results:
        value = _extract_accuracy_value(item)
        if value is not None:
            vals.append(value)
    if not vals:
        return 0.0
    return sum(vals) / len(vals) * 100.0


def aggregate_mra(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate MRA (Mean Relative Accuracy) metric.
    Standard lmms-eval aggregation function.
    Returns: MRA percentage (0-100)
    """
    if not results:
        return 0.0

    vals = []
    for item in results:
        value = _extract_mra_value(item)
        if value is not None:
            vals.append(value)
    if not vals:
        return 0.0
    return sum(vals) / len(vals) * 100.0


# =============================================================================
# Legacy/Additional Aggregation Functions
# =============================================================================

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Legacy aggregation function for detailed per-task-type metrics.
    Returns detailed breakdown by task type.
    """
    by_type = {}
    for item in results:
        task_type = item.get("task_type") or "unknown"
        by_type.setdefault(task_type, []).append(item)

    metrics = {}
    for task_type, items in by_type.items():
        if not items:
            continue
        if "accuracy" in items[0] or "correct" in items[0]:
            acc = sum(x.get("accuracy", float(x.get("correct", False))) for x in items) / len(items) * 100.0
            metrics[f"{task_type}_accuracy"] = acc
        if "mra" in items[0]:
            mra = sum(x.get("mra", 0) for x in items) / len(items) * 100.0
            metrics[f"{task_type}_mra"] = mra

    metrics["overall_accuracy"] = aggregate_accuracy(results)
    metrics["overall_mra"] = aggregate_mra(results)
    return metrics
