#!/usr/bin/env python3
"""Recompute VSI-Super-Wild aggregate metrics from lmms-eval sample files."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _import_utils():
    root = Path(__file__).resolve().parents[1]
    utils_path = root / "tasks" / "vsi_super_wild" / "utils.py"
    spec = importlib.util.spec_from_file_location("vsi_super_wild_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import task utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="lmms-eval *_samples_*.jsonl file")
    args = parser.parse_args()

    task_utils = _import_utils()
    rows = _load_samples(args.samples)
    metric_items = []
    for row in rows:
        if "accuracy" in row or "mra" in row or "correct" in row:
            metric_items.append(row)
            continue
        doc = row.get("doc") or {}
        response = row.get("filtered_resps") or row.get("resps") or ""
        if isinstance(response, list):
            response = response[0] if response else ""
        metric_items.append(task_utils.process_results(doc, [str(response)]))

    print(json.dumps({
        "num_samples": len(rows),
        "accuracy": task_utils.aggregate_accuracy(metric_items),
        "mra": task_utils.aggregate_mra(metric_items),
    }, indent=2))


if __name__ == "__main__":
    main()
