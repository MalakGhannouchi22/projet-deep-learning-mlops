"""Audit a FiftyOne-exported DACL10K dataset without loading images into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _records(samples_path: Path) -> list[dict[str, Any]]:
    with samples_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("Expected samples.json to contain a top-level 'samples' list")
    return payload["samples"]


def audit_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Return reproducible image, split, and annotation statistics."""
    samples = _records(dataset_dir / "samples.json")
    split_counts = Counter()
    label_counts = Counter()
    images_with_labels = 0
    annotation_count = 0

    for sample in samples:
        split_counts.update(sample.get("tags", []))
        detections = sample.get("od_ground_truth", {}).get("detections", [])
        polylines = sample.get("semseg_ground_truth_filled", {}).get("polylines", [])
        annotations = detections or polylines
        if annotations:
            images_with_labels += 1
        annotation_count += len(annotations)
        label_counts.update(
            item.get("label", "unknown") for item in annotations if isinstance(item, dict)
        )

    return {
        "dataset": dataset_dir.name,
        "samples": len(samples),
        "splits": dict(sorted(split_counts.items())),
        "images_with_labels": images_with_labels,
        "annotations": annotation_count,
        "classes": dict(sorted(label_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dacl10k"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_dataset(args.data_dir)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
