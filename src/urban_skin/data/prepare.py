"""Prepare DACL10K metadata for reproducible segmentation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_samples(samples_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Expected samples.json to contain a top-level 'samples' list")
    return samples


def prepare_manifest(dataset_dir: Path, output_path: Path) -> dict[str, Any]:
    samples = load_samples(dataset_dir / "samples.json")
    labels = sorted(
        {
            polygon["label"]
            for sample in samples
            for polygon in sample.get("semseg_ground_truth_filled", {}).get("polylines", [])
            if polygon.get("label")
        }
    )
    label_to_id = {label: index + 1 for index, label in enumerate(labels)}
    records = []
    for sample in samples:
        annotations = sample.get("semseg_ground_truth_filled", {}).get("polylines", [])
        records.append(
            {
                "image": sample["filepath"],
                "split": (sample.get("tags") or ["unassigned"])[0],
                "width": sample.get("metadata", {}).get("width"),
                "height": sample.get("metadata", {}).get("height"),
                "annotations": [
                    {"label": polygon["label"], "points": polygon["points"]}
                    for polygon in annotations
                    if polygon.get("label") in label_to_id
                ],
            }
        )
    manifest = {
        "dataset": dataset_dir.name,
        "classes": labels,
        "label_to_id": label_to_id,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dacl10k"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/dacl10k_manifest.json")
    )
    args = parser.parse_args()
    manifest = prepare_manifest(args.data_dir, args.output)
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "classes": len(manifest["classes"]),
                "records": len(manifest["records"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
