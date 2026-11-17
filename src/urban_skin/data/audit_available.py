"""Audit all locally available Urban Skin datasets."""

from __future__ import annotations

import json
from pathlib import Path


def count(root: Path, suffixes: set[str]) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def main() -> None:
    data = Path("data")
    result = {
        "dacl10k": {"images": count(data / "dacl10k", {".jpg", ".png"}), "labeled": True},
        "walls": {
            "images": count(data / "Building defect on walls.v4i.yolov8", {".jpg", ".png"}),
            "labeled": True,
        },
        "facade": {
            "images": count(
                data / "building-facade-segmentation-original.v6i.coco-segmentation",
                {".jpg", ".png"},
            ),
            "labeled": True,
        },
        "corrosion": {
            "images": count(
                data / "Corrosion Condition State Semantic Segmentation Dataset",
                {".jpeg", ".png"},
            ),
            "labeled": True,
        },
        "rdd2022": {
            "images": count(data / "rdd2022-DatasetNinja", {".jpg", ".png"}),
            "labeled": False,
            "note": "local copy contains test images/annotations only",
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
