"""Create normalized segmentation manifests for the available labeled datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _relative(path: Path) -> str:
    return path.as_posix()


def _wall_manifest(root: Path) -> dict[str, Any]:
    names = ["crack", "mold", "peeling_paint", "stairstep_crack", "water_seepage"]
    records: list[dict[str, Any]] = []
    image_dir = root / "train" / "images"
    label_dir = root / "train" / "labels"
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        annotations = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) != 5:
                    continue
                class_id, x, y, width, height = map(float, values)
                left, top = x - width / 2, y - height / 2
                right, bottom = x + width / 2, y + height / 2
                annotations.append(
                    {
                        "label": names[int(class_id)],
                        "points": [[[left, top], [right, top], [right, bottom], [left, bottom]]],
                    }
                )
        records.append(
            {"image": _relative(image_path), "split": "train", "annotations": annotations}
        )
    return {"dataset": root.name, "classes": names, "records": records}


def _facade_manifest(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    classes: dict[int, str] = {}
    for split in ("train", "valid", "test"):
        annotation_path = root / split / "_annotations.coco.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        for category in payload["categories"]:
            classes[int(category["id"])] = f"{category['name']}_{category['id']}"
        images = {int(image["id"]): image for image in payload["images"]}
        grouped: dict[int, list[dict[str, Any]]] = {}
        for annotation in payload["annotations"]:
            grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
        for image_id, image in images.items():
            annotations = []
            for annotation in grouped.get(image_id, []):
                segmentation = annotation.get("segmentation", [])
                for polygon in segmentation:
                    if len(polygon) < 6:
                        continue
                    points = [[polygon[i], polygon[i + 1]] for i in range(0, len(polygon), 2)]
                    annotations.append(
                        {"label": classes[int(annotation["category_id"])], "points": [points]}
                    )
            records.append(
                {
                    "image": _relative(root / split / image["file_name"]),
                    "split": "validation" if split == "valid" else split,
                    "annotations": annotations,
                }
            )
    labels = sorted(set(classes.values()))
    return {"dataset": root.name, "classes": labels, "records": records}


def _corrosion_manifest(root: Path) -> dict[str, Any]:
    base = (
        root
        / "Corrosion Condition State Classification"
        / "Corrosion Condition State Classification"
    )
    records: list[dict[str, Any]] = []
    labels: set[str] = set()
    for split_name, split in (("train", "train"), ("Test", "validation")):
        image_dir = base / "original" / split_name / "images"
        json_dir = base / "original" / split_name / "json"
        mask_dir = base / "original" / split_name / "masks"
        for image_path in sorted(image_dir.glob("*")):
            label_path = json_dir / f"{image_path.stem}.json"
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not label_path.is_file() or not mask_path.is_file():
                continue
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            annotation_labels = [shape["label"] for shape in payload.get("shapes", [])]
            labels.update(annotation_labels)
            records.append(
                {
                    "image": _relative(image_path),
                    "mask": _relative(mask_path),
                    "split": split,
                    "annotations": [{"label": label, "points": []} for label in annotation_labels],
                }
            )
    return {"dataset": root.name, "classes": sorted(labels), "records": records}


def prepare(dataset: str, data_root: Path, output_root: Path) -> dict[str, Any]:
    candidates = {
        "walls": data_root / "Building defect on walls.v4i.yolov8",
        "facade": data_root / "building-facade-segmentation-original.v6i.coco-segmentation",
        "corrosion": data_root / "Corrosion Condition State Semantic Segmentation Dataset",
    }
    if dataset not in candidates:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(candidates)}")
    root = candidates[dataset]
    manifest = {
        "walls": _wall_manifest,
        "facade": _facade_manifest,
        "corrosion": _corrosion_manifest,
    }[dataset](root)
    manifest["label_to_id"] = {label: index + 1 for index, label in enumerate(manifest["classes"])}
    output = output_root / f"{dataset}_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "dataset": dataset,
        "records": len(manifest["records"]),
        "classes": len(manifest["classes"]),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("walls", "facade", "corrosion"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.data_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
