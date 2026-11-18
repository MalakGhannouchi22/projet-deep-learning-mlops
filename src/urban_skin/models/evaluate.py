"""Evaluate segmentation predictions and derive an interpretable health score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from urban_skin.data.segmentation_dataset import Dacl10kSegmentationDataset
from urban_skin.models.segmentation import build_segmentation_model


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_segmentation_model(
        checkpoint.get("architecture", "tiny_unet"), checkpoint["number_of_classes"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = Dacl10kSegmentationDataset(
        Path(args.manifest), "validation", checkpoint["image_size"], args.max_samples
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    intersection = torch.zeros(checkpoint["number_of_classes"], dtype=torch.float64)
    union = torch.zeros_like(intersection)
    with torch.no_grad():
        for images, masks in loader:
            predictions = model(images).argmax(dim=1)
            for class_id in range(1, checkpoint["number_of_classes"]):
                predicted = predictions == class_id
                actual = masks == class_id
                intersection[class_id] += (predicted & actual).sum()
                union[class_id] += (predicted | actual).sum()
    valid = union[1:] > 0
    mean_iou = float((intersection[1:][valid] / union[1:][valid]).mean()) if valid.any() else 0.0
    result = {
        "validation_samples": len(dataset),
        "mean_iou": round(mean_iou, 6),
        "segmentation_quality_percent": round(mean_iou * 100, 2),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/dacl10k_manifest.json")
    parser.add_argument("--checkpoint", default="models/dacl10k_tiny_unet.pt")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
