"""Evaluate a normalized segmentation checkpoint with IoU and Dice metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from urban_skin.data.universal_segmentation_dataset import UniversalSegmentationDataset
from urban_skin.models.segmentation import build_segmentation_model


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = UniversalSegmentationDataset(
        Path(args.manifest), "validation", checkpoint["image_size"], args.max_samples
    )
    if not dataset:
        raise ValueError("The manifest has no validation records to evaluate.")
    model = build_segmentation_model(
        checkpoint.get("architecture", "tiny_unet"), checkpoint["number_of_classes"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    intersections = torch.zeros(checkpoint["number_of_classes"], dtype=torch.float64)
    predicted_areas = torch.zeros_like(intersections)
    actual_areas = torch.zeros_like(intersections)
    unions = torch.zeros_like(intersections)
    with torch.no_grad():
        for images, masks in DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        ):
            predictions = model(images).argmax(dim=1)
            for class_id in range(1, checkpoint["number_of_classes"]):
                predicted = predictions == class_id
                actual = masks == class_id
                intersections[class_id] += (predicted & actual).sum()
                predicted_areas[class_id] += predicted.sum()
                actual_areas[class_id] += actual.sum()
                unions[class_id] += (predicted | actual).sum()
    valid = unions[1:] > 0
    iou = intersections[1:][valid] / unions[1:][valid]
    dice_denominator = predicted_areas[1:] + actual_areas[1:]
    dice_valid = dice_denominator > 0
    dice = (2 * intersections[1:][dice_valid]) / dice_denominator[dice_valid]
    result = {
        "validation_samples": len(dataset),
        "mean_iou": round(float(iou.mean()) if valid.any() else 0.0, 6),
        "mean_dice": round(float(dice.mean()) if dice_valid.any() else 0.0, 6),
        "segmentation_quality_percent": round(
            float(iou.mean() * 100) if valid.any() else 0.0, 2
        ),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
