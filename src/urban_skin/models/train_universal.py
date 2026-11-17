"""Train Tiny U-Net on one normalized dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from urban_skin.data.universal_segmentation_dataset import UniversalSegmentationDataset
from urban_skin.models.segmentation import TinyUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_data = UniversalSegmentationDataset(
        args.manifest, "train", args.image_size, args.max_samples
    )
    validation_data = UniversalSegmentationDataset(
        args.manifest, "validation", args.image_size, args.max_samples
    )
    if not train_data:
        raise ValueError(f"No training records found in {args.manifest}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TinyUNet(len(manifest["classes"]) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, masks in DataLoader(train_data, args.batch_size, shuffle=True, num_workers=0):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        epoch_loss = total_loss / len(train_data)
        history.append({"epoch": epoch + 1, "train_loss": round(epoch_loss, 6)})
        print(json.dumps(history[-1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "number_of_classes": len(manifest["classes"]) + 1,
            "image_size": args.image_size,
            "dataset": args.dataset,
            "history": history,
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "device": str(device),
                "train_samples": len(train_data),
                "validation_samples": len(validation_data),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
