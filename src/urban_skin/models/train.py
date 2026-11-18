"""Train a reproducible DACL10K segmentation baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from urban_skin.data.segmentation_dataset import Dacl10kSegmentationDataset
from urban_skin.models.losses import pixel_class_weights
from urban_skin.models.segmentation import TinyUNet


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = Dacl10kSegmentationDataset(
        Path(args.manifest), "train", args.image_size, args.max_samples
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = TinyUNet(args.number_of_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss(
        weight=pixel_class_weights(dataset, args.number_of_classes).to(device)
    )
    history = []
    mlflow_run = None
    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        mlflow_run = mlflow.start_run(run_name=args.run_name)
        mlflow.log_params(
            {
                "architecture": "TinyUNet",
                "dataset": "dacl10k",
                "samples": len(dataset),
                "epochs": args.epochs,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "device": str(device),
            }
        )
    try:
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for images, masks in loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                loss = loss_function(model(images), masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
            epoch_loss = total_loss / max(len(dataset), 1)
            history.append({"epoch": epoch + 1, "train_loss": round(epoch_loss, 6)})
            if mlflow_run:
                mlflow.log_metric("train_loss", epoch_loss, step=epoch + 1)
            print(json.dumps(history[-1]))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "number_of_classes": args.number_of_classes,
                "image_size": args.image_size,
                "history": history,
            },
            output,
        )
        if mlflow_run:
            mlflow.log_artifact(str(output))
    finally:
        if mlflow_run:
            mlflow.end_run()
    return {
        "output": str(output),
        "device": str(device),
        "samples": len(dataset),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/dacl10k_manifest.json")
    parser.add_argument("--output", default="models/dacl10k_tiny_unet.pt")
    parser.add_argument("--number-of-classes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="urban-skin-dacl10k")
    parser.add_argument("--run-name", default="tiny-unet-baseline")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
