"""PyTorch dataset for DACL10K polygon segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class Dacl10kSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        image_size: int = 256,
        max_samples: int | None = None,
    ) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parents[1] / "dacl10k"
        self.root = root
        self.label_to_id: dict[str, int] = manifest["label_to_id"]
        self.records = [record for record in manifest["records"] if record["split"] == split]
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image_path = self.root / record["image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.new("L", (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(mask)
        for annotation in record["annotations"]:
            points = annotation["points"][0]
            scaled = [
                (point[0] * self.image_size, point[1] * self.image_size) for point in points
            ]
            draw.polygon(scaled, fill=self.label_to_id[annotation["label"]])
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.int64)
        return (
            torch.from_numpy(image_array).permute(2, 0, 1),
            torch.from_numpy(mask_array),
        )
