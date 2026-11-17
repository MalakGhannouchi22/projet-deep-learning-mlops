"""Dataset loader for normalized polygon or bitmap-mask manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class UniversalSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        image_size: int = 128,
        max_samples: int | None = None,
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.root = Path.cwd()
        self.label_to_id = manifest["label_to_id"]
        self.records = [record for record in manifest["records"] if record["split"] == split]
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(self.root / record["image"]) as source:
            image = source.convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if record.get("mask"):
            with Image.open(self.root / record["mask"]) as source_mask:
                mask = source_mask.convert("L").resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST
                )
            mask_array = np.asarray(mask, dtype=np.int64)
            if mask_array.max() > len(self.label_to_id):
                mask_array = (mask_array > 0).astype(np.int64)
        else:
            mask = Image.new("L", (self.image_size, self.image_size), 0)
            draw = ImageDraw.Draw(mask)
            for annotation in record["annotations"]:
                points = annotation["points"][0]
                scaled = [(x * self.image_size, y * self.image_size) for x, y in points]
                draw.polygon(scaled, fill=self.label_to_id[annotation["label"]])
            mask_array = np.asarray(mask, dtype=np.int64)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image_array).permute(2, 0, 1), torch.from_numpy(mask_array)
