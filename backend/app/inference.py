"""Model loading and inference boundary.

The service deliberately fails clearly until a trained ONNX artifact is supplied.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from urban_skin.decision.health import condition_score, priority

from .schemas import ModelMetadata, PredictionResponse


class InferenceService:
    def __init__(self) -> None:
        self.model_path = Path(os.getenv("MODEL_PATH", "models/urban_skin.onnx"))
        self.model_version = os.getenv("MODEL_VERSION", "untrained")
        self.classes = self._load_classes()
        self._torch_model = None

    def _load_classes(self) -> list[str]:
        manifest_path = Path("data/processed/dacl10k_manifest.json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return ["background", *manifest["classes"]]
        return ["background", "Hollowareas", "Spalling", "Crack", "Rust"]

    @property
    def model_loaded(self) -> bool:
        return self.model_path.is_file()

    def _load_torch_model(self):
        if self._torch_model is None and self.model_path.suffix == ".pt":
            import torch

            from urban_skin.models.segmentation import build_segmentation_model

            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            model = build_segmentation_model(
                checkpoint.get("architecture", "tiny_unet"),
                checkpoint["number_of_classes"],
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self._torch_model = (model, checkpoint["image_size"])
        return self._torch_model

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="urban-skin",
            architecture=(
                "TinyUNet PyTorch baseline"
                if self.model_path.suffix == ".pt"
                else "ONNX runtime adapter"
            ),
            task="semantic segmentation / damage detection",
            model_loaded=self.model_loaded,
            model_version=self.model_version,
            classes=self.classes,
        )

    async def predict(self, image: UploadFile) -> PredictionResponse:
        if not self.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Model artifact is not available. Train or place an ONNX model "
                    "at MODEL_PATH."
                ),
            )
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JPEG, PNG, and WebP images are accepted.",
            )
        contents = await image.read()
        try:
            with Image.open(io.BytesIO(contents)) as decoded:
                decoded.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is not a valid image.",
            ) from exc

        if self.model_path.suffix != ".pt":
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="ONNX inference integration is the next packaging milestone.",
            )
        import torch

        model, image_size = self._load_torch_model()
        with Image.open(io.BytesIO(contents)) as decoded:
            resized = decoded.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        started = time.perf_counter()
        with torch.no_grad():
            prediction = model(tensor).argmax(dim=1)[0].numpy()
        latency_ms = (time.perf_counter() - started) * 1000
        damage_fraction = float((prediction > 0).mean())
        score = condition_score(damage_fraction, confidence=1.0)
        return PredictionResponse(
            filename=image.filename or "upload",
            model_version=self.model_version,
            latency_ms=round(latency_ms, 3),
            detections=[],
            condition_score=score,
            priority=priority(score),
        )
