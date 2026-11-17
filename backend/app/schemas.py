"""Pydantic contracts exposed by the inference API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    device: str


class ModelMetadata(BaseModel):
    name: str
    architecture: str
    task: str
    model_loaded: bool
    model_version: str
    classes: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    filename: str
    model_version: str
    latency_ms: float
    detections: list[dict[str, object]] = Field(default_factory=list)
    condition_score: float | None = None
    priority: str | None = None
