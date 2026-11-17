"""FastAPI application for Urban Skin."""

import os
import platform

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import InferenceService
from .schemas import HealthResponse, ModelMetadata, PredictionResponse

app = FastAPI(
    title="Urban Skin API",
    version="0.1.0",
    description="Damage perception and explainable infrastructure-priority service.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = InferenceService()


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=service.model_loaded,
        model_version=service.model_version,
        device=os.getenv("DEVICE", platform.processor() or "cpu"),
    )


@app.get("/model/metadata", response_model=ModelMetadata, tags=["model"])
async def model_metadata() -> ModelMetadata:
    return service.metadata()


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(image: UploadFile) -> PredictionResponse:
    return await service.predict(image)
