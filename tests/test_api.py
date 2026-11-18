from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app, service

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_model_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "model_path", Path("__missing_test_model__.onnx"))


def test_health_reports_explicit_untrained_state() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False


def test_model_metadata_is_typed_and_documented() -> None:
    response = client.get("/model/metadata")
    assert response.status_code == 200
    assert response.json()["task"] == "semantic segmentation / damage detection"


def test_predict_returns_service_unavailable_without_weights() -> None:
    response = client.post(
        "/predict",
        files={"image": ("road.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 503
    assert "ONNX model" in response.json()["detail"]
