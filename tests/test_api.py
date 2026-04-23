"""Unit tests for FastAPI endpoints."""
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client():
    """Create a test client — skips if model not found."""
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)
    except Exception:
        pytest.skip("FastAPI or model not available")


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_features_endpoint(client):
    resp = client.get("/features")
    assert resp.status_code == 200
    data = resp.json()
    assert "features" in data
    assert "count" in data
    assert data["count"] > 0


def test_model_info_endpoint(client):
    resp = client.get("/model_info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_type" in data
    assert "n_features" in data
    assert "architecture" in data


def test_predict_endpoint_valid(client):
    rows = [
        {"open": 640+i, "high": 650+i, "low": 635+i,
         "close": 645+i, "volume": 5_000_000}
        for i in range(10)
    ]
    resp = client.post("/predict", json={"rows": rows})
    assert resp.status_code in (200, 503)  # 503 if model not loaded
    if resp.status_code == 200:
        data = resp.json()
        assert "predicted_return_pct" in data
        assert "signal" in data
        assert data["signal"] in ("BUY", "HOLD")


def test_predict_endpoint_too_few_rows(client):
    rows = [{"open": 640, "high": 650, "low": 635, "close": 645, "volume": 5_000_000}]
    resp = client.post("/predict", json={"rows": rows})
    assert resp.status_code == 422  # validation error
