"""
test_api.py — API integration tests using FastAPI TestClient.
"""
import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

API_KEY = "bg-dev-key-2026"
AUTH_HEADERS = {"X-API-Key": API_KEY}

SCORE_PAYLOAD = {
    "msisdn": "254722000001",
    "time_since_swap_min": 10,
    "amount": 10000,
    "balance_before": 12000,
}


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_without_api_key_returns_401():
    response = client.post("/score", json=SCORE_PAYLOAD)
    assert response.status_code == 401


def test_score_with_valid_api_key_returns_200():
    response = client.post("/score", json=SCORE_PAYLOAD, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "action" in data


def test_demo_fraud_returns_200_and_high_score():
    response = client.get("/demo/fraud")
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] >= 81


def test_demo_legit_returns_200_and_low_score():
    response = client.get("/demo/legit")
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] <= 20


def test_stats_returns_200():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
