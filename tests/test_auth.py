"""
tests/test_auth.py
CardioTracker ML v2.2 — APIKeyMiddleware тесттері
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from middleware.auth import APIKeyMiddleware

_TEST_KEY = "test-key-12345"


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware, api_key=_TEST_KEY)

    @app.get("/")
    async def root(): return {"page": "home"}

    @app.get("/health")
    async def health(): return {"status": "ok"}

    @app.post("/ml/risk-classification")
    async def risk(): return {"risk_group": "Норма (A)"}

    @app.post("/ml/check-alerts")
    async def alerts(): return {"alerts": []}

    return app


@pytest.fixture(scope="module")
def client():
    return TestClient(_build_app(), raise_server_exceptions=False)


class TestPublicEndpoints:
    def test_root_no_key(self, client):
        assert client.get("/").status_code == 200

    def test_health_no_key(self, client):
        assert client.get("/health").status_code == 200

    def test_health_with_wrong_key(self, client):
        resp = client.get("/health", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 200


class TestProtectedNoKey:
    def test_risk_no_key_returns_403(self, client):
        assert client.post("/ml/risk-classification", json={}).status_code == 403

    def test_alerts_no_key_returns_403(self, client):
        assert client.post("/ml/check-alerts", json={}).status_code == 403

    def test_missing_key_error_code(self, client):
        body = client.post("/ml/risk-classification", json={}).json()
        assert body["code"] == "MISSING_API_KEY"

    def test_missing_key_mentions_header(self, client):
        body = client.post("/ml/risk-classification", json={}).json()
        assert "X-API-Key" in body["detail"]


class TestInvalidKey:
    def test_wrong_key_returns_403(self, client):
        resp = client.post(
            "/ml/risk-classification", json={},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_empty_key_returns_403(self, client):
        resp = client.post(
            "/ml/risk-classification", json={},
            headers={"X-API-Key": ""},
        )
        assert resp.status_code == 403

    def test_invalid_key_error_code(self, client):
        body = client.post(
            "/ml/risk-classification", json={},
            headers={"X-API-Key": "wrong"},
        ).json()
        assert body["code"] == "INVALID_API_KEY"


class TestValidKey:
    def test_valid_key_passes_middleware(self, client):
        resp = client.post(
            "/ml/risk-classification", json={},
            headers={"X-API-Key": _TEST_KEY},
        )
        assert resp.status_code != 403

    def test_valid_key_gets_200(self, client):
        resp = client.post(
            "/ml/risk-classification", json={},
            headers={"X-API-Key": _TEST_KEY},
        )
        assert resp.status_code == 200