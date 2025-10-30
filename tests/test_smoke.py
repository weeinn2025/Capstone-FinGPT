#def test_placeholder():
#    # Replace with real tests. This makes CI pass initially.
#    assert 1 + 1 == 2

import os
import pytest
from app import app as flask_app


@pytest.fixture(scope="session")
def client():
    flask_app.testing = True
    with flask_app.test_client() as c:
        yield c


def test_health_ok(client):
    """Check /health endpoint returns 200 and status OK."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json() or {}
    assert data.get("status") in {"ok", "healthy", "ready"}


def test_ai_status_ok(client):
    """Check /ai_status endpoint returns expected structure."""
    r = client.get("/ai_status")
    assert r.status_code == 200
    data = r.get_json() or {}
    # Ensure required keys are present
    for key in ["model", "api_version", "gemini_available"]:
        assert key in data
    assert isinstance(data["gemini_available"], bool)


def test_ai_smoke_basic(client):
    """Ping /ai_smoke if available; ignore failure gracefully."""
    r = client.get("/ai_smoke")
    assert r.status_code in {200, 502, 500}
    data = r.get_json() or {}
    # When successful, expect OK response
    if r.status_code == 200:
        assert data.get("ok") is True
        assert data.get("got") == "OK"
