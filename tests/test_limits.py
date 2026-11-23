"""Rate limiting and upload bounds, exercised over HTTP."""

import io

import pytest
from fastapi.testclient import TestClient

from backend import limits
from backend.config import MAX_RESUME_BYTES
from backend.main import app


@pytest.fixture(autouse=True)
def clear_buckets():
    limits.reset()
    yield
    limits.reset()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_reports_readiness_not_just_liveness(client):
    """A process that is up but unconfigured should say so, not report ok."""
    body = client.get("/").json()
    assert body["status"] in ("ok", "degraded")
    assert "ready" in body and "api_keys_configured" in body


def test_requests_under_the_limit_pass(client):
    for _ in range(limits.LIMITS["/detect_role"] - 1):
        # 400 is fine - the point is that it is not 429.
        assert client.post("/detect_role", json={"goal_statement": "x"}).status_code != 429


def test_exceeding_the_limit_returns_429_with_retry_after(client):
    limit = limits.LIMITS["/detect_role"]
    for _ in range(limit):
        client.post("/detect_role", json={"goal_statement": "x"})

    response = client.post("/detect_role", json={"goal_statement": "x"})
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0


def test_limits_are_per_path_not_global(client):
    """Exhausting one endpoint must not lock the user out of the others."""
    for _ in range(limits.LIMITS["/detect_role"] + 2):
        client.post("/detect_role", json={"goal_statement": "x"})

    assert client.get("/").status_code == 200


def test_limits_are_per_client(client):
    limit = limits.LIMITS["/detect_role"]
    for _ in range(limit + 1):
        client.post("/detect_role", json={"goal_statement": "x"},
                    headers={"X-Forwarded-For": "10.0.0.1"})

    other = client.post("/detect_role", json={"goal_statement": "x"},
                        headers={"X-Forwarded-For": "10.0.0.2"})
    assert other.status_code != 429


def test_unlimited_paths_are_not_throttled(client):
    for _ in range(60):
        assert client.get("/").status_code == 200


def test_oversized_upload_is_rejected(client):
    payload = b"%PDF-1.4\n" + b"x" * (MAX_RESUME_BYTES + 1024)
    response = client.post(
        "/upload_resume",
        files={"file": ("big.pdf", io.BytesIO(payload), "application/pdf")},
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_normal_upload_is_accepted(client):
    response = client.post(
        "/upload_resume",
        files={"file": ("cv.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    # Unsupported format, but it got past the size gate - which is what matters.
    assert response.status_code == 200
    assert response.json()["warnings"]


def test_cors_does_not_advertise_a_wildcard_with_credentials(client):
    """allow_origins=* plus allow_credentials is rejected by browsers as invalid."""
    response = client.get("/", headers={"Origin": "https://evil.example"})
    assert response.headers.get("access-control-allow-origin") != "*"
