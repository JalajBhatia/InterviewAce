"""
Tests for Vercel serverless deployment configuration, routing, and MySQL persistence layer.
"""
from __future__ import annotations

import os
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.index import app
from backend import config, db


def test_api_index_exposes_fastapi_app():
    """Verify Vercel serverless entrypoint api.index exposes the FastAPI app."""
    assert app is not None
    assert app.title == "InterviewAce API"


def test_root_html_redirect_and_json_health():
    """Verify GET / redirects browsers to /app/ while returning JSON for API clients."""
    client = TestClient(app)

    # API client without text/html Accept header receives JSON
    resp_api = client.get("/")
    assert resp_api.status_code == 200
    assert resp_api.headers["content-type"].startswith("application/json")
    data = resp_api.json()
    assert data["status"] in ("ok", "degraded")
    assert "database" in data

    # Browser client requesting text/html gets redirected to /app/
    resp_browser = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"}, follow_redirects=False)
    assert resp_browser.status_code in (302, 307)
    assert resp_browser.headers["location"] == "/app/"


def test_app_without_trailing_slash_redirects():
    """Verify GET /app redirects to /app/."""
    client = TestClient(app)
    resp = client.get("/app", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/app/"


def test_frontend_static_serving_at_app():
    """Verify frontend HTML and assets are served at /app/."""
    client = TestClient(app)
    resp = client.get("/app/")
    assert resp.status_code == 200
    assert "InterviewAce" in resp.text

    resp_css = client.get("/app/style.css")
    assert resp_css.status_code == 200
    assert resp_css.headers["content-type"].startswith("text/css")

    resp_js = client.get("/app/script.js")
    assert resp_js.status_code == 200


def test_db_status_endpoint():
    """Verify /db/status returns valid health telemetry."""
    client = TestClient(app)
    resp = client.get("/db/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "configured" in data
    assert "type" in data


def test_trailing_slashes_and_route_aliases():
    """Verify trailing slashes on critical endpoints do not 404."""
    client = TestClient(app)

    resp_health_slash = client.get("/health/")
    assert resp_health_slash.status_code == 200

    resp_ready_slash = client.get("/ready/")
    assert resp_ready_slash.status_code == 200

    resp_db_slash = client.get("/db/status/")
    assert resp_db_slash.status_code == 200


def test_vercel_serverless_rewrite_path_normalization():
    """
    Simulate Vercel serverless environment where rewrites channel incoming requests
    to api/index.py while passing the original request path in x-matched-path or x-forwarded-uri.
    """
    client = TestClient(app)

    # 1. /health requested through rewrite
    resp_health = client.get("/api/index.py", headers={"x-matched-path": "/health"})
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] in ("ok", "degraded")

    # 2. /app/ requested through rewrite
    resp_app = client.get("/api/index.py", headers={"x-matched-path": "/app/"})
    assert resp_app.status_code == 200
    assert "InterviewAce" in resp_app.text

    # 3. /db/status requested through rewrite
    resp_db = client.get("/api/index.py", headers={"x-matched-path": "/db/status"})
    assert resp_db.status_code == 200
    assert "configured" in resp_db.json()

    # 4. / requested through rewrite by browser
    resp_root_browser = client.get("/api/index.py", headers={"x-matched-path": "/", "Accept": "text/html"}, follow_redirects=False)
    assert resp_root_browser.status_code in (302, 307)
    assert resp_root_browser.headers["location"] == "/app/"

    # 5. / requested through rewrite by API client
    resp_root_api = client.get("/api/index.py", headers={"x-matched-path": "/"})
    assert resp_root_api.status_code == 200
    assert "InterviewAce Backend is running" in resp_root_api.json()["message"]

    # 6. x-forwarded-uri fallback
    resp_fwd = client.get("/api/index.py", headers={"x-forwarded-uri": "/health"})
    assert resp_fwd.status_code == 200

    # 7. Direct /api/index.py invocation without headers maps to root
    resp_direct = client.get("/api/index.py")
    assert resp_direct.status_code == 200

    # 8. /api/ prefixed requests e.g. /api/health
    resp_api_prefix = client.get("/api/health")
    assert resp_api_prefix.status_code == 200


def test_mysql_url_parsing():
    """Verify get_mysql_config parses full connection strings."""
    with patch.dict(os.environ, {"MYSQL_URL": "mysql://dbuser:secretpass@dbhost.example.com:3307/interview_db?ssl=true"}, clear=False):
        from backend.db import get_mysql_config
        cfg = get_mysql_config()
        assert cfg is not None
        assert cfg["host"] == "dbhost.example.com"
        assert cfg["port"] == 3307
        assert cfg["user"] == "dbuser"
        assert cfg["password"] == "secretpass"
        assert cfg["database"] == "interview_db"
        assert cfg["ssl"] is True


def test_mysql_individual_env_parsing():
    """Verify get_mysql_config parses individual MYSQL_* variables."""
    with patch.dict(os.environ, {
        "MYSQL_URL": "",
        "MYSQL_HOST": "sql.example.com",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "testuser",
        "MYSQL_PASSWORD": "mypassword",
        "MYSQL_DATABASE": "testdb",
        "MYSQL_SSL": "false"
    }, clear=False):
        from backend.db import get_mysql_config
        cfg = get_mysql_config()
        assert cfg is not None
        assert cfg["host"] == "sql.example.com"
        assert cfg["user"] == "testuser"
        assert cfg["database"] == "testdb"
        assert cfg["ssl"] is False


def test_serverless_tempdir_fallback():
    """Verify that build_checkpointer handles read-only filesystem gracefully."""
    from backend.graph import build_checkpointer
    # Should build checkpointer cleanly without throwing OSError
    saver = build_checkpointer()
    assert saver is not None
