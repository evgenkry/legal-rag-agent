"""Интеграционный тест POST /admin/ingest (с моком индексации)"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def admin_client(monkeypatch):
    """Клиент с включённым admin API."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-secret")

    from src.core.config import get_settings
    get_settings.cache_clear()
    try:
        yield TestClient(app)
    finally:
        get_settings.cache_clear()


def test_ingest_success_returns_200_and_indexed(admin_client):
    """Успешный ingest -> 200, status=indexed, chunks_added."""
    mock_embed = object()
    with (
        patch("src.api.routes.admin.get_embed_model", return_value=mock_embed),
        patch("src.api.routes.admin.index_single_document", return_value=5),
    ):
        resp = admin_client.post(
            "/admin/ingest",
            files={"file": ("test_doc.md", b"# Test\n\nContent")},
            data={"source_name": "custom"},
            headers={"X-API-KEY": "test-secret"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "indexed"
    assert data["ingest_id"]
    assert data["source"] == "custom"
    assert data["filename"] == "test_doc.md"
    assert data["chunks_added"] == 5


def test_ingest_indexing_error_returns_200_failed_indexing(admin_client):
    """Ошибка индексации -> 200, status=failed_indexing, error заполнен."""
    mock_embed = object()
    with (
        patch("src.api.routes.admin.get_embed_model", return_value=mock_embed),
        patch("src.api.routes.admin.index_single_document", side_effect=RuntimeError("DB error")),
    ):
        resp = admin_client.post(
            "/admin/ingest",
            files={"file": ("fail.md", b"# Fail")},
            data={"source_name": "custom"},
            headers={"X-API-KEY": "test-secret"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed_indexing"
    assert "error" in data
    assert "DB error" in data["error"]
    assert data["chunks_added"] == 0


def test_ingest_timeout_returns_200_failed_indexing(admin_client):
    """Таймаут индексации -> 200, status=failed_indexing, error про timeout."""
    import asyncio

    async def timeout_raiser(*args, **kwargs):
        raise asyncio.TimeoutError()

    mock_embed = object()
    with (
        patch("src.api.routes.admin.get_embed_model", return_value=mock_embed),
        patch("src.api.routes.admin.asyncio.wait_for", side_effect=timeout_raiser),
    ):
        resp = admin_client.post(
            "/admin/ingest",
            files={"file": ("slow.md", b"# Slow")},
            data={"source_name": "custom"},
            headers={"X-API-KEY": "test-secret"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed_indexing"
    assert "error" in data
    assert "timeout" in data["error"].lower() or "timed out" in data["error"].lower()
