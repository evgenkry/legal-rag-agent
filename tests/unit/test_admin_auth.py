"""Тесты авторизации admin endpoints"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


def _clear_settings_cache():
    from src.core.config import get_settings
    get_settings.cache_clear()


def test_admin_ingest_without_api_key_returns_503(monkeypatch):
    """ADMIN_API_KEY не задан -> 503"""
    monkeypatch.setenv("ADMIN_API_KEY", "")
    _clear_settings_cache()
    try:
        client = TestClient(app)
        resp = client.post(
            "/admin/ingest",
            files={"file": ("test.md", b"# Test")},
            data={"source_name": "custom"},
        )
        assert resp.status_code == 503
        assert "ADMIN_API_KEY" in resp.json().get("detail", "")
    finally:
        _clear_settings_cache()


def test_admin_ingest_without_header_returns_401(monkeypatch):
    """X-API-KEY отсутствует (но ADMIN_API_KEY задан) -> 401"""
    monkeypatch.setenv("ADMIN_API_KEY", "secret123")
    _clear_settings_cache()
    try:
        client = TestClient(app)
        resp = client.post(
            "/admin/ingest",
            files={"file": ("test.md", b"# Test")},
            data={"source_name": "custom"},
        )
        assert resp.status_code == 401
        detail = resp.json().get("detail", "").lower()
        assert "x-api-key" in detail or "invalid" in detail
    finally:
        _clear_settings_cache()


def test_admin_ingest_with_wrong_key_returns_401(monkeypatch):
    """Неверный X-API-KEY -> 401"""
    monkeypatch.setenv("ADMIN_API_KEY", "secret123")
    _clear_settings_cache()
    try:
        client = TestClient(app)
        resp = client.post(
            "/admin/ingest",
            files={"file": ("test.md", b"# Test")},
            data={"source_name": "custom"},
            headers={"X-API-KEY": "wrong-key"},
        )
        assert resp.status_code == 401
    finally:
        _clear_settings_cache()


def test_admin_ingest_with_valid_key_returns_200(monkeypatch):
    """Верный X-API-KEY -> 200, status=indexed"""
    monkeypatch.setenv("ADMIN_API_KEY", "secret123")
    _clear_settings_cache()
    try:
        mock_embed = object()
        with (
            patch("src.api.routes.admin.get_embed_model", return_value=mock_embed),
            patch("src.api.routes.admin.index_single_document", return_value=3),
        ):
            client = TestClient(app)
            resp = client.post(
                "/admin/ingest",
                files={"file": ("test_auth.md", b"# Test\n\nContent")},
                data={"source_name": "custom"},
                headers={"X-API-KEY": "secret123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "indexed"
        assert "ingest_id" in data
        assert data["chunks_added"] == 3
    finally:
        _clear_settings_cache()
